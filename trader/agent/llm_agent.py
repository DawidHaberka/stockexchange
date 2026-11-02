"""Agent LLM: dla tickerów z triggerem pyta Claude o decyzję w ściśle określonym JSON.

Claude jest wywoływany przez Claude Code CLI (`claude -p`) na subskrypcji — bez płatnego API.
- format wymuszony schematem JSON (`--json-schema`) + walidacja pydantic jako druga warstwa,
- niepoprawna odpowiedź / odmowa / błąd CLI => HOLD,
- każde wywołanie zapisywane w llm_calls (tokeny, równoważnik ceny API, model faktycznie użyty).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from ..config import AppConfig
from ..data.snapshot import MarketSnapshot
from ..storage.db import Database, iso, parse_iso
from .base import PortfolioView, RunContext
from .claude_cli import CLIResult
from .context import build_ticker_context, cooldown_until
from .schemas import DECISION_JSON_SCHEMA, AgentDecision, hold, parse_decision
from .triggers import TickerHistory, evaluate_triggers

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Wywołanie Claude się nie powiodło (limit, błąd, timeout) w trybie ścisłym (backtest)."""


class LLMAgent:
    name = "llm"

    def __init__(self, cli, cfg: AppConfig, db: Database, state_key: str = "llm", fail_on_error: bool = False):
        self.cli = cli            # ClaudeCLI (lub atrapa w testach) z metodą ask(prompt, schema)
        self.cfg = cfg
        self.db = db
        # osobny stan triggerów dla dry-run i live, żeby dry-run nie „zużył” first_evaluation
        self.state_key = state_key
        # backtest: błąd wywołania przerywa przebieg (żeby nie wstawiać wymuszonych HOLD-ów do wyników)
        self.fail_on_error = fail_on_error

    # ------------------------------------------------------------------ public
    def decide(self, snapshot: MarketSnapshot, portfolio: PortfolioView, ctx: RunContext) -> list[AgentDecision]:
        results: list[AgentDecision | None] = [None] * len(self.cfg.watchlist)
        pending: list[tuple[int, str, str, list[str]]] = []     # (indeks, ticker, prompt, triggery)
        prices: dict[str, float] = {}
        for i, sym in enumerate(self.cfg.watchlist):
            snap = snapshot.get(sym)
            if snap is None or not snap.data_ok:
                problems = "; ".join(snap.problems) if snap else "brak danych"
                results[i] = AgentDecision(hold(sym, f"HOLD: niekompletne dane ({problems})"), "no_data")
                continue
            prices[sym] = snap.price
            hist = self._history(sym)
            pos = portfolio.position(sym)
            triggers = evaluate_triggers(snap, hist, pos, portfolio.stop_levels.get(sym), self.cfg.triggers)
            if not triggers:
                log.info("[llm] %s: no trigger -> HOLD bez wywołania modelu", sym)
                results[i] = AgentDecision(hold(sym, "no trigger"), "no_trigger")
                continue
            cd = cooldown_until(portfolio.last_trade_at.get(sym), self.cfg.risk.cooldown_minutes)
            prompt = build_ticker_context(snap, portfolio, ctx, self.cfg, triggers,
                                          news_since=hist.last_run_at if hist else None, cooldown_until=cd)
            pending.append((i, sym, prompt, triggers))

        # wywołania modelu (równolegle, jeśli llm.max_parallel > 1); zapis do bazy w wątku głównym
        workers = min(self.cfg.llm.max_parallel, len(pending))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                answers = list(pool.map(lambda p: self.cli.ask(p[2], DECISION_JSON_SCHEMA), pending))
        else:
            answers = [self.cli.ask(p[2], DECISION_JSON_SCHEMA) for p in pending]
        for (i, sym, prompt, triggers), res in zip(pending, answers, strict=True):
            results[i] = self._record(sym, prompt, triggers, ctx, res)

        failed = [r for r in results if r is not None and r.source == "llm_error"]
        if failed and self.fail_on_error:
            raise LLMUnavailable(f"{len(failed)} wywołań Claude nieudanych, np. {failed[0].validation_error}")
        for r in results:
            if r is not None and r.ticker in prices:
                # błąd wywołania nie liczy się jako ocena — ticker zostanie oceniony ponownie
                called = r.source not in ("no_trigger", "llm_error")
                self.db.update_ticker_state(self.state_key, r.ticker, ctx.now, prices[r.ticker], called)
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------ internals
    def _history(self, sym: str) -> TickerHistory | None:
        row = self.db.get_ticker_state(self.state_key, sym)
        if row is None:
            return None
        return TickerHistory(
            last_run_at=parse_iso(row["last_run_at"]), last_price=row["last_price"],
            last_llm_at=parse_iso(row["last_llm_at"]), last_llm_price=row["last_llm_price"],
        )

    def _record(self, sym: str, prompt: str, triggers: list[str], ctx: RunContext, res: CLIResult) -> AgentDecision:
        llm = self.cfg.llm
        raw = json.dumps(res.structured, ensure_ascii=False) if res.structured is not None else res.text
        call_id = self.db.insert_llm_call({
            "run_id": ctx.run_id, "ticker": sym, "created_at": iso(ctx.now),
            "model_requested": llm.model, "model_served": res.model_served, "effort": llm.effort,
            "prompt_hash": llm.prompt_hash, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cache_creation_tokens": res.cache_creation_tokens, "cache_read_tokens": res.cache_read_tokens,
            "cost_usd": res.cost_equiv_usd, "latency_ms": res.duration_ms, "stop_reason": res.stop_reason,
            "request_id": res.session_id, "error": res.error, "user_prompt": prompt, "raw_response": raw,
        })

        if not res.ok:
            log.error("[llm] %s: błąd wywołania Claude (%s) -> HOLD", sym, res.error)
            return AgentDecision(hold(sym, f"HOLD: błąd wywołania Claude ({res.error})"[:500]), "llm_error",
                                 triggers, call_id, res.error)
        if res.stop_reason == "refusal":
            log.warning("[llm] %s: model odmówił -> HOLD", sym)
            return AgentDecision(hold(sym, "HOLD: odmowa modelu"), "refusal", triggers, call_id, "refusal")
        if res.stop_reason == "max_tokens":
            log.warning("[llm] %s: odpowiedź ucięta (max_tokens) -> HOLD", sym)
            return AgentDecision(hold(sym, "HOLD: odpowiedź ucięta (max_tokens)"), "invalid", triggers, call_id,
                                 "max_tokens")

        decision, err = parse_decision(raw, sym)
        if err:
            log.warning("[llm] %s: niepoprawna odpowiedź (%s) -> HOLD", sym, err)
            return AgentDecision(decision, "invalid", triggers, call_id, err)
        log.info("[llm] %s: %s size=%.1f%% conf=%.2f | %s | model %s, równoważnik API $%.4f", sym,
                 decision.action.value, decision.size_pct, decision.confidence, decision.reasoning,
                 res.model_served, res.cost_equiv_usd)
        return AgentDecision(decision, "llm", triggers, call_id)
