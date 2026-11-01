"""Agent LLM: dla tickerów z triggerem pyta Claude o decyzję w ściśle określonym JSON.

- structured outputs (output_config.format) + walidacja pydantic jako druga warstwa,
- niepoprawna odpowiedź / odmowa / błąd API => HOLD,
- każde wywołanie zapisywane w llm_calls (tokeny, koszt USD, model faktycznie użyty).
"""

from __future__ import annotations

import logging
import time

import anthropic

from ..config import AppConfig, Pricing
from ..data.snapshot import MarketSnapshot
from ..storage.db import Database, iso, parse_iso, utcnow
from .base import PortfolioView, RunContext
from .context import build_ticker_context, cooldown_until
from .schemas import DECISION_JSON_SCHEMA, AgentDecision, hold, parse_decision
from .triggers import TickerHistory, evaluate_triggers

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def compute_cost_usd(pricing: Pricing | None, usage) -> float | None:
    if pricing is None or usage is None:
        return None
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (inp * pricing.input + out * pricing.output + cw * pricing.cache_write + cr * pricing.cache_read) / 1e6


def make_anthropic_client(cfg: AppConfig, api_key: str | None) -> anthropic.Anthropic:
    if not api_key:
        raise RuntimeError("Brak ANTHROPIC_API_KEY w .env")
    return anthropic.Anthropic(api_key=api_key, timeout=cfg.llm.timeout_s, max_retries=cfg.llm.max_retries)


class LLMAgent:
    name = "llm"

    def __init__(self, client, cfg: AppConfig, db: Database, state_key: str = "llm"):
        self.client = client
        self.cfg = cfg
        self.db = db
        # osobny stan triggerów dla dry-run i live, żeby dry-run nie „zużył” first_evaluation
        self.state_key = state_key

    # ------------------------------------------------------------------ public
    def decide(self, snapshot: MarketSnapshot, portfolio: PortfolioView, ctx: RunContext) -> list[AgentDecision]:
        results: list[AgentDecision] = []
        for sym in self.cfg.watchlist:
            snap = snapshot.get(sym)
            if snap is None or not snap.data_ok:
                problems = "; ".join(snap.problems) if snap else "brak danych"
                results.append(AgentDecision(hold(sym, f"HOLD: niekompletne dane ({problems})"), "no_data"))
                continue

            hist = self._history(sym)
            pos = portfolio.position(sym)
            triggers = evaluate_triggers(snap, hist, pos, portfolio.stop_levels.get(sym), self.cfg.triggers)
            llm_called = False
            if not triggers:
                log.info("[llm] %s: no trigger -> HOLD bez wywołania API", sym)
                results.append(AgentDecision(hold(sym, "no trigger"), "no_trigger"))
            else:
                cd = cooldown_until(portfolio.last_trade_at.get(sym), self.cfg.risk.cooldown_minutes)
                prompt = build_ticker_context(
                    snap, portfolio, ctx, self.cfg, triggers,
                    news_since=hist.last_run_at if hist else None, cooldown_until=cd,
                )
                result = self._ask(sym, prompt, triggers, ctx)
                results.append(result)
                # błąd API nie liczy się jako ocena — ticker zostanie oceniony ponownie
                llm_called = result.source != "api_error"
            self.db.update_ticker_state(self.state_key, sym, ctx.now, snap.price, llm_called)
        return results

    # ------------------------------------------------------------------ internals
    def _history(self, sym: str) -> TickerHistory | None:
        row = self.db.get_ticker_state(self.state_key, sym)
        if row is None:
            return None
        return TickerHistory(
            last_run_at=parse_iso(row["last_run_at"]), last_price=row["last_price"],
            last_llm_at=parse_iso(row["last_llm_at"]), last_llm_price=row["last_llm_price"],
        )

    def _request(self, prompt: str):
        llm = self.cfg.llm
        params = dict(
            model=llm.model,
            max_tokens=llm.max_tokens,
            # stały prompt systemowy na początku => cache między tickerami w jednym przebiegu
            system=[{"type": "text", "text": llm.system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": llm.effort,
                "format": {"type": "json_schema", "schema": DECISION_JSON_SCHEMA},
            },
        )
        if llm.refusal_fallback:
            return self.client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
        return self.client.messages.create(**params)

    def _ask(self, sym: str, prompt: str, triggers: list[str], ctx: RunContext) -> AgentDecision:
        llm = self.cfg.llm
        call = {
            "run_id": ctx.run_id, "ticker": sym, "created_at": iso(utcnow()),
            "model_requested": llm.model, "effort": llm.effort, "prompt_hash": llm.prompt_hash,
            "user_prompt": prompt,
        }
        t0 = time.monotonic()
        try:
            resp = self._request(prompt)
        except anthropic.APIError as exc:
            call.update(latency_ms=int((time.monotonic() - t0) * 1000), error=f"{type(exc).__name__}: {exc}",
                        cost_usd=0.0)
            call_id = self.db.insert_llm_call(call)
            log.error("[llm] %s: błąd API (%s) -> HOLD", sym, type(exc).__name__)
            return AgentDecision(hold(sym, f"HOLD: błąd API Claude ({type(exc).__name__})"), "api_error",
                                 triggers, call_id, str(exc)[:500])

        latency = int((time.monotonic() - t0) * 1000)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        served = getattr(resp, "model", None) or llm.model
        usage = getattr(resp, "usage", None)
        pricing = self.cfg.pricing_for(served) or self.cfg.pricing_for(llm.model)
        cost = compute_cost_usd(pricing, usage)
        if cost is None:
            log.warning("[llm] brak cennika dla modelu %s — koszt zapisany jako 0", served)
        stop_reason = getattr(resp, "stop_reason", None)
        call.update(
            model_served=served, latency_ms=latency, stop_reason=stop_reason,
            request_id=getattr(resp, "_request_id", None), raw_response=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cost_usd=cost or 0.0,
        )
        call_id = self.db.insert_llm_call(call)

        if stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            log.warning("[llm] %s: model odmówił (kategoria=%s) -> HOLD", sym, cat)
            return AgentDecision(hold(sym, f"HOLD: odmowa modelu (kategoria: {cat})"), "refusal", triggers, call_id,
                                 f"refusal:{cat}")
        if stop_reason == "max_tokens":
            log.warning("[llm] %s: odpowiedź ucięta (max_tokens) -> HOLD", sym)
            return AgentDecision(hold(sym, "HOLD: odpowiedź ucięta (max_tokens)"), "invalid", triggers, call_id,
                                 "max_tokens")

        decision, err = parse_decision(text, sym)
        if err:
            log.warning("[llm] %s: niepoprawna odpowiedź (%s) -> HOLD", sym, err)
            return AgentDecision(decision, "invalid", triggers, call_id, err)
        log.info("[llm] %s: %s size=%.1f%% conf=%.2f | %s | koszt $%.4f", sym, decision.action.value,
                 decision.size_pct, decision.confidence, decision.reasoning, cost or 0.0)
        return AgentDecision(decision, "llm", triggers, call_id)

