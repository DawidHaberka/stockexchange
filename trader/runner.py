"""Orkiestracja jednego przebiegu (`main.py --run` / `--dry-run`).

dane (wspólne) -> dla każdego agenta: rekoncyliacja -> limit dziennej straty ->
decyzje -> risk manager -> zapis -> wykonanie (lub dry-run) -> stan portfela.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .agent.base import PortfolioView, RunContext
from .agent.llm_agent import LLMAgent
from .agent.schemas import AgentDecision, hold
from .agent.sma_agent import SMAAgent
from .broker import NY, Broker
from .config import AppConfig
from .data.market_data import MarketDataSource
from .data.snapshot import MarketSnapshot, build_snapshot
from .execution.executor import ExecutionResult, Executor
from .risk.manager import RiskManager, RiskVerdict
from .scheduler import evaluate_schedule
from .storage.db import Database, iso, utcnow

log = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    agent: str
    equity_start: float | None = None
    equity_end: float | None = None
    daily_change_pct: float | None = None
    halted: bool = False
    reconcile_notes: list[str] = field(default_factory=list)
    verdicts: list[RiskVerdict] = field(default_factory=list)
    executions: list[ExecutionResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunSummary:
    status: str                       # ok | skipped | error
    reason: str
    mode: str
    run_id: int | None = None
    slot: datetime | None = None
    agents: dict[str, AgentRunResult] = field(default_factory=dict)
    data_problems: dict[str, list[str]] = field(default_factory=dict)
    llm_calls: int = 0
    llm_cost_equiv_usd: float = 0.0     # równoważnik ceny API; przy subskrypcji nie jest pobierany


class Runner:
    def __init__(self, cfg: AppConfig, db: Database, brokers: dict[str, Broker], market_data: MarketDataSource,
                 llm_cli=None, now_fn: Callable[[], datetime] = utcnow,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.db = db
        self.brokers = brokers
        self.md = market_data
        self.llm_cli = llm_cli
        self.now_fn = now_fn
        self.sleep = sleep

    # ------------------------------------------------------------------ main entry
    def run(self, dry_run: bool, force: bool = False, agents: list[str] | None = None) -> RunSummary:
        agents = [a for a in (agents or ["llm", "sma"]) if a in self.brokers]
        if not agents:
            return RunSummary("error", "brak skonfigurowanych kont dla wybranych agentów", "dry_run")

        live = (not dry_run) and self.cfg.execution.orders_enabled
        if not dry_run and not self.cfg.execution.orders_enabled:
            log.warning("execution.orders_enabled=false w config.yaml — przebieg w trybie DRY-RUN (bez zleceń).")
        mode = "live" if live else "dry_run"
        if force and live:
            return RunSummary("error", "--force jest dozwolone tylko z --dry-run", mode)

        now = self.now_fn()
        ref = self.brokers[agents[0]]
        clock = ref.get_clock()
        today_ny = now.astimezone(NY).date()
        session = ref.get_session(today_ny)
        sched = evaluate_schedule(now, clock, session, self.cfg.experiment,
                                  lambda slot: self.db.slot_processed(slot, mode))
        if not sched.due:
            if not force:
                log.info("Pomijam przebieg: %s", sched.reason)
                return RunSummary("skipped", sched.reason, mode, slot=sched.slot_start)
            log.warning("FORCE (dry-run): ignoruję harmonogram (%s)", sched.reason)

        session_date = today_ny.isoformat()
        config_hash = self.db.register_config(self.cfg)
        run_id = self.db.start_run(session_date, sched.slot_start if sched.due else None, mode, config_hash)
        summary = RunSummary("ok", sched.reason, mode, run_id, sched.slot_start)
        log.info("=== Przebieg #%d (%s) | %s | wariant: model=%s effort=%s freq=%s prompt=%s config=%s ===",
                 run_id, mode, sched.reason, self.cfg.llm.model, self.cfg.llm.effort,
                 self.cfg.experiment.frequency, self.cfg.llm.prompt_hash, config_hash)

        errors: list[str] = []
        try:
            snapshot = build_snapshot(self.md, self.cfg, now)
            summary.data_problems = {s: t.problems for s, t in snapshot.tickers.items() if t.problems}
            ctx = RunContext(run_id, now, session.open if session else None, session.close if session else None)
            for name in agents:
                res = AgentRunResult(name)
                summary.agents[name] = res
                try:
                    self._run_agent(name, snapshot, ctx, session_date, live, res)
                except Exception as exc:
                    log.exception("Agent %s: błąd przebiegu", name)
                    res.error = f"{type(exc).__name__}: {exc}"
                    errors.append(f"{name}: {res.error}")
        except Exception as exc:
            log.exception("Błąd przebiegu")
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            status = "error" if errors else "ok"
            self.db.finish_run(run_id, status, "; ".join(errors) or None)
            summary.status = status
            if errors:
                summary.reason = "; ".join(errors)
            row = self.db.query_one("SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd), 0) AS c FROM llm_calls "
                                    "WHERE run_id = ?", (run_id,))
            summary.llm_calls, summary.llm_cost_equiv_usd = int(row["n"]), float(row["c"])
        return summary

    # ------------------------------------------------------------------ per agent
    def _make_agent(self, name: str, live: bool):
        if name == "llm":
            if self.llm_cli is None:
                raise RuntimeError("Brak Claude Code CLI dla agenta LLM")
            return LLMAgent(self.llm_cli, self.cfg, self.db, state_key="llm" if live else "llm_dry")
        if name == "sma":
            return SMAAgent(self.cfg)
        raise ValueError(f"Nieznany agent: {name}")

    def _run_agent(self, name: str, snapshot: MarketSnapshot, ctx: RunContext, session_date: str, live: bool,
                   res: AgentRunResult) -> None:
        broker = self.brokers[name]
        risk = RiskManager(self.cfg, asset_lookup=broker.get_asset)
        executor = Executor(broker, self.db, self.cfg, name, dry_run=not live, sleep=self.sleep)

        account = broker.get_account()
        positions = broker.get_positions()
        res.equity_start = account.equity
        self.db.insert_snapshot(name, ctx.run_id, session_date, "run_start", account, positions)

        if live and self.db.get_experiment_start(name) is None:
            spy = snapshot.get(self.cfg.benchmark_symbol)
            # znacznik czasu z zegara bazy (spójny z taken_at/created_at innych tabel)
            self.db.set_experiment_start(name, utcnow(), session_date, account.equity, account.account_number,
                                         self.cfg.benchmark_symbol, spy.price if spy else None)
            log.info("[%s] START EKSPERYMENTU: kapitał startowy %.2f USD (z konta), %s @ %s", name,
                     account.equity, self.cfg.benchmark_symbol, spy.price if spy else "n/a")

        res.reconcile_notes = executor.reconcile(ctx.run_id, snapshot)
        positions = broker.get_positions()
        account = broker.get_account()

        portfolio = PortfolioView(
            agent=name, account=account, positions=positions,
            stop_levels={s: lvl for s in positions if (lvl := self.db.get_stop_level(name, s)) is not None},
            open_orders=broker.get_open_orders(),
            last_trade_at={s: t for s in self.cfg.watchlist if (t := self.db.last_trade_time(name, s)) is not None},
        )

        # dzienny limit straty — sprawdzany na początku każdego przebiegu
        halted, halt_reason = self.db.is_halted(name, session_date)
        breached, pct = risk.daily_loss_breached(account)
        res.daily_change_pct = pct
        if breached and not halted:
            halt_reason = f"equity {pct:+.2f}% vs zamknięcie poprzedniej sesji (limit -{self.cfg.risk.daily_loss_limit_pct:g}%)"
            self.db.set_halted(name, session_date, halt_reason)
            halted = True
            log.warning("[%s] DZIENNY LIMIT STRATY PRZEKROCZONY: %s — brak handlu do następnej sesji", name,
                        halt_reason)
        res.halted = halted

        if halted:
            decisions = [AgentDecision(hold(s, f"HOLD: handel wstrzymany — {halt_reason}"), "halted")
                         for s in self.cfg.watchlist]
        else:
            decisions = self._make_agent(name, live).decide(snapshot, portfolio, ctx)

        verdicts = risk.review(decisions, portfolio, snapshot, ctx.now, halted, halt_reason)
        res.verdicts = verdicts
        decision_ids = [self._persist_decision(name, ctx, v, live) for v in verdicts]

        res.executions = executor.execute(ctx.run_id, verdicts, decision_ids)

        account_end = broker.get_account()
        positions_end = broker.get_positions()
        res.equity_end = account_end.equity
        self.db.insert_snapshot(name, ctx.run_id, session_date, "run_end", account_end, positions_end)

    def _persist_decision(self, agent: str, ctx: RunContext, v: RiskVerdict, live: bool) -> int:
        ad = v.agent_decision
        d = ad.decision
        variant = self.cfg.variant()
        if agent == "sma":
            model = f"sma{self.cfg.sma.fast}/{self.cfg.sma.slow}-1h"
            effort = prompt_hash = None
        else:
            model, effort, prompt_hash = variant["model"], variant["effort"], variant["prompt_hash"]
        it = v.intent
        did = self.db.insert_decision({
            "run_id": ctx.run_id, "agent": agent, "ticker": d.ticker, "created_at": iso(utcnow()),
            "action": d.action.value, "size_pct": d.size_pct, "confidence": d.confidence, "reasoning": d.reasoning,
            "source": ad.source, "triggers": ad.triggers, "llm_call_id": ad.llm_call_id,
            "validation_error": ad.validation_error, "risk_status": v.status, "risk_reason": v.reason,
            "order_side": it.side if it else None, "order_qty": it.qty if it else None,
            "limit_price": it.limit_price if it else None, "stop_price": it.stop_price if it else None,
            "notional": it.notional if it else None, "dry_run": 0 if live else 1,
            "config_hash": variant["config_hash"], "model": model, "effort": effort,
            "frequency": variant["frequency"], "prompt_hash": prompt_hash,
        })
        log.info("[%s] decyzja %s %s (%s, size=%.1f%%, conf=%.2f) -> risk: %s — %s | uzasadnienie: %s",
                 agent, d.ticker, d.action.value, ad.source, d.size_pct, d.confidence, v.status, v.reason,
                 d.reasoning)
        return did
