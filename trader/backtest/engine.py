"""Silnik backtestu portfelowego (replay) — ten sam kod co na żywo, inny „świat”.

Dla każdej sesji i każdego slotu (10:00, 11:00, ... zgodnie z harmonogramem live):
  1. symulowani brokerzy przetwarzają świece do chwili slotu (stop-lossy, wycena),
  2. źródło danych zostaje „zamrożone” na tę chwilę (PointInTimeMarketData),
  3. uruchamiany jest prawdziwy Runner: rekoncyliacja, dzienny limit straty, decyzje agentów,
     risk manager, egzekucja (na SimBroker) i zapis do osobnej bazy backtestu.

Replay agenta LLM:
  - wolno go uruchomić wyłącznie na danych PO granicy wiedzy modelu (+ bufor) — inaczej odmowa,
  - błąd wywołania Claude (limit, timeout...) NIE zamienia się w HOLD: slot jest cofany (baza +
    brokerzy) i backtest się zatrzymuje; ponowne uruchomienie tej samej komendy wznawia od tego slotu.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import AppConfig
from ..runner import Runner
from ..storage.db import Database
from .history import HistoricalStore, PointInTimeMarketData
from .sim_broker import SimBroker

log = logging.getLogger(__name__)
STATE_KEY = "backtest_state"


class LookAheadForbidden(RuntimeError):
    """Replay LLM na danych sprzed granicy wiedzy modelu."""


class BacktestPaused(RuntimeError):
    """Backtest zatrzymany w sposób wznawialny (np. limit wywołań Claude)."""


@dataclass
class BacktestSpec:
    name: str
    start: date
    end: date
    agents: list[str]
    fresh: bool = False


@dataclass
class BacktestResult:
    spec: BacktestSpec
    cfg: AppConfig
    db_path: str
    out_dir: Path
    equity: dict[str, pd.Series]             # agent / "spy" -> equity (start + koniec każdej sesji)
    days: int
    slots: int
    elapsed_s: float
    completed: bool = True
    extra: dict = field(default_factory=dict)


def check_llm_window(cfg: AppConfig, start: date) -> date:
    """Zwraca najwcześniejszą dozwoloną datę replayu LLM albo rzuca LookAheadForbidden."""
    not_before = cfg.backtest.llm_not_before(cfg.llm.model)
    if not_before is None:
        raise LookAheadForbidden(
            f"Brak granicy wiedzy dla modelu {cfg.llm.model} w config.yaml (backtest.llm_knowledge_cutoff). "
            "Replay LLM bez niej groziłby look-ahead bias — odmawiam.")
    if start < not_before:
        raise LookAheadForbidden(
            f"Replay LLM od {start} jest niedozwolony: model {cfg.llm.model} może znać dane sprzed "
            f"{not_before} (granica wiedzy + {cfg.backtest.llm_cutoff_buffer_days} dni bufora). "
            f"Użyj --start {not_before} lub później.")
    return not_before


def prepare_cfg(cfg: AppConfig, out_dir: Path, db_path: str) -> AppConfig:
    c = cfg.model_copy(deep=True)
    c.execution.orders_enabled = True         # zlecenia trafiają do SimBroker — nigdy do Alpaca
    c.execution.fill_timeout_s = 0
    c.experiment.frequency = "1h"             # świece 1H = rozdzielczość backtestu
    c.storage.db_path = db_path
    c.storage.reports_dir = str(out_dir)
    return c


def session_slots(open_: datetime, close: datetime, cfg: AppConfig) -> list[datetime]:
    first = open_ + timedelta(minutes=cfg.experiment.first_run_after_open_min)
    last = close - timedelta(minutes=cfg.experiment.last_run_before_close_min)
    out, t = [], first
    while t <= last:
        out.append(t)
        t += timedelta(minutes=cfg.experiment.frequency_minutes)
    return out


class BacktestEngine:
    def __init__(self, cfg: AppConfig, store: HistoricalStore, spec: BacktestSpec, llm_cli=None,
                 db_path: str | None = None, progress: Callable[[str], None] = print):
        if not spec.agents:
            raise ValueError("Podaj co najmniej jednego agenta")
        if "llm" in spec.agents:
            check_llm_window(cfg, spec.start)
            if llm_cli is None:
                raise ValueError("Replay LLM wymaga Claude Code CLI")
        self.out_dir = Path(cfg.backtest.dir) / spec.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(self.out_dir / "backtest.db")
        self.cfg = prepare_cfg(cfg, self.out_dir, self.db_path)
        self.store = store
        self.spec = spec
        self.llm_cli = llm_cli
        self.progress = progress
        self.strict = "llm" in spec.agents
        self._now: datetime | None = None

    # ------------------------------------------------------------------ stan
    def _load_state(self, db: Database) -> dict | None:
        raw = db.get_meta(STATE_KEY)
        return json.loads(raw) if raw else None

    def _save_state(self, db: Database, st: dict, brokers: dict[str, SimBroker]) -> None:
        st["brokers"] = {a: b.to_state() for a, b in brokers.items()}
        db.set_meta(STATE_KEY, json.dumps(st))

    # ------------------------------------------------------------------ główna pętla
    def run(self) -> BacktestResult:
        t0 = _time.monotonic()
        if self.spec.fresh and self.db_path != ":memory:" and Path(self.db_path).exists():
            Path(self.db_path).unlink()
            for suffix in ("-wal", "-shm"):
                Path(self.db_path + suffix).unlink(missing_ok=True)
        db = Database(self.db_path)
        cfg, store, bench = self.cfg, self.store, self.cfg.benchmark_symbol
        brokers = {a: SimBroker(a, store, cfg.backtest.start_equity, cfg.backtest.spread_bps, bench)
                   for a in self.spec.agents}
        md = PointInTimeMarketData(store, cfg.backtest.spread_bps)
        runner = Runner(cfg, db, brokers, md, self.llm_cli, now_fn=lambda: self._now, sleep=lambda s: None,
                        llm_fail_on_error=self.strict)

        days = store.trading_days(bench, self.spec.start, self.spec.end)
        if not days:
            raise ValueError(f"Brak sesji z danymi w okresie {self.spec.start} – {self.spec.end}")

        st = self._load_state(db)
        if st is not None:
            if st["config_hash"] != cfg.config_hash or st["agents"] != self.spec.agents:
                raise RuntimeError("Zapisany stan backtestu pochodzi z innej konfiguracji/agentów — użyj --fresh "
                                   "albo innej nazwy (--name).")
            if db.query_one("SELECT 1 FROM runs WHERE status = 'running' LIMIT 1"):
                raise RuntimeError("Poprzedni przebieg został przerwany twardo w trakcie slotu (np. zabity proces) — "
                                   "stanu nie da się bezpiecznie odtworzyć. Uruchom z --fresh.")
            for a, b in brokers.items():
                b.load_state(st["brokers"][a])
            self.progress(f"Wznawiam backtest '{self.spec.name}' po slocie {st['last_slot']}")
        else:
            first_open, _ = store.session_hours(bench, days[0])
            first_slot = session_slots(first_open, store.session_hours(bench, days[0])[1], cfg)[0]
            spy_base = store.last_close(bench, first_slot)
            for b in brokers.values():
                b.now = first_slot
            st = {"config_hash": cfg.config_hash, "agents": self.spec.agents, "last_slot": None,
                  "days_done": [], "day_open": None, "spy_base": spy_base,
                  "curve": {a: [[first_slot.isoformat(), cfg.backtest.start_equity]] for a in self.spec.agents},
                  "spy": [[first_slot.isoformat(), cfg.backtest.start_equity]], "slots": 0}

        last_slot = datetime.fromisoformat(st["last_slot"]) if st["last_slot"] else None
        for day in days:
            if day.isoformat() in st["days_done"]:
                continue
            open_, close = store.session_hours(bench, day)
            if st["day_open"] != day.isoformat():
                for b in brokers.values():
                    b.advance_to(open_)
                    b.begin_day()
                st["day_open"] = day.isoformat()
            for slot in session_slots(open_, close, cfg):
                if last_slot is not None and slot <= last_slot:
                    continue
                for b in brokers.values():
                    b.advance_to(slot)
                self._now = slot
                md.as_of = slot
                checkpoint = self._checkpoint(db, brokers)
                try:
                    summary = runner.run(dry_run=False, agents=self.spec.agents)
                except KeyboardInterrupt:
                    self._restore(db, brokers, checkpoint)
                    raise BacktestPaused(f"przerwano w slocie {slot:%Y-%m-%d %H:%M} NY (slot cofnięty).\n"
                                         "Uruchom tę samą komendę ponownie, żeby wznowić.") from None
                if summary.status != "ok":
                    if checkpoint is not None:
                        self._restore(db, brokers, checkpoint)
                    msg = f"slot {slot:%Y-%m-%d %H:%M} NY: {summary.status} — {summary.reason}"
                    if "LLMUnavailable" in (summary.reason or "") or "Claude" in (summary.reason or ""):
                        raise BacktestPaused(msg + "\nUruchom tę samą komendę ponownie, żeby wznowić od tego slotu.")
                    raise RuntimeError(msg)
                st["last_slot"] = slot.isoformat()
                st["slots"] += 1
                last_slot = slot
                self._save_state(db, st, brokers)
            for b in brokers.values():
                b.advance_to(close)
            for a, b in brokers.items():
                st["curve"][a].append([close.isoformat(), b.equity])
            spy_close = store.last_close(bench, close)
            st["spy"].append([close.isoformat(), cfg.backtest.start_equity * spy_close / st["spy_base"]])
            st["days_done"].append(day.isoformat())
            st["day_open"] = None
            self._save_state(db, st, brokers)
            self.progress(f"{day}: " + " | ".join(
                f"{a.upper()} {b.equity:,.2f}" for a, b in brokers.items()) + f" | SPY {st['spy'][-1][1]:,.2f}"
                + (f" | wywołania Claude: {self._llm_calls(db)}" if "llm" in self.spec.agents else ""))

        equity = {a: _series(st["curve"][a]) for a in self.spec.agents}
        equity["spy"] = _series(st["spy"])
        res = BacktestResult(self.spec, cfg, self.db_path, self.out_dir, equity, len(st["days_done"]),
                             st["slots"], _time.monotonic() - t0)
        res.extra["db"] = db
        return res

    # ------------------------------------------------------------------ atomowość slotu
    @staticmethod
    def _checkpoint(db: Database, brokers: dict[str, SimBroker]) -> tuple[sqlite3.Connection, dict]:
        mem = sqlite3.connect(":memory:")
        db.conn.backup(mem)
        return mem, {a: json.dumps(b.to_state()) for a, b in brokers.items()}

    @staticmethod
    def _restore(db: Database, brokers: dict[str, SimBroker], checkpoint) -> None:
        mem, states = checkpoint
        mem.backup(db.conn)
        mem.close()
        for a, b in brokers.items():
            b.load_state(json.loads(states[a]))

    @staticmethod
    def _llm_calls(db: Database) -> int:
        return int(db.query_one("SELECT COUNT(*) AS n FROM llm_calls")["n"])


def _series(points: list) -> pd.Series:
    return pd.Series([float(v) for _, v in points], index=pd.DatetimeIndex([pd.Timestamp(t) for t, _ in points]))
