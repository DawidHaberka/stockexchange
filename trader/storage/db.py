"""Repozytorium SQLite: decyzje, zlecenia, transakcje, stan portfela, koszty LLM."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        # źródło czasu dla znaczników w bazie; orkiestrator podmienia je na swój zegar
        # (w backteście = czas symulowany), żeby wszystkie tabele miały spójny czas
        self.clock = utcnow
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.init()

    def init(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ helpers
    def _insert(self, table: str, data: dict[str, Any]) -> int:
        cols = ", ".join(data.keys())
        qs = ", ".join("?" for _ in data)
        cur = self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(data.values()))
        return int(cur.lastrowid)

    def _update(self, table: str, row_id: int, data: dict[str, Any]) -> None:
        if not data:
            return
        sets = ", ".join(f"{k} = ?" for k in data)
        self.conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*data.values(), row_id])

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params))

    def query_one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ config
    def register_config(self, cfg) -> str:
        h = cfg.config_hash
        self.conn.execute(
            """INSERT OR IGNORE INTO config_versions
               (config_hash, created_at, model, effort, frequency, prompt_hash, system_prompt, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (h, iso(self.clock()), cfg.llm.model, cfg.llm.effort, cfg.experiment.frequency,
             cfg.llm.prompt_hash, cfg.llm.system_prompt, cfg.canonical_json()),
        )
        return h

    # ------------------------------------------------------------------ runs
    def start_run(self, session_date: str, slot_start: datetime | None, mode: str, config_hash: str) -> int:
        return self._insert("runs", {
            "started_at": iso(self.clock()), "session_date": session_date, "slot_start": iso(slot_start),
            "mode": mode, "status": "running", "config_hash": config_hash,
        })

    def finish_run(self, run_id: int, status: str, message: str | None = None) -> None:
        self._update("runs", run_id, {"finished_at": iso(self.clock()), "status": status, "message": message})

    def slot_processed(self, slot_start: datetime, mode: str) -> bool:
        row = self.query_one("SELECT 1 FROM runs WHERE slot_start = ? AND mode = ? LIMIT 1", (iso(slot_start), mode))
        return row is not None

    # ------------------------------------------------------------------ experiment start
    def get_experiment_start(self, agent: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM experiment_start WHERE agent = ?", (agent,))

    def set_experiment_start(self, agent: str, started_at: datetime, session_date: str, start_equity: float,
                             account_number: str | None, benchmark_symbol: str, benchmark_price: float | None) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO experiment_start
               (agent, started_at, session_date, start_equity, account_number, benchmark_symbol, benchmark_price)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (agent, iso(started_at), session_date, start_equity, account_number, benchmark_symbol, benchmark_price),
        )

    # ------------------------------------------------------------------ llm calls / decisions
    def insert_llm_call(self, data: dict[str, Any]) -> int:
        return self._insert("llm_calls", data)

    def insert_decision(self, data: dict[str, Any]) -> int:
        if isinstance(data.get("triggers"), (list, tuple)):
            data = {**data, "triggers": json.dumps(list(data["triggers"]), ensure_ascii=False)}
        return self._insert("decisions", data)

    def update_decision(self, decision_id: int, data: dict[str, Any]) -> None:
        self._update("decisions", decision_id, data)

    # ------------------------------------------------------------------ orders / trades
    def insert_order(self, data: dict[str, Any]) -> int:
        return self._insert("orders", data)

    def update_order(self, order_id: int, data: dict[str, Any]) -> None:
        self._update("orders", order_id, data)

    def get_order_by_alpaca_id(self, alpaca_order_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM orders WHERE alpaca_order_id = ?", (alpaca_order_id,))

    def tracked_unsettled_orders(self, agent: str) -> list[sqlite3.Row]:
        """Zlecenia live, których realizacja nie została jeszcze rozliczona."""
        return self.query(
            "SELECT * FROM orders WHERE agent = ? AND dry_run = 0 AND fill_recorded = 0 AND alpaca_order_id IS NOT NULL",
            (agent,),
        )

    def insert_trade(self, data: dict[str, Any]) -> int:
        return self._insert("trades", data)

    def last_trade_time(self, agent: str, symbol: str) -> datetime | None:
        row = self.query_one(
            "SELECT MAX(filled_at) AS t FROM trades WHERE agent = ? AND symbol = ?", (agent, symbol)
        )
        return parse_iso(row["t"]) if row and row["t"] else None

    def trades(self, agent: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM trades WHERE agent = ? ORDER BY filled_at, id", (agent,))

    # ------------------------------------------------------------------ snapshots
    def insert_snapshot(self, agent: str, run_id: int | None, session_date: str, kind: str, account, positions) -> int:
        pos_json = json.dumps(
            {s: {"qty": p.qty, "avg_entry_price": p.avg_entry_price, "market_value": p.market_value,
                 "current_price": p.current_price} for s, p in positions.items()},
            ensure_ascii=False,
        )
        return self._insert("portfolio_snapshots", {
            "agent": agent, "run_id": run_id, "taken_at": iso(self.clock()), "session_date": session_date,
            "kind": kind, "equity": account.equity, "last_equity": account.last_equity, "cash": account.cash,
            "positions_value": sum(p.market_value for p in positions.values()), "positions_json": pos_json,
        })

    # ------------------------------------------------------------------ risk state
    def is_halted(self, agent: str, session_date: str) -> tuple[bool, str | None]:
        row = self.query_one("SELECT halted, reason FROM risk_state WHERE agent = ? AND session_date = ?",
                             (agent, session_date))
        return (bool(row["halted"]), row["reason"]) if row else (False, None)

    def set_halted(self, agent: str, session_date: str, reason: str) -> None:
        self.conn.execute(
            """INSERT INTO risk_state (agent, session_date, halted, reason, updated_at) VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(agent, session_date) DO UPDATE SET halted = 1, reason = excluded.reason,
               updated_at = excluded.updated_at""",
            (agent, session_date, reason, iso(self.clock())),
        )

    # ------------------------------------------------------------------ ticker state
    def get_ticker_state(self, agent: str, symbol: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM ticker_state WHERE agent = ? AND symbol = ?", (agent, symbol))

    def update_ticker_state(self, agent: str, symbol: str, run_at: datetime, price: float,
                            llm_called: bool) -> None:
        existing = self.get_ticker_state(agent, symbol)
        last_llm_at = iso(run_at) if llm_called else (existing["last_llm_at"] if existing else None)
        last_llm_price = price if llm_called else (existing["last_llm_price"] if existing else None)
        self.conn.execute(
            """INSERT INTO ticker_state (agent, symbol, last_run_at, last_price, last_llm_at, last_llm_price)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(agent, symbol) DO UPDATE SET last_run_at = excluded.last_run_at,
               last_price = excluded.last_price, last_llm_at = excluded.last_llm_at,
               last_llm_price = excluded.last_llm_price""",
            (agent, symbol, iso(run_at), price, last_llm_at, last_llm_price),
        )

    # ------------------------------------------------------------------ stop levels
    def get_stop_level(self, agent: str, symbol: str) -> float | None:
        row = self.query_one("SELECT stop_price FROM stop_levels WHERE agent = ? AND symbol = ?", (agent, symbol))
        return float(row["stop_price"]) if row else None

    def set_stop_level(self, agent: str, symbol: str, stop_price: float, atr_used: float | None) -> None:
        self.conn.execute(
            """INSERT INTO stop_levels (agent, symbol, stop_price, atr_used, updated_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(agent, symbol) DO UPDATE SET stop_price = excluded.stop_price,
               atr_used = excluded.atr_used, updated_at = excluded.updated_at""",
            (agent, symbol, stop_price, atr_used, iso(self.clock())),
        )

    def delete_stop_level(self, agent: str, symbol: str) -> None:
        self.conn.execute("DELETE FROM stop_levels WHERE agent = ? AND symbol = ?", (agent, symbol))

    # ------------------------------------------------------------------ benchmark / reports / meta
    def upsert_benchmark_close(self, symbol: str, session_date: str, close: float) -> None:
        self.conn.execute(
            """INSERT INTO benchmark_daily (symbol, session_date, close) VALUES (?, ?, ?)
               ON CONFLICT(symbol, session_date) DO UPDATE SET close = excluded.close""",
            (symbol, session_date, close),
        )

    def save_report(self, session_date: str, metrics: dict, path: str | None) -> None:
        self.conn.execute(
            """INSERT INTO daily_reports (session_date, created_at, metrics_json, report_path) VALUES (?, ?, ?, ?)
               ON CONFLICT(session_date) DO UPDATE SET created_at = excluded.created_at,
               metrics_json = excluded.metrics_json, report_path = excluded.report_path""",
            (session_date, iso(self.clock()), json.dumps(metrics, ensure_ascii=False, default=str), path),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
