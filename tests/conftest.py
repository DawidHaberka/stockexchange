"""Wspólne fixtury: fałszywy broker Alpaca, fałszywe dane rynkowe i fałszywy klient Claude.

Żaden test nie łączy się z siecią.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.agent.base import PortfolioView
from trader.agent.claude_cli import CLIResult
from trader.broker import NY, AccountState, AssetInfo, ClockState, OrderState, PositionState, SessionHours
from trader.config import load_config
from trader.data.market_data import NewsItem, QuoteInfo
from trader.storage.db import Database
from tests.fake_claude import fake_claude  # noqa: F401  (fixture z fałszywym programem `claude`)

ROOT = Path(__file__).resolve().parents[1]

# poniedziałek 2026-11-02, 10:00 NY (EST, UTC-5)
NOW = datetime(2026, 11, 2, 10, 0, tzinfo=NY).astimezone(timezone.utc)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c.storage.db_path = str(tmp_path / "test.db")
    c.storage.reports_dir = str(tmp_path / "reports")
    c.storage.log_dir = str(tmp_path / "logs")
    c.execution.poll_interval_s = 0.01
    c.execution.fill_timeout_s = 0.05
    return c


@pytest.fixture
def db(cfg):
    d = Database(cfg.storage.db_path)
    yield d
    d.close()


# --------------------------------------------------------------------------- broker
class FakeBroker:
    """Konto paper w pamięci. fill_mode[symbol]: full | partial | none | reject."""

    def __init__(self, name="llm", equity=10_000.0, cash=None, last_equity=None, positions=None,
                 account_number=None, is_open=True, now=NOW):
        self.name = name
        self.cash = equity if cash is None else cash
        self.positions: dict[str, PositionState] = dict(positions or {})
        self._equity_override = None
        self.last_equity = equity if last_equity is None else last_equity
        self.account_number = account_number or f"PA{name.upper()}0001"
        self.orders: dict[str, OrderState] = {}
        self.fill_mode: dict[str, str] = {}
        self.prices: dict[str, float] = {}
        self.submitted: list = []
        self.canceled: list[str] = []
        self.is_open = is_open
        self.now = now
        self.trading_blocked = False
        self.assets: dict[str, AssetInfo] = {}

    # ---- konto
    @property
    def equity(self):
        if self._equity_override is not None:
            return self._equity_override
        return self.cash + sum(p.market_value for p in self.positions.values())

    def set_equity(self, value):
        self._equity_override = value

    def get_account(self):
        return AccountState(f"id-{self.name}", self.account_number, "active", self.equity, self.last_equity,
                            self.cash, self.cash * 2, self.trading_blocked)

    def get_positions(self):
        return {s: p for s, p in self.positions.items() if p.qty > 0}

    def get_clock(self):
        n = self.now.astimezone(NY)
        return ClockState(n, self.is_open, n + timedelta(days=1), n.replace(hour=16, minute=0))

    def get_session(self, day: date):
        if day.weekday() >= 5:
            return None
        return SessionHours(day, datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY),
                            datetime(day.year, day.month, day.day, 16, 0, tzinfo=NY))

    def get_asset(self, symbol):
        return self.assets.get(symbol, AssetInfo(symbol, "us_equity", True, "active", True))

    # ---- zlecenia
    def get_open_orders(self, symbols=None):
        return [o for o in self.orders.values() if not o.is_terminal and (symbols is None or o.symbol in symbols)]

    def submit_order(self, req):
        self.submitted.append(req)
        mode = self.fill_mode.get(req.symbol, "full")
        if req.order_type == "limit" and mode == "reject":
            raise RuntimeError("insufficient buying power")
        if req.order_type == "limit" and req.side == "sell":
            held = self.positions.get(req.symbol)
            locked = sum(o.qty for o in self.get_open_orders([req.symbol]) if o.order_type == "stop")
            if held is None or held.qty - locked < req.qty:
                raise RuntimeError("insufficient qty available for order")
        o = OrderState(id=str(uuid.uuid4()), client_order_id=req.client_order_id, symbol=req.symbol, side=req.side,
                       order_type=req.order_type, qty=req.qty, filled_qty=0, filled_avg_price=None, status="new",
                       limit_price=req.limit_price, stop_price=req.stop_price, submitted_at=self.now)
        self.orders[o.id] = o
        return o

    def _apply_fill(self, o: OrderState, qty: float):
        price = o.limit_price
        o.filled_qty = qty
        o.filled_avg_price = price
        o.filled_at = self.now
        pos = self.positions.get(o.symbol)
        if o.side == "buy":
            self.cash -= qty * price
            if pos:
                new_qty = pos.qty + qty
                pos.avg_entry_price = (pos.avg_entry_price * pos.qty + price * qty) / new_qty
                pos.qty = new_qty
                pos.market_value = new_qty * price
            else:
                self.positions[o.symbol] = PositionState(o.symbol, qty, price, qty * price, price)
        else:
            self.cash += qty * price
            pos.qty -= qty
            pos.market_value = pos.qty * pos.current_price
            if pos.qty <= 0:
                del self.positions[o.symbol]

    def get_order(self, order_id):
        o = self.orders[order_id]
        if o.order_type == "limit" and o.status == "new":
            mode = self.fill_mode.get(o.symbol, "full")
            if mode == "full":
                self._apply_fill(o, o.qty)
                o.status = "filled"
            elif mode == "partial":
                self._apply_fill(o, max(1, int(o.qty // 2)))
                o.status = "partially_filled"
            else:
                o.status = "accepted"
        return o

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        o = self.orders[order_id]
        if not o.is_terminal:
            o.status = "canceled"

    def trigger_stop(self, symbol, price):
        """Symuluje zadziałanie stop-lossa między uruchomieniami."""
        for o in self.orders.values():
            if o.symbol == symbol and o.order_type == "stop" and not o.is_terminal:
                o.limit_price = price
                self._apply_fill(o, o.qty)
                o.status = "filled"


# --------------------------------------------------------------------------- dane
def make_bars(n: int, start: datetime, step: timedelta, base: float, drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base * np.cumprod(1 + drift + rng.normal(0, 0.003, n))
    opens = np.concatenate([[base], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.integers(50_000, 150_000, n).astype(float)
    idx = pd.DatetimeIndex([start + i * step for i in range(n)], tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx)


def rth_hourly_index(end: datetime, sessions: int) -> list[datetime]:
    """Godziny 9:00..15:00 NY w ostatnich `sessions` dniach roboczych przed `end`."""
    out = []
    d = end.astimezone(NY).date() - timedelta(days=1)
    while len(out) < sessions * 7:
        if d.weekday() < 5:
            for h in range(15, 8, -1):
                out.append(datetime(d.year, d.month, d.day, h, 0, tzinfo=NY).astimezone(timezone.utc))
        d -= timedelta(days=1)
    return sorted(out)


class FakeMarketData:
    def __init__(self, symbols, now=NOW, trends=None, news=None):
        self.symbols = symbols
        self.now = now
        self.trends = trends or {}
        self.news_items: dict[str, list[NewsItem]] = news or {}
        self._hourly = {}
        self._daily = {}
        for i, s in enumerate(symbols):
            drift = self.trends.get(s, 0.0005 if i % 2 == 0 else -0.0005)
            idx = rth_hourly_index(now, 20)
            df = make_bars(len(idx), idx[0], timedelta(hours=1), 100 + 10 * i, drift, seed=i)
            df.index = pd.DatetimeIndex(idx)
            self._hourly[s] = df
            ddf = make_bars(120, now - timedelta(days=121), timedelta(days=1), 100 + 10 * i, drift * 3, seed=100 + i)
            self._daily[s] = ddf

    def hourly_bars(self, symbols, start, end):
        return {s: self._hourly[s] for s in symbols if s in self._hourly}

    def daily_bars(self, symbols, start, end):
        return {s: self._daily[s] for s in symbols if s in self._daily}

    def latest_quotes(self, symbols):
        out = {}
        for s in symbols:
            last = float(self._hourly[s]["close"].iloc[-1])
            out[s] = QuoteInfo(bid=round(last - 0.01, 2), ask=round(last + 0.01, 2), last=round(last, 2))
        return out

    def news(self, symbol, start, limit):
        return [n for n in self.news_items.get(symbol, []) if n.created_at >= start][:limit]


# --------------------------------------------------------------------------- Claude (CLI)
EQUIV_COST = 0.0213


def cli_ok(structured=None, text=None, model="claude-opus-5-5", stop_reason="tool_use") -> CLIResult:
    """Wynik udanego `claude -p` (jak zwraca ClaudeCLI.ask)."""
    if structured is not None and text is None:
        text = json.dumps(structured)
    return CLIResult(True, structured=structured, text=text, model_served=model, input_tokens=1200,
                     output_tokens=800, cache_creation_tokens=0, cache_read_tokens=900,
                     cost_equiv_usd=EQUIV_COST, duration_ms=1500, stop_reason=stop_reason, session_id="sess-test")


class FakeCLI:
    """Atrapa ClaudeCLI. decide(prompt) -> dict (structured_output) | str (surowy tekst) | CLIResult."""

    def __init__(self, decide):
        self.decide = decide
        self.calls: list[dict] = []

    def ask(self, prompt, schema, model=None, effort=None, system_prompt=None):
        self.calls.append({"prompt": prompt, "schema": schema, "model": model, "effort": effort})
        out = self.decide(prompt)
        if isinstance(out, CLIResult):
            return out
        if isinstance(out, dict):
            return cli_ok(structured=out)
        return cli_ok(text=out)


def ticker_from_prompt(prompt: str) -> str:
    return prompt.split("\n", 1)[0].replace("Ticker:", "").strip()


@pytest.fixture
def portfolio_factory():
    def make(equity=10_000.0, cash=None, positions=None, last_equity=None, open_orders=None, last_trade_at=None,
             stop_levels=None, agent="llm"):
        acc = AccountState("id", "PA1", "active", equity, equity if last_equity is None else last_equity,
                           equity if cash is None else cash, (equity if cash is None else cash) * 2, False)
        return PortfolioView(agent, acc, positions or {}, stop_levels or {}, open_orders or [], last_trade_at or {})
    return make
