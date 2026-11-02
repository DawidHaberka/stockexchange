"""Symulowany broker (interfejs Broker) — konto paper w backteście.

Model realizacji (prosty, konserwatywny, zgodny z tym, jak działa wersja live):
- zlecenie limit jest wykonywane od razu po cenie ask (BUY) / bid (SELL) z chwili decyzji,
  o ile mieści się w limicie (live składa limit ask+0,1%, więc to się zgadza); inaczej nie wchodzi,
- stop-loss GTC jest sprawdzany na kolejnych świecach 1H: gdy low <= stop, sprzedaż po
  min(stop, open świecy) minus pół spreadu (luka cenowa = gorsza cena),
- wycena pozycji = zamknięcie ostatniej przetworzonej świecy,
- last_equity = equity na zamknięciu poprzedniej sesji (jak w Alpaca) — do dziennego limitu straty.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pandas as pd

from ..broker import (NY, AccountState, AssetInfo, ClockState, OrderRequest, OrderState, PositionState,
                      SessionHours)
from .history import HistoricalStore

OPEN_STATUSES = {"new", "accepted", "partially_filled"}


class SimBroker:
    def __init__(self, name: str, store: HistoricalStore, start_equity: float, spread_bps: float,
                 calendar_symbol: str):
        self.name = name
        self.store = store
        self.spread_half = spread_bps / 2 / 1e4
        self.calendar_symbol = calendar_symbol
        self.cash = float(start_equity)
        self.last_equity = float(start_equity)
        self.positions: dict[str, dict] = {}       # symbol -> {qty, avg, price}
        self.orders: dict[str, dict] = {}          # id -> pola OrderState
        self.now: datetime | None = None
        self.processed_until: datetime | None = None   # koniec ostatniej przetworzonej świecy

    # ------------------------------------------------------------------ stan (wznawianie)
    def to_state(self) -> dict:
        return {
            "cash": self.cash, "last_equity": self.last_equity,
            "positions": {k: dict(v) for k, v in self.positions.items()},
            "orders": {k: {**v, "submitted_at": _iso(v["submitted_at"]), "filled_at": _iso(v["filled_at"])}
                       for k, v in self.orders.items()},
            "now": _iso(self.now), "processed_until": _iso(self.processed_until),
        }

    def load_state(self, st: dict) -> None:
        self.cash = st["cash"]
        self.last_equity = st["last_equity"]
        self.positions = {k: dict(v) for k, v in st["positions"].items()}
        self.orders = {k: {**v, "submitted_at": _dt(v["submitted_at"]), "filled_at": _dt(v["filled_at"])}
                       for k, v in st["orders"].items()}
        self.now = _dt(st["now"])
        self.processed_until = _dt(st["processed_until"])

    # ------------------------------------------------------------------ czas
    @property
    def equity(self) -> float:
        return self.cash + sum(p["qty"] * p["price"] for p in self.positions.values())

    def begin_day(self) -> None:
        """Nowa sesja: last_equity = equity na zamknięciu poprzedniej."""
        self.last_equity = self.equity

    def advance_to(self, t: datetime) -> list[dict]:
        """Przetwarza świece zakończone do `t`: stop-lossy i wycena. Zwraca zrealizowane stopy."""
        fills = []
        events: list[tuple[pd.Timestamp, str, pd.Series]] = []
        symbols = set(self.positions) | {o["symbol"] for o in self.orders.values() if o["status"] in OPEN_STATUSES}
        for sym in symbols:
            bars = self.store.bars_between(sym, self.processed_until, t)
            for ts, row in bars.iterrows():
                events.append((ts, sym, row))
        events.sort(key=lambda e: (e[0], e[1]))
        for ts, sym, row in events:
            fills += self._process_bar(sym, ts, row)
        self.processed_until = t
        self.now = t
        return fills

    def _process_bar(self, sym: str, ts: pd.Timestamp, row: pd.Series) -> list[dict]:
        fills = []
        for o in sorted(self.orders.values(), key=lambda o: o["submitted_at"]):
            if o["symbol"] != sym or o["status"] not in OPEN_STATUSES or o["order_type"] != "stop":
                continue
            if row["low"] <= o["stop_price"]:
                pos = self.positions.get(sym)
                qty = min(o["qty"], pos["qty"] if pos else 0)
                if qty <= 0:
                    o["status"] = "canceled"
                    continue
                px = min(o["stop_price"], float(row["open"])) * (1 - self.spread_half)
                self._fill(o, qty, px, ts.to_pydatetime())
                fills.append(dict(o))
        if sym in self.positions:
            self.positions[sym]["price"] = float(row["close"])
        return fills

    def _fill(self, o: dict, qty: float, price: float, when: datetime) -> None:
        sym = o["symbol"]
        pos = self.positions.get(sym)
        if o["side"] == "buy":
            self.cash -= qty * price
            if pos:
                new_q = pos["qty"] + qty
                pos["avg"] = (pos["avg"] * pos["qty"] + price * qty) / new_q
                pos["qty"] = new_q
            else:
                self.positions[sym] = {"qty": qty, "avg": price, "price": price}
        else:
            self.cash += qty * price
            pos["qty"] -= qty
            if pos["qty"] <= 1e-9:
                del self.positions[sym]
        o.update(status="filled", filled_qty=qty, filled_avg_price=round(price, 4), filled_at=when)

    def _quote(self, sym: str) -> float | None:
        return self.store.last_close(sym, self.now)

    # ------------------------------------------------------------------ interfejs Broker
    def get_account(self) -> AccountState:
        return AccountState(f"sim-{self.name}", f"SIM-{self.name.upper()}", "active", self.equity, self.last_equity,
                            self.cash, self.cash, False)

    def get_positions(self) -> dict[str, PositionState]:
        out = {}
        for s, p in self.positions.items():
            if p["qty"] > 0:
                out[s] = PositionState(s, p["qty"], p["avg"], p["qty"] * p["price"], p["price"],
                                       p["price"] / p["avg"] - 1 if p["avg"] else 0.0)
        return out

    def get_clock(self) -> ClockState:
        now = self.now.astimezone(NY)
        open_, close = self.store.session_hours(self.calendar_symbol, now.date())
        return ClockState(now, open_ <= now < close, open_, close)

    def get_session(self, day: date) -> SessionHours | None:
        if day not in set(self.store.trading_days(self.calendar_symbol, day, day)):
            return None
        open_, close = self.store.session_hours(self.calendar_symbol, day)
        return SessionHours(day, open_, close)

    def get_asset(self, symbol: str) -> AssetInfo:
        return AssetInfo(symbol, "us_equity", True, "active", True)

    def _state(self, o: dict) -> OrderState:
        return OrderState(o["id"], o["client_order_id"], o["symbol"], o["side"], o["order_type"], o["qty"],
                          o["filled_qty"], o["filled_avg_price"], o["status"], o["limit_price"], o["stop_price"],
                          o["submitted_at"], o["filled_at"], o["filled_at"] or o["submitted_at"])

    def get_open_orders(self, symbols: list[str] | None = None) -> list[OrderState]:
        return [self._state(o) for o in self.orders.values()
                if o["status"] in OPEN_STATUSES and (symbols is None or o["symbol"] in symbols)]

    def get_order(self, order_id: str) -> OrderState:
        return self._state(self.orders[order_id])

    def submit_order(self, req: OrderRequest) -> OrderState:
        if req.qty <= 0:
            raise RuntimeError("qty must be > 0")
        o = {"id": str(uuid.uuid4()), "client_order_id": req.client_order_id, "symbol": req.symbol,
             "side": req.side, "order_type": req.order_type, "qty": float(req.qty), "filled_qty": 0.0,
             "filled_avg_price": None, "status": "accepted", "limit_price": req.limit_price,
             "stop_price": req.stop_price, "submitted_at": self.now, "filled_at": None}
        if req.order_type == "limit":
            last = self._quote(req.symbol)
            if last is None:
                raise RuntimeError(f"brak ceny {req.symbol}")
            if req.side == "sell":
                pos = self.positions.get(req.symbol)
                locked = sum(x["qty"] for x in self.orders.values()
                             if x["symbol"] == req.symbol and x["order_type"] == "stop" and x["status"] in OPEN_STATUSES)
                if pos is None or pos["qty"] - locked < req.qty - 1e-9:
                    raise RuntimeError("insufficient qty available for order")
                px = last * (1 - self.spread_half)
                if px >= req.limit_price - 1e-9:
                    self.orders[o["id"]] = o
                    self._fill(o, req.qty, px, self.now)
            else:
                px = last * (1 + self.spread_half)
                if req.qty * px > self.cash + 1e-6:
                    raise RuntimeError("insufficient buying power")
                if px <= req.limit_price + 1e-9:
                    self.orders[o["id"]] = o
                    self._fill(o, req.qty, px, self.now)
        elif req.order_type != "stop":
            raise RuntimeError(f"nieobsługiwany typ zlecenia {req.order_type}")
        self.orders[o["id"]] = o
        return self._state(o)

    def cancel_order(self, order_id: str) -> None:
        o = self.orders[order_id]
        if o["status"] in OPEN_STATUSES:
            o["status"] = "canceled"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None
