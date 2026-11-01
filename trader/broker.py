"""Warstwa brokera: cienki wrapper na alpaca-py TradingClient dla jednego konta paper.

Reszta systemu (risk, execution, orchestrator) zna tylko interfejs `Broker`
i proste dataclassy poniżej — dzięki temu testy używają FakeBroker bez sieci.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from .config import NY_TZ_NAME, AlpacaCredentials
from .safety import assert_paper_url

NY = ZoneInfo(NY_TZ_NAME)

TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended"}


@dataclass
class AccountState:
    account_id: str
    account_number: str
    status: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    trading_blocked: bool


@dataclass
class PositionState:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    current_price: float
    unrealized_plpc: float = 0.0
    qty_available: float | None = None


@dataclass
class OrderState:
    id: str
    client_order_id: str
    symbol: str
    side: str          # buy | sell
    order_type: str    # limit | stop | market | ...
    qty: float
    filled_qty: float
    filled_avg_price: float | None
    status: str
    limit_price: float | None = None
    stop_price: float | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES


@dataclass
class ClockState:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass
class SessionHours:
    date: date
    open: datetime   # tz-aware, NY
    close: datetime  # tz-aware, NY


@dataclass
class AssetInfo:
    symbol: str
    asset_class: str
    tradable: bool
    status: str
    fractionable: bool = False


@dataclass
class OrderRequest:
    symbol: str
    side: str            # buy | sell
    qty: int
    order_type: str      # limit | stop
    client_order_id: str
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"


class BrokerError(RuntimeError):
    pass


class Broker(Protocol):
    name: str

    def get_account(self) -> AccountState: ...
    def get_positions(self) -> dict[str, PositionState]: ...
    def get_clock(self) -> ClockState: ...
    def get_session(self, day: date) -> SessionHours | None: ...
    def get_asset(self, symbol: str) -> AssetInfo: ...
    def get_open_orders(self, symbols: list[str] | None = None) -> list[OrderState]: ...
    def get_order(self, order_id: str) -> OrderState: ...
    def submit_order(self, req: OrderRequest) -> OrderState: ...
    def cancel_order(self, order_id: str) -> None: ...


# --------------------------------------------------------------------------- #
# Implementacja Alpaca
# --------------------------------------------------------------------------- #

def _f(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _enum_val(v) -> str:
    return str(getattr(v, "value", v) or "").lower()


def _order_from_alpaca(o) -> OrderState:
    return OrderState(
        id=str(o.id),
        client_order_id=str(o.client_order_id or ""),
        symbol=o.symbol,
        side=_enum_val(o.side),
        order_type=_enum_val(o.order_type or o.type),
        qty=_f(o.qty),
        filled_qty=_f(o.filled_qty),
        filled_avg_price=_f(o.filled_avg_price) if o.filled_avg_price is not None else None,
        status=_enum_val(o.status),
        limit_price=_f(o.limit_price) if o.limit_price is not None else None,
        stop_price=_f(o.stop_price) if o.stop_price is not None else None,
        submitted_at=o.submitted_at,
        filled_at=o.filled_at,
        updated_at=o.updated_at,
    )


class AlpacaBroker:
    """Jedno konto Alpaca Paper. Konstruktor odmawia każdego URL innego niż paper."""

    def __init__(self, name: str, creds: AlpacaCredentials):
        from alpaca.trading.client import TradingClient

        paper_url = assert_paper_url(creds.base_url)
        self.name = name
        self._client = TradingClient(
            api_key=creds.api_key,
            secret_key=creds.secret_key,
            paper=True,
            url_override=paper_url,
        )
        # Podwójna kontrola: klient musi faktycznie wskazywać na paper.
        base = getattr(self._client, "_base_url", "")
        assert_paper_url(str(getattr(base, "value", base)))
        self._asset_cache: dict[str, AssetInfo] = {}

    @property
    def client(self):
        return self._client

    def get_account(self) -> AccountState:
        a = self._client.get_account()
        return AccountState(
            account_id=str(a.id),
            account_number=str(a.account_number),
            status=_enum_val(a.status),
            equity=_f(a.equity),
            last_equity=_f(a.last_equity),
            cash=_f(a.cash),
            buying_power=_f(a.buying_power),
            trading_blocked=bool(a.trading_blocked or a.account_blocked),
        )

    def get_positions(self) -> dict[str, PositionState]:
        out = {}
        for p in self._client.get_all_positions():
            out[p.symbol] = PositionState(
                symbol=p.symbol,
                qty=_f(p.qty),
                avg_entry_price=_f(p.avg_entry_price),
                market_value=_f(p.market_value),
                current_price=_f(p.current_price),
                unrealized_plpc=_f(p.unrealized_plpc),
                qty_available=_f(p.qty_available) if p.qty_available is not None else None,
            )
        return out

    def get_clock(self) -> ClockState:
        c = self._client.get_clock()
        return ClockState(
            timestamp=c.timestamp.astimezone(NY),
            is_open=bool(c.is_open),
            next_open=c.next_open.astimezone(NY),
            next_close=c.next_close.astimezone(NY),
        )

    def get_session(self, day: date) -> SessionHours | None:
        from alpaca.trading.requests import GetCalendarRequest

        cal = self._client.get_calendar(GetCalendarRequest(start=day, end=day))
        for c in cal:
            if c.date == day:
                # alpaca-py zwraca naiwne datetime w czasie NY
                o = c.open if c.open.tzinfo else c.open.replace(tzinfo=NY)
                cl = c.close if c.close.tzinfo else c.close.replace(tzinfo=NY)
                return SessionHours(date=day, open=o.astimezone(NY), close=cl.astimezone(NY))
        return None

    def get_asset(self, symbol: str) -> AssetInfo:
        if symbol not in self._asset_cache:
            a = self._client.get_asset(symbol)
            self._asset_cache[symbol] = AssetInfo(
                symbol=a.symbol,
                asset_class=_enum_val(a.asset_class),
                tradable=bool(a.tradable),
                status=_enum_val(a.status),
                fractionable=bool(a.fractionable),
            )
        return self._asset_cache[symbol]

    def get_open_orders(self, symbols: list[str] | None = None) -> list[OrderState]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, symbols=symbols)
        return [_order_from_alpaca(o) for o in self._client.get_orders(filter=req)]

    def get_order(self, order_id: str) -> OrderState:
        return _order_from_alpaca(self._client.get_order_by_id(order_id))

    def submit_order(self, req: OrderRequest) -> OrderState:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest

        side = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if req.time_in_force == "gtc" else TimeInForce.DAY
        if req.order_type == "limit":
            payload = LimitOrderRequest(
                symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif,
                limit_price=req.limit_price, client_order_id=req.client_order_id,
            )
        elif req.order_type == "stop":
            if req.side != "sell":
                raise BrokerError("Stop-loss może być tylko zleceniem sprzedaży.")
            payload = StopOrderRequest(
                symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif,
                stop_price=req.stop_price, client_order_id=req.client_order_id,
            )
        else:
            raise BrokerError(f"Nieobsługiwany typ zlecenia: {req.order_type}")
        return _order_from_alpaca(self._client.submit_order(payload))

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)


def make_broker(name: str, creds: AlpacaCredentials | None) -> AlpacaBroker:
    if creds is None:
        raise BrokerError(f"Brak kluczy Alpaca dla konta '{name}' w .env")
    return AlpacaBroker(name, creds)


__all__ = [
    "AccountState", "PositionState", "OrderState", "ClockState", "SessionHours", "AssetInfo",
    "OrderRequest", "Broker", "BrokerError", "AlpacaBroker", "make_broker", "NY",
    "TERMINAL_ORDER_STATUSES",
]
