"""Metryki ewaluacji: total return, Sharpe, max drawdown, win rate (FIFO)."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.astype(float).pct_change().dropna()


def total_return_pct(equity: pd.Series) -> float | None:
    if equity is None or len(equity) < 1 or not equity.iloc[0]:
        return None
    return (float(equity.iloc[-1]) / float(equity.iloc[0]) - 1) * 100


def sharpe_ratio(equity: pd.Series, risk_free_annual: float = 0.0) -> float | None:
    r = daily_returns(equity)
    if len(r) < 2:
        return None
    excess = r - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    if not sd or math.isnan(sd):
        return None
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def max_drawdown_pct(equity: pd.Series) -> float | None:
    if equity is None or len(equity) < 1:
        return None
    e = equity.astype(float)
    dd = e / e.cummax() - 1
    return float(dd.min() * 100)


@dataclass
class ClosedTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    exit_at: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty


def fifo_closed_trades(trades: list[dict]) -> list[ClosedTrade]:
    """Paruje sprzedaże z wcześniejszymi kupnami (FIFO). Jedna sprzedaż = jedna zamknięta transakcja
    (średnia ważona ceny wejścia zjedzonych lotów)."""
    lots: dict[str, deque] = defaultdict(deque)
    closed: list[ClosedTrade] = []
    for t in sorted(trades, key=lambda x: (x["filled_at"], x.get("id", 0))):
        sym, qty, price = t["symbol"], float(t["qty"]), float(t["price"])
        if t["side"] == "buy":
            lots[sym].append([qty, price])
            continue
        remaining, cost, matched = qty, 0.0, 0.0
        while remaining > 1e-9 and lots[sym]:
            lot = lots[sym][0]
            take = min(lot[0], remaining)
            cost += take * lot[1]
            matched += take
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-9:
                lots[sym].popleft()
        if matched > 0:
            closed.append(ClosedTrade(sym, matched, cost / matched, price, t["filled_at"]))
    return closed


def win_rate_pct(closed: list[ClosedTrade]) -> float | None:
    if not closed:
        return None
    return sum(1 for c in closed if c.pnl > 0) / len(closed) * 100


def summarize(equity: pd.Series, risk_free_annual: float = 0.0) -> dict:
    return {
        "start_value": float(equity.iloc[0]) if len(equity) else None,
        "end_value": float(equity.iloc[-1]) if len(equity) else None,
        "total_return_pct": total_return_pct(equity),
        "sharpe": sharpe_ratio(equity, risk_free_annual),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "days": int(len(equity)),
    }
