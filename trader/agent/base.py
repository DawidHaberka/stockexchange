"""Wspólne typy agentów: widok portfela przekazywany agentom i risk managerowi."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..broker import AccountState, OrderState, PositionState
from ..data.snapshot import MarketSnapshot
from .schemas import AgentDecision


@dataclass
class PortfolioView:
    agent: str
    account: AccountState
    positions: dict[str, PositionState]
    stop_levels: dict[str, float] = field(default_factory=dict)
    open_orders: list[OrderState] = field(default_factory=list)
    last_trade_at: dict[str, datetime] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        return self.account.equity

    @property
    def cash(self) -> float:
        return self.account.cash

    @property
    def n_positions(self) -> int:
        return sum(1 for p in self.positions.values() if p.qty > 0)

    def position(self, symbol: str) -> PositionState | None:
        p = self.positions.get(symbol)
        return p if p and p.qty > 0 else None


@dataclass
class RunContext:
    run_id: int | None
    now: datetime
    session_open: datetime | None
    session_close: datetime | None


class Agent(Protocol):
    name: str

    def decide(self, snapshot: MarketSnapshot, portfolio: PortfolioView, ctx: RunContext) -> list[AgentDecision]: ...
