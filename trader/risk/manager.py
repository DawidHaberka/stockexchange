"""Risk manager — ma ostatnie słowo nad każdym agentem.

Reguły (wszystkie konfigurowalne w config.yaml, sekcja risk):
- tylko tickery z watchlisty, tylko akcje/ETF-y (us_equity), handlowalne;
- tylko long: SELL wyłącznie z istniejącej pozycji, nigdy więcej niż posiadana ilość;
- bez dźwigni: zakupy tylko z gotówki (cash), nie z buying_power/marginu;
- max `max_position_pct` equity w jednej pozycji (większe zlecenie jest przycinane);
- max `max_open_positions` otwartych pozycji;
- każda nowa pozycja musi mieć wyliczalny stop-loss = cena - k x ATR;
- cooldown po transakcji w tickerze;
- dzienny limit straty: po przekroczeniu brak handlu do następnej sesji;
- brak nowego zlecenia, gdy w tickerze wisi niezrealizowane zlecenie.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from ..agent.base import PortfolioView
from ..agent.schemas import Action, AgentDecision
from ..broker import AccountState, AssetInfo
from ..config import AppConfig
from ..data.snapshot import MarketSnapshot

log = logging.getLogger(__name__)


def round_price(p: float) -> float:
    """Ceny >= $1: krok 0.01; poniżej $1: krok 0.0001 (wymogi Alpaca)."""
    return round(p, 2) if p >= 1 else round(p, 4)


@dataclass
class OrderIntent:
    symbol: str
    side: str            # buy | sell
    qty: int
    limit_price: float
    stop_price: float | None
    notional: float
    atr_for_stop: float | None = None


@dataclass
class RiskVerdict:
    agent_decision: AgentDecision
    status: str          # approved | resized | rejected | hold
    reason: str
    intent: OrderIntent | None = None

    @property
    def is_order(self) -> bool:
        return self.status in ("approved", "resized") and self.intent is not None


class RiskManager:
    def __init__(self, cfg: AppConfig, asset_lookup: Callable[[str], AssetInfo] | None = None):
        self.cfg = cfg
        self.r = cfg.risk
        self.asset_lookup = asset_lookup

    # ------------------------------------------------------------------ daily loss
    def daily_loss_pct(self, account: AccountState) -> float | None:
        """Zmiana equity względem zamknięcia poprzedniej sesji (last_equity), w %."""
        if not account.last_equity or account.last_equity <= 0:
            return None
        return (account.equity / account.last_equity - 1) * 100

    def daily_loss_breached(self, account: AccountState) -> tuple[bool, float | None]:
        pct = self.daily_loss_pct(account)
        return (pct is not None and pct <= -self.r.daily_loss_limit_pct), pct

    # ------------------------------------------------------------------ review
    def review(self, decisions: list[AgentDecision], portfolio: PortfolioView, snapshot: MarketSnapshot,
               now: datetime, halted: bool = False, halt_reason: str | None = None) -> list[RiskVerdict]:
        verdicts: list[RiskVerdict] = []
        cash_left = max(0.0, portfolio.cash) * (1 - self.r.cash_buffer_pct / 100)
        held = {s for s, p in portfolio.positions.items() if p.qty > 0}
        new_positions = 0
        pending_symbols = {o.symbol for o in portfolio.open_orders if o.order_type != "stop" and not o.is_terminal}
        touched: set[str] = set()

        for ad in decisions:
            d = ad.decision
            sym = d.ticker

            def reject(reason: str, ad: AgentDecision = ad) -> None:
                verdicts.append(RiskVerdict(ad, "rejected", reason))

            if d.action == Action.HOLD:
                verdicts.append(RiskVerdict(ad, "hold", "HOLD — brak zlecenia"))
                continue
            if halted:
                reject(f"dzienny limit straty — handel wstrzymany do następnej sesji ({halt_reason or ''})".strip())
                continue
            if portfolio.account.trading_blocked:
                reject("konto ma zablokowany handel")
                continue
            if sym not in self.cfg.watchlist:
                reject(f"{sym} spoza watchlisty")
                continue
            if sym in touched:
                reject("już jest decyzja dla tego tickera w tym przebiegu")
                continue
            asset_problem = self._asset_problem(sym)
            if asset_problem:
                reject(asset_problem)
                continue
            if sym in pending_symbols:
                reject("w tickerze wisi niezrealizowane zlecenie")
                continue
            last = portfolio.last_trade_at.get(sym)
            if last and self.r.cooldown_minutes > 0:
                until = last + timedelta(minutes=self.r.cooldown_minutes)
                if now < until:
                    reject(f"cooldown po transakcji do {until.isoformat(timespec='minutes')}")
                    continue

            snap = snapshot.get(sym)
            if snap is None or snap.price is None:
                reject("brak bieżącej ceny")
                continue
            offset = self.cfg.execution.limit_offset_bps / 1e4

            if d.action == Action.SELL:
                pos = portfolio.position(sym)
                if pos is None:
                    reject("brak pozycji — krótka sprzedaż zabroniona")
                    continue
                full = int(math.floor(pos.qty + 1e-9))
                qty = full if d.size_pct >= 100 else int(math.floor(pos.qty * d.size_pct / 100 + 1e-9))
                qty = min(qty, full)
                if qty < 1:
                    reject(f"ilość do sprzedaży < 1 akcji ({d.size_pct:.1f}% z {pos.qty:g})")
                    continue
                ref = snap.quote.bid if snap.quote.bid and snap.quote.bid > 0 else snap.price
                limit = round_price(ref * (1 - offset))
                verdicts.append(RiskVerdict(
                    ad, "approved", f"SELL {qty} z {pos.qty:g} akcji",
                    OrderIntent(sym, "sell", qty, limit, None, qty * limit),
                ))
                touched.add(sym)
                continue

            # ---------------- BUY ----------------
            equity = portfolio.equity
            if equity <= 0:
                reject("equity <= 0")
                continue
            pos = portfolio.position(sym)
            if pos is None and len(held) + new_positions >= self.r.max_open_positions:
                reject(f"limit otwartych pozycji ({self.r.max_open_positions})")
                continue
            requested = equity * d.size_pct / 100
            current_value = pos.market_value if pos else 0.0
            cap = equity * self.r.max_position_pct / 100 - current_value
            if cap <= 0:
                reject(f"pozycja już ma >= {self.r.max_position_pct:g}% equity")
                continue
            notional = min(requested, cap)
            reasons = []
            if notional < requested:
                reasons.append(f"przycięte do limitu {self.r.max_position_pct:g}% equity")
            if notional > cash_left:
                notional = cash_left
                reasons.append("przycięte do dostępnej gotówki (bez dźwigni)")
            if notional <= 0:
                reject("brak wolnej gotówki (bez dźwigni)")
                continue

            ref = snap.quote.ask if snap.quote.ask and snap.quote.ask > 0 else snap.price
            limit = round_price(ref * (1 + offset))
            qty = int(math.floor(notional / limit))
            if qty < 1:
                reject(f"kwota {notional:.2f} USD < cena 1 akcji ({limit:.2f})")
                continue

            atr = snap.atr_for(self.r.stop_loss_atr_timeframe)
            if not atr or atr <= 0:
                reject(f"brak ATR({self.r.stop_loss_atr_timeframe}) — nie da się ustawić stop-lossa")
                continue
            stop = round_price(limit - self.r.stop_loss_atr_mult * atr)
            if stop <= 0:
                reject("stop-loss wyszedłby <= 0")
                continue

            cost = qty * limit
            cash_left -= cost
            if pos is None:
                new_positions += 1
            touched.add(sym)
            status = "resized" if reasons else "approved"
            reason = (f"BUY {qty} szt. @ limit {limit:.2f} (~{cost:.2f} USD, "
                      f"{cost / equity * 100:.1f}% equity), stop {stop:.2f} = cena - "
                      f"{self.r.stop_loss_atr_mult:g}xATR({self.r.stop_loss_atr_timeframe})")
            if reasons:
                reason += "; " + "; ".join(reasons)
            verdicts.append(RiskVerdict(ad, status, reason, OrderIntent(sym, "buy", qty, limit, stop, cost, atr)))

        return verdicts

    def _asset_problem(self, sym: str) -> str | None:
        if self.asset_lookup is None:
            return None
        try:
            a = self.asset_lookup(sym)
        except Exception as exc:  # brak informacji o aktywie = nie handlujemy
            return f"nie udało się sprawdzić aktywa ({exc})"
        if a.asset_class != "us_equity":
            return f"klasa aktywa {a.asset_class} niedozwolona (tylko akcje/ETF)"
        if not a.tradable or a.status not in ("active", ""):
            return "aktywo niehandlowalne"
        return None
