"""Agent bazowy („głupi”): crossover SMA fast/slow na świecach 1H.

Reguła stanu: SMA_fast > SMA_slow => trzymaj pozycję target_size_pct equity,
SMA_fast < SMA_slow => zamknij pozycję. Gdy kandydatów do kupna jest więcej niż
wolnych miejsc w portfelu, pierwszeństwo ma największa różnica między średnimi.
Ten sam risk manager co dla LLM.
"""

from __future__ import annotations

import logging

from ..config import AppConfig
from ..data.snapshot import MarketSnapshot
from .base import PortfolioView, RunContext
from .schemas import Action, AgentDecision, Decision, hold

log = logging.getLogger(__name__)


class SMAAgent:
    name = "sma"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def decide(self, snapshot: MarketSnapshot, portfolio: PortfolioView, ctx: RunContext) -> list[AgentDecision]:
        fast, slow = self.cfg.sma.fast, self.cfg.sma.slow
        sells: list[AgentDecision] = []
        buys: list[tuple[float, AgentDecision]] = []
        holds: list[AgentDecision] = []

        for sym in self.cfg.watchlist:
            snap = snapshot.get(sym)
            ind = snap.hourly if snap else None
            if snap is None or not snap.data_ok or ind is None or ind.sma_fast is None or ind.sma_slow is None:
                holds.append(AgentDecision(hold(sym, f"HOLD: za mało danych 1H do SMA{fast}/{slow}"), "no_data"))
                continue

            spread = ind.sma_fast / ind.sma_slow - 1
            pos = portfolio.position(sym)
            desc = (f"SMA{fast}(1H)={ind.sma_fast:.2f} {'>' if spread > 0 else '<='} SMA{slow}(1H)="
                    f"{ind.sma_slow:.2f} (różnica {spread * 100:+.2f}%)")
            if spread > 0:
                if pos:
                    holds.append(AgentDecision(hold(sym, f"{desc} — trend wzrostowy, trzymam pozycję."), "rule"))
                else:
                    d = Decision(ticker=sym, action=Action.BUY, size_pct=self.cfg.sma.target_size_pct,
                                 confidence=1.0, reasoning=f"{desc} — trend wzrostowy, otwieram pozycję.")
                    buys.append((spread, AgentDecision(d, "rule")))
            else:
                if pos:
                    d = Decision(ticker=sym, action=Action.SELL, size_pct=100, confidence=1.0,
                                 reasoning=f"{desc} — trend spadkowy, zamykam pozycję.")
                    sells.append(AgentDecision(d, "rule"))
                else:
                    holds.append(AgentDecision(hold(sym, f"{desc} — trend spadkowy, brak pozycji."), "rule"))

        buys.sort(key=lambda t: t[0], reverse=True)
        ordered = sells + [b for _, b in buys] + holds
        for ad in ordered:
            if ad.action != Action.HOLD:
                log.info("[sma] %s: %s %.0f%% | %s", ad.ticker, ad.action.value, ad.decision.size_pct,
                         ad.decision.reasoning)
        return ordered
