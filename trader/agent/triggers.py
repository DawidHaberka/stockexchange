"""Triggery decydujące, czy dla danego tickera warto wywołać LLM (koszt API!)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..broker import PositionState
from ..config import TriggerCfg
from ..data.news import new_since
from ..data.snapshot import TickerSnapshot


@dataclass
class TickerHistory:
    last_run_at: datetime | None
    last_price: float | None
    last_llm_at: datetime | None
    last_llm_price: float | None


def evaluate_triggers(snap: TickerSnapshot, hist: TickerHistory | None, position: PositionState | None,
                      stop_price: float | None, cfg: TriggerCfg) -> list[str]:
    """Zwraca listę powodów wywołania LLM. Pusta lista = HOLD bez wywołania API."""
    triggers: list[str] = []
    price = snap.price
    atr_1h = snap.hourly.atr14

    # 1) ticker nigdy nie oceniany przez LLM
    if (hist is None or hist.last_llm_at is None) and cfg.on_first_evaluation:
        triggers.append("first_evaluation")

    # 2) nowy news od poprzedniego uruchomienia
    if cfg.on_new_news:
        fresh = new_since(snap.news, hist.last_run_at if hist else None)
        if fresh:
            triggers.append(f"new_news:{len(fresh)}")

    # 3) ruch ceny > k x ATR(1H) od ceny odniesienia
    if hist is not None and price and atr_1h:
        ref = hist.last_price if cfg.price_reference == "last_run" else hist.last_llm_price
        if ref:
            move = abs(price - ref) / atr_1h
            if move > cfg.price_move_atr_mult:
                triggers.append(f"price_move:{move:.2f}xATR1H")

    # 4) pozycja blisko stop-lossa
    if position is not None and stop_price and price and atr_1h:
        dist = (price - stop_price) / atr_1h
        if dist <= cfg.near_stop_atr_mult:
            triggers.append(f"near_stop:{dist:.2f}xATR1H")

    return triggers
