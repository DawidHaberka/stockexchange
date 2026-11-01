from trader.agent.schemas import Action, AgentDecision, Decision
from trader.broker import PositionState
from trader.data.indicators import IndicatorSet
from trader.data.market_data import QuoteInfo
from trader.data.snapshot import DailySummary, MarketSnapshot, TickerSnapshot
from tests.conftest import NOW


def tsnap(symbol, price, atr_1h=1.0, atr_1d=3.0, sma_fast=None, sma_slow=None, news=None):
    hourly = IndicatorSet(100, price, sma_fast if sma_fast is not None else price,
                          sma_slow if sma_slow is not None else price, 55.0, atr_1h, 1000.0, 900.0)
    daily = DailySummary(100, price, price, price, 50.0, atr_1d, 1.0, 2.0, price * 1.05, price * 0.95)
    return TickerSnapshot(symbol, NOW, QuoteInfo(round(price - 0.01, 2), round(price + 0.01, 2), price), hourly, [],
                          daily, news or [])


def snapshot(prices: dict, **kw) -> MarketSnapshot:
    return MarketSnapshot(NOW, {s: tsnap(s, p, **kw) for s, p in prices.items()})


def dec(ticker, action, size=10.0, conf=0.8, source="llm"):
    a = Action(action)
    return AgentDecision(Decision(ticker=ticker, action=a, size_pct=0 if a == Action.HOLD else size,
                                  confidence=conf, reasoning="test"), source)


def pos(symbol, qty, price, entry=None):
    return PositionState(symbol, qty, entry or price, qty * price, price)
