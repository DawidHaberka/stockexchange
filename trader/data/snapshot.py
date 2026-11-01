"""Jednorazowy zrzut danych rynkowych dla całej watchlisty — wspólny dla obu agentów."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ..broker import NY
from ..config import AppConfig
from .indicators import IndicatorSet, compute_indicators
from .market_data import MarketDataSource, NewsItem, QuoteInfo, drop_today_daily, filter_rth_hourly
from .news import fetch_watchlist_news

log = logging.getLogger(__name__)


@dataclass
class DailySummary:
    bars: int
    prev_close: float | None
    sma20: float | None
    sma50: float | None
    rsi14: float | None
    atr14: float | None
    ret_5d_pct: float | None
    ret_20d_pct: float | None
    high_20d: float | None
    low_20d: float | None

    @property
    def trend(self) -> str:
        if self.prev_close is None or self.sma20 is None or self.sma50 is None:
            return "unknown"
        if self.prev_close > self.sma20 > self.sma50:
            return "up"
        if self.prev_close < self.sma20 < self.sma50:
            return "down"
        return "mixed"


@dataclass
class TickerSnapshot:
    symbol: str
    asof: datetime
    quote: QuoteInfo
    hourly: IndicatorSet
    hourly_closes: list[tuple[str, float]]
    daily: DailySummary
    news: list[NewsItem] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def price(self) -> float | None:
        return self.quote.price

    @property
    def change_today_pct(self) -> float | None:
        if self.price is None or not self.daily.prev_close:
            return None
        return (self.price / self.daily.prev_close - 1) * 100

    @property
    def data_ok(self) -> bool:
        return self.price is not None and self.hourly.atr14 is not None

    def atr_for(self, timeframe: str) -> float | None:
        return self.daily.atr14 if timeframe == "1d" else self.hourly.atr14


@dataclass
class MarketSnapshot:
    asof: datetime
    tickers: dict[str, TickerSnapshot]

    def get(self, symbol: str) -> TickerSnapshot | None:
        return self.tickers.get(symbol)


def summarize_daily(df: pd.DataFrame) -> DailySummary:
    if df is None or df.empty:
        return DailySummary(0, None, None, None, None, None, None, None, None, None)
    ind = compute_indicators(df, 20, 50)
    close = df["close"].astype(float)

    def ret(n: int) -> float | None:
        if len(close) <= n:
            return None
        return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100

    tail = df.tail(20)
    return DailySummary(
        bars=len(df), prev_close=ind.close, sma20=ind.sma_fast, sma50=ind.sma_slow, rsi14=ind.rsi14,
        atr14=ind.atr14, ret_5d_pct=ret(5), ret_20d_pct=ret(20),
        high_20d=float(tail["high"].max()), low_20d=float(tail["low"].min()),
    )


def build_snapshot(md: MarketDataSource, cfg: AppConfig, now: datetime) -> MarketSnapshot:
    symbols = list(cfg.watchlist)
    if cfg.benchmark_symbol not in symbols:
        symbols.append(cfg.benchmark_symbol)

    hourly = md.hourly_bars(symbols, now - timedelta(days=cfg.data.hourly_lookback_days), now)
    daily = md.daily_bars(symbols, now - timedelta(days=cfg.data.daily_lookback_days), now)
    quotes = md.latest_quotes(symbols)
    news = fetch_watchlist_news(md, cfg.watchlist, now, cfg.data.news_lookback_hours, cfg.data.max_news_per_ticker)

    tickers: dict[str, TickerSnapshot] = {}
    for s in symbols:
        problems: list[str] = []
        h = filter_rth_hourly(hourly.get(s, pd.DataFrame()), now)
        d = drop_today_daily(daily.get(s, pd.DataFrame()), now)
        if h.empty:
            problems.append("brak świec 1H")
        if d.empty:
            problems.append("brak świec 1D")
        ind = compute_indicators(h, cfg.sma.fast, cfg.sma.slow)
        if not ind.complete:
            problems.append(f"za mało świec 1H ({ind.bars}) do pełnych wskaźników")
        q = quotes.get(s) or QuoteInfo(None, None, None)
        if q.price is None:
            problems.append("brak bieżącej ceny")
        tail = h.tail(cfg.data.hourly_closes_in_context)
        closes = [(ts.tz_convert(NY).strftime("%m-%d %H:%M") if ts.tzinfo else str(ts), round(float(c), 4))
                  for ts, c in zip(tail.index, tail["close"], strict=True)]
        tickers[s] = TickerSnapshot(
            symbol=s, asof=now, quote=q, hourly=ind, hourly_closes=closes,
            daily=summarize_daily(d), news=news.get(s, []), problems=problems,
        )
        if problems:
            log.warning("Dane %s: %s", s, "; ".join(problems))
    return MarketSnapshot(asof=now, tickers=tickers)
