"""Pobieranie danych rynkowych z Alpaca: świece 1H/1D, ostatnie kwotowania, newsy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import pandas as pd

from ..broker import NY
from ..config import AlpacaCredentials

log = logging.getLogger(__name__)


@dataclass
class QuoteInfo:
    bid: float | None
    ask: float | None
    last: float | None

    @property
    def price(self) -> float | None:
        if self.last and self.last > 0:
            return self.last
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return None


@dataclass
class NewsItem:
    id: str
    headline: str
    source: str
    created_at: datetime
    summary: str = ""
    url: str | None = None
    symbols: list[str] = field(default_factory=list)


class MarketDataSource(Protocol):
    def hourly_bars(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]: ...
    def daily_bars(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]: ...
    def latest_quotes(self, symbols: list[str]) -> dict[str, QuoteInfo]: ...
    def news(self, symbol: str, start: datetime, limit: int) -> list[NewsItem]: ...


def split_bars(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Dzieli DataFrame z MultiIndex (symbol, timestamp) na słownik per symbol."""
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.index, pd.MultiIndex):
        for sym in symbols:
            if sym in df.index.get_level_values(0):
                part = df.xs(sym, level=0).sort_index()
                out[sym] = part[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def filter_rth_hourly(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Zostawia świece 1H sesji regularnej (start 9:00–15:00 NY) i odrzuca niezamkniętą ostatnią świecę."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ny = idx.tz_convert(NY)
    mask = (ny.hour >= 9) & (ny.hour < 16) & (ny.dayofweek < 5)
    complete = (idx + pd.Timedelta(hours=1)) <= pd.Timestamp(now).tz_convert("UTC")
    return df[mask & complete]


def drop_today_daily(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Odrzuca dzisiejszą (niezamkniętą) świecę dzienną."""
    if df.empty:
        return df
    idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    today = now.astimezone(NY).date()
    dates = idx.tz_convert(NY).date
    return df[[d < today for d in dates]]


class AlpacaMarketData:
    def __init__(self, creds: AlpacaCredentials, feed: str = "iex", adjustment: str = "all"):
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self._stock = StockHistoricalDataClient(api_key=creds.api_key, secret_key=creds.secret_key)
        self._news = NewsClient(api_key=creds.api_key, secret_key=creds.secret_key)
        self.feed = feed
        self.adjustment = adjustment

    def _feed(self):
        from alpaca.data.enums import DataFeed

        return DataFeed.SIP if self.feed == "sip" else DataFeed.IEX

    def _bars(self, symbols, timeframe, start, end) -> dict[str, pd.DataFrame]:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end,
            feed=self._feed(), adjustment=Adjustment.RAW if self.adjustment == "raw" else Adjustment.ALL,
        )
        return split_bars(self._stock.get_stock_bars(req).df, symbols)

    def hourly_bars(self, symbols, start, end):
        from alpaca.data.timeframe import TimeFrame

        return self._bars(symbols, TimeFrame.Hour, start, end)

    def daily_bars(self, symbols, start, end):
        from alpaca.data.timeframe import TimeFrame

        return self._bars(symbols, TimeFrame.Day, start, end)

    def latest_quotes(self, symbols):
        from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

        quotes = self._stock.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=self._feed()))
        trades = self._stock.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self._feed()))
        out = {}
        for s in symbols:
            q = quotes.get(s)
            t = trades.get(s)
            out[s] = QuoteInfo(
                bid=float(q.bid_price) if q and q.bid_price else None,
                ask=float(q.ask_price) if q and q.ask_price else None,
                last=float(t.price) if t and t.price else None,
            )
        return out

    def news(self, symbol, start, limit):
        from alpaca.data.requests import NewsRequest

        if limit <= 0:
            return []
        res = self._news.get_news(NewsRequest(symbols=symbol, start=start, limit=limit, sort="desc",
                                              include_content=False))
        return [_news_item(n) for n in res.data.get("news", [])[:limit]]

    def news_range(self, symbol: str, start: datetime, end: datetime) -> list[NewsItem]:
        """Wszystkie newsy tickera w przedziale (stronicowanie automatyczne) — do backtestów."""
        from alpaca.data.requests import NewsRequest

        res = self._news.get_news(NewsRequest(symbols=symbol, start=start, end=end, sort="asc",
                                              include_content=False))
        return [_news_item(n) for n in res.data.get("news", [])]


def _news_item(n) -> NewsItem:
    return NewsItem(
        id=str(n.id), headline=n.headline or "", source=n.source or "",
        created_at=n.created_at, summary=(n.summary or "")[:300], url=n.url,
        symbols=list(n.symbols or []),
    )

