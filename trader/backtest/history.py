"""Dane historyczne do backtestu i ich widok „z danego momentu” (point-in-time).

`HistoricalStore` trzyma świece 1H (sesja regularna), 1D i newsy dla całego okresu.
`PointInTimeMarketData` udaje źródło danych na żywo w chwili `as_of`: zwraca WYŁĄCZNIE
zamknięte świece i newsy opublikowane do `as_of` — agenci dostają dokładnie to, co
widzieliby wtedy na żywo, bez zaglądania w przyszłość (look-ahead).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from ..broker import NY
from ..data.market_data import NewsItem, QuoteInfo, filter_rth_hourly

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)


def _utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df = df.copy()
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


@dataclass
class HistoricalStore:
    hourly: dict[str, pd.DataFrame]                       # świece 1H sesji regularnej, indeks UTC = start świecy
    daily: dict[str, pd.DataFrame]                        # świece 1D, indeks UTC
    news: dict[str, list[NewsItem]] = field(default_factory=dict)

    def __post_init__(self):
        self.hourly = {s: _utc_index(df) for s, df in self.hourly.items()}
        self.daily = {s: _utc_index(df) for s, df in self.daily.items()}
        self.news = {s: sorted(items, key=lambda n: n.created_at) for s, items in self.news.items()}
        self._news_times = {s: [n.created_at for n in items] for s, items in self.news.items()}

    # ------------------------------------------------------------------ kalendarz
    def trading_days(self, symbol: str, start: date, end: date) -> list[date]:
        df = self.hourly.get(symbol)
        if df is None or df.empty:
            return []
        days = sorted({ts.date() for ts in df.index.tz_convert(NY)})
        return [d for d in days if start <= d <= end]

    def session_hours(self, symbol: str, day: date) -> tuple[datetime, datetime]:
        """Otwarcie 9:30; zamknięcie 16:00 albo wcześniej, jeśli świece kończą się wcześniej (skrócona sesja)."""
        df = self.hourly[symbol]
        ny = df.index.tz_convert(NY)
        hours = [ts.hour for ts in ny if ts.date() == day]
        open_ = datetime.combine(day, time(9, 30), tzinfo=NY)
        last = max(hours) if hours else 15
        close = datetime.combine(day, time(16, 0), tzinfo=NY) if last >= 15 else \
            datetime.combine(day, time(last + 1, 0), tzinfo=NY)
        return open_, close

    def bars_between(self, symbol: str, after_end: datetime | None, until_end: datetime) -> pd.DataFrame:
        """Świece 1H, których koniec (start + 1h) jest w przedziale (after_end, until_end]."""
        df = self.hourly.get(symbol)
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        ends = df.index + HOUR
        mask = ends <= pd.Timestamp(until_end)
        if after_end is not None:
            mask &= ends > pd.Timestamp(after_end)
        return df[mask]

    def last_close(self, symbol: str, as_of: datetime) -> float | None:
        df = self.hourly.get(symbol)
        if df is None or df.empty:
            return None
        cutoff = pd.Timestamp(as_of) - HOUR          # świeca zamknięta: start <= as_of - 1h
        i = df.index.searchsorted(cutoff, side="right")
        return float(df["close"].iloc[i - 1]) if i > 0 else None


class PointInTimeMarketData:
    """Źródło danych zgodne z MarketDataSource, ale „zamrożone” w chwili `as_of`."""

    def __init__(self, store: HistoricalStore, spread_bps: float = 2.0):
        self.store = store
        self.spread_bps = spread_bps
        self.as_of: datetime | None = None

    def _now(self) -> pd.Timestamp:
        if self.as_of is None:
            raise RuntimeError("PointInTimeMarketData: nie ustawiono as_of")
        return pd.Timestamp(self.as_of)

    def hourly_bars(self, symbols, start, end):
        now = self._now()
        out = {}
        for s in symbols:
            df = self.store.hourly.get(s)
            if df is None:
                continue
            lo = df.index.searchsorted(pd.Timestamp(start), side="left")
            hi = df.index.searchsorted(now - HOUR, side="right")      # tylko zamknięte świece
            out[s] = df.iloc[lo:hi]
        return out

    def daily_bars(self, symbols, start, end):
        now = self._now()
        today = now.tz_convert(NY).date()
        out = {}
        for s in symbols:
            df = self.store.daily.get(s)
            if df is None:
                continue
            dates = df.index.tz_convert(NY).date
            mask = (df.index >= pd.Timestamp(start)) & pd.Series([d < today for d in dates], index=df.index).to_numpy()
            out[s] = df[mask]
        return out

    def latest_quotes(self, symbols):
        now = self._now().to_pydatetime()
        half = self.spread_bps / 2 / 1e4
        out = {}
        for s in symbols:
            last = self.store.last_close(s, now)
            out[s] = QuoteInfo(bid=None if last is None else round(last * (1 - half), 4),
                               ask=None if last is None else round(last * (1 + half), 4), last=last)
        return out

    def news(self, symbol, start, limit):
        now = self._now().to_pydatetime()
        items = self.store.news.get(symbol, [])
        visible = [n for n in items if start <= n.created_at <= now]
        return list(reversed(visible))[:limit]


# --------------------------------------------------------------------------- ładowanie z Alpaca (+ cache)
def _cache_key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def load_history(md, symbols: list[str], start: date, end: date, hourly_lookback_days: int,
                 daily_lookback_days: int, with_news: bool, news_lookback_hours: int, cache_dir: str | Path,
                 feed: str, adjustment: str) -> HistoricalStore:
    """Pobiera historię z Alpaca (AlpacaMarketData) z cache na dysku."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t_start = datetime.combine(start, time(0), tzinfo=NY)
    t_end = datetime.combine(end + timedelta(days=1), time(0), tzinfo=NY)
    key = _cache_key(sorted(symbols), start, end, hourly_lookback_days, daily_lookback_days, with_news,
                     news_lookback_hours, feed, adjustment)
    path = cache_dir / f"history_{key}.pkl"
    if path.exists():
        log.info("Historia z cache: %s", path)
        with path.open("rb") as fh:
            return pickle.load(fh)

    log.info("Pobieram świece 1H %s → %s", start, end)
    hourly_raw = md.hourly_bars(symbols, t_start - timedelta(days=hourly_lookback_days), t_end)
    hourly = {s: filter_rth_hourly(df, t_end + timedelta(days=1)) for s, df in hourly_raw.items()}
    log.info("Pobieram świece 1D")
    daily = md.daily_bars(symbols, t_start - timedelta(days=daily_lookback_days), t_end)
    news: dict[str, list[NewsItem]] = {}
    if with_news:
        for s in symbols:
            log.info("Pobieram newsy %s", s)
            news[s] = md.news_range(s, t_start - timedelta(hours=news_lookback_hours), t_end)
    store = HistoricalStore(hourly, daily, news)
    with path.open("wb") as fh:
        pickle.dump(store, fh)
    log.info("Zapisano cache historii: %s", path)
    return store
