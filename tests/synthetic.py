"""Syntetyczna historia rynku do testów backtestu (spójne świece 1H i 1D, newsy z datami)."""

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from trader.backtest.history import HistoricalStore
from trader.broker import NY
from trader.data.market_data import NewsItem


def weekdays(start: date, n: int, back: int = 0) -> list[date]:
    days, d = [], start
    while len(days) < back:          # dni rozgrzewki przed startem
        d -= timedelta(days=1)
        if d.weekday() < 5:
            days.insert(0, d)
    d = start
    while len(days) < back + n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def make_store(symbols, start: date, n_days: int, warmup_days: int = 60, trends=None, seed: int = 0,
               news_every_days: int = 3, early_close: date | None = None) -> HistoricalStore:
    trends = trends or {}
    days = weekdays(start, n_days, warmup_days)
    hourly, daily, news = {}, {}, {}
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i)
        drift = trends.get(sym, 0.0006 if i % 2 == 0 else -0.0004)
        idx, rows = [], []
        price = 100.0 + 15 * i
        for d in days:
            last_hour = 12 if d == early_close else 15
            for h in range(9, last_hour + 1):
                o = price
                c = o * (1 + drift + rng.normal(0, 0.004))
                hi = max(o, c) * (1 + rng.uniform(0, 0.002))
                lo = min(o, c) * (1 - rng.uniform(0, 0.002))
                rows.append((o, hi, lo, c, float(rng.integers(50_000, 150_000))))
                idx.append(datetime.combine(d, time(h), tzinfo=NY).astimezone(timezone.utc))
                price = c
        hdf = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))
        hourly[sym] = hdf
        ny = hdf.index.tz_convert(NY)
        g = hdf.groupby(ny.date)
        ddf = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                            "close": g["close"].last(), "volume": g["volume"].sum()})
        ddf.index = pd.DatetimeIndex([datetime.combine(d, time(0), tzinfo=NY) for d in ddf.index]).tz_convert("UTC")
        daily[sym] = ddf
        items = []
        for k, d in enumerate(days):
            if k % news_every_days == i % news_every_days:
                items.append(NewsItem(f"{sym}-{k}", f"{sym} news {d}", "test",
                                      datetime.combine(d, time(11, 20), tzinfo=NY)))
        news[sym] = items
    return HistoricalStore(hourly, daily, news)
