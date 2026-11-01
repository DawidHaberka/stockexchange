"""Newsy z Alpaca News API: pobieranie dla watchlisty i filtrowanie „nowych od ostatniego uruchomienia”."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .market_data import MarketDataSource, NewsItem

log = logging.getLogger(__name__)


def fetch_watchlist_news(md: MarketDataSource, symbols: list[str], now: datetime,
                         lookback_hours: int, limit_per_ticker: int) -> dict[str, list[NewsItem]]:
    start = now - timedelta(hours=lookback_hours)
    out: dict[str, list[NewsItem]] = {}
    for s in symbols:
        try:
            items = md.news(s, start, limit_per_ticker)
        except Exception as exc:  # brak newsów nie może zatrzymać przebiegu
            log.warning("News dla %s niedostępne: %s", s, exc)
            items = []
        out[s] = sorted(items, key=lambda n: n.created_at, reverse=True)[:limit_per_ticker]
    return out


def new_since(items: list[NewsItem], since: datetime | None) -> list[NewsItem]:
    if since is None:
        return list(items)
    return [n for n in items if n.created_at > since]
