"""Backtest WYŁĄCZNIE agenta SMA (backtesting.py).

Agenta LLM celowo nie backtestujemy: dane historyczne sprzed cutoffu wiedzy modelu
dawałyby look-ahead bias (model „zna” przyszłość tych dat). LLM oceniamy tylko
forward, na żywym paper tradingu.

Strategia odwzorowuje agenta live: świece 1H (sesja regularna), SMA fast/slow,
pozycja = target_size_pct equity, stop-loss = cena wejścia - k x ATR (ATR dzienny
liczony z poprzednich dni lub ATR 1H — wg risk.stop_loss_atr_timeframe), koszt
wykonania ~ limit_offset_bps na stronę.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from ..config import AppConfig
from ..data.indicators import atr as atr_series
from ..data.indicators import sma as sma_series
from ..data.market_data import MarketDataSource, filter_rth_hourly

log = logging.getLogger(__name__)


def daily_atr_on_hourly(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR(14) z dziennych świec zbudowanych z 1H, przesunięty o dzień (bez look-ahead) i rozlany na 1H."""
    idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    ny = df.copy()
    ny.index = idx.tz_convert("America/New_York")
    daily = ny.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    d_atr = atr_series(daily["High"], daily["Low"], daily["Close"], n).shift(1)
    mapped = d_atr.reindex(ny.index.normalize()).to_numpy()
    return pd.Series(mapped, index=df.index)


def make_strategy(cfg: AppConfig) -> type[Strategy]:
    fast, slow = cfg.sma.fast, cfg.sma.slow
    size = cfg.sma.target_size_pct / 100
    mult = cfg.risk.stop_loss_atr_mult
    use_daily_atr = cfg.risk.stop_loss_atr_timeframe == "1d"

    class SmaCrossLive(Strategy):
        def init(self):
            close = pd.Series(self.data.Close)
            self.fast = self.I(lambda: sma_series(close, fast).to_numpy(), name=f"SMA{fast}")
            self.slow = self.I(lambda: sma_series(close, slow).to_numpy(), name=f"SMA{slow}")
            if use_daily_atr:
                stop_atr = self.data.df["ATR_D"].to_numpy()
            else:
                stop_atr = atr_series(pd.Series(self.data.High), pd.Series(self.data.Low), close, 14).to_numpy()
            self.stop_atr = self.I(lambda: stop_atr, name="ATR_stop", overlay=False)

        def next(self):
            f, s, a = self.fast[-1], self.slow[-1], self.stop_atr[-1]
            if np.isnan(f) or np.isnan(s):
                return
            if f > s:
                if not self.position and not np.isnan(a) and a > 0:
                    price = self.data.Close[-1]
                    self.buy(size=size, sl=price - mult * a)
            elif self.position:
                self.position.close()

    return SmaCrossLive


@dataclass
class BacktestResult:
    symbol: str
    stats: pd.Series | None
    error: str | None = None


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    out = out[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    out["ATR_D"] = daily_atr_on_hourly(out)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    return out


def run_backtest(cfg: AppConfig, df: pd.DataFrame, symbol: str, cash: float = 10_000.0) -> BacktestResult:
    frame = prepare_frame(df)
    if len(frame) < cfg.sma.slow + 20:
        return BacktestResult(symbol, None, f"za mało danych ({len(frame)} świec)")
    bt = Backtest(frame, make_strategy(cfg), cash=cash,
                  commission=cfg.execution.limit_offset_bps / 1e4, finalize_trades=True)
    return BacktestResult(symbol, bt.run())


def backtest_watchlist(cfg: AppConfig, md: MarketDataSource, now: datetime, days: int = 365,
                       symbols: list[str] | None = None, cash: float = 10_000.0) -> list[BacktestResult]:
    symbols = symbols or cfg.watchlist
    bars = md.hourly_bars(symbols, now - timedelta(days=days), now)
    results = []
    for s in symbols:
        df = bars.get(s)
        if df is None or df.empty:
            results.append(BacktestResult(s, None, "brak danych"))
            continue
        try:
            results.append(run_backtest(cfg, filter_rth_hourly(df, now), s, cash))
        except Exception as exc:
            log.exception("Backtest %s", s)
            results.append(BacktestResult(s, None, str(exc)))
    return results


STAT_KEYS = [
    ("Return [%]", "Zwrot %"), ("Buy & Hold Return [%]", "Buy&Hold %"), ("Sharpe Ratio", "Sharpe"),
    ("Max. Drawdown [%]", "Max DD %"), ("# Trades", "Transakcje"), ("Win Rate [%]", "Win rate %"),
]


def results_table(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Ticker": r.symbol}
        for key, label in STAT_KEYS:
            row[label] = None if r.stats is None else r.stats.get(key)
        row["Uwagi"] = r.error or ""
        rows.append(row)
    return pd.DataFrame(rows)
