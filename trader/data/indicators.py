"""Wskaźniki techniczne liczone w czystym pandas (bez TA-Lib).

RSI i ATR w wariancie Wildera (wygładzanie wykładnicze alpha = 1/n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # brak strat -> RSI = 100 (o ile mamy dość danych)
    out = out.where(~(avg_loss == 0) | avg_gain.isna(), 100.0)
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    v = series.iloc[-1]
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)


@dataclass
class IndicatorSet:
    bars: int
    close: float | None
    sma_fast: float | None
    sma_slow: float | None
    rsi14: float | None
    atr14: float | None
    volume: float | None
    volume_avg20: float | None
    fast: int = 20
    slow: int = 50

    @property
    def volume_ratio(self) -> float | None:
        if self.volume is None or not self.volume_avg20:
            return None
        return self.volume / self.volume_avg20

    @property
    def atr_pct(self) -> float | None:
        if self.atr14 is None or not self.close:
            return None
        return self.atr14 / self.close * 100

    @property
    def complete(self) -> bool:
        return None not in (self.close, self.sma_fast, self.sma_slow, self.atr14)


def compute_indicators(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> IndicatorSet:
    """df: kolumny open/high/low/close/volume, indeks rosnący w czasie."""
    if df is None or df.empty:
        return IndicatorSet(0, None, None, None, None, None, None, None, fast, slow)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    return IndicatorSet(
        bars=len(df),
        close=_last(close),
        sma_fast=_last(sma(close, fast)),
        sma_slow=_last(sma(close, slow)),
        rsi14=_last(rsi(close, 14)),
        atr14=_last(atr(df["high"].astype(float), df["low"].astype(float), close, 14)),
        volume=_last(vol),
        volume_avg20=_last(vol.rolling(20, min_periods=20).mean()),
        fast=fast,
        slow=slow,
    )
