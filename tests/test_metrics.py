import math

import pandas as pd
import pytest

from trader.data.indicators import atr, rsi, sma
from trader.evaluation.metrics import fifo_closed_trades, max_drawdown_pct, sharpe_ratio, total_return_pct, win_rate_pct


def test_total_return_and_drawdown():
    eq = pd.Series([100, 110, 99, 120])
    assert total_return_pct(eq) == pytest.approx(20.0)
    assert max_drawdown_pct(eq) == pytest.approx((99 / 110 - 1) * 100)


def test_sharpe_none_for_flat_or_short():
    assert sharpe_ratio(pd.Series([100, 100, 100])) is None
    assert sharpe_ratio(pd.Series([100])) is None


def test_sharpe_positive_for_uptrend():
    eq = pd.Series([100 * (1.001 + 0.002 * ((-1) ** i)) ** i for i in range(30)])
    assert sharpe_ratio(eq) is not None


def test_fifo_win_rate():
    trades = [
        {"symbol": "A", "side": "buy", "qty": 10, "price": 100, "filled_at": "1"},
        {"symbol": "A", "side": "buy", "qty": 10, "price": 110, "filled_at": "2"},
        {"symbol": "A", "side": "sell", "qty": 15, "price": 108, "filled_at": "3"},   # 10@100 + 5@110 -> zysk
        {"symbol": "A", "side": "sell", "qty": 5, "price": 105, "filled_at": "4"},    # 5@110 -> strata
    ]
    closed = fifo_closed_trades(trades)
    assert len(closed) == 2
    assert closed[0].entry_price == pytest.approx((10 * 100 + 5 * 110) / 15)
    assert win_rate_pct(closed) == 50.0


def test_indicators_basic():
    close = pd.Series([float(i) for i in range(1, 61)])
    assert sma(close, 20).iloc[-1] == pytest.approx(sum(range(41, 61)) / 20)
    assert rsi(close, 14).iloc[-1] == pytest.approx(100.0)
    a = atr(close + 1, close - 1, close, 14).iloc[-1]
    assert a == pytest.approx(2.0, rel=1e-6)
    assert math.isnan(sma(close, 100).iloc[-1])
