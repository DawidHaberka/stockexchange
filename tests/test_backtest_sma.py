from datetime import timedelta

from trader.evaluation.backtest_sma import backtest_watchlist, results_table
from tests.conftest import NOW, FakeMarketData


def test_backtest_runs_on_synthetic_hourly_data(cfg):
    md = FakeMarketData(["AAPL", "XOM"], trends={"AAPL": 0.001, "XOM": -0.001})
    res = backtest_watchlist(cfg, md, NOW + timedelta(hours=1), days=60, symbols=["AAPL", "XOM"])
    table = results_table(res)
    assert list(table["Ticker"]) == ["AAPL", "XOM"]
    assert all(r.stats is not None for r in res), [r.error for r in res]
    assert "Zwrot %" in table.columns
