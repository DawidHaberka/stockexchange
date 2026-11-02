"""`main.py --backtest / --backtest-grid` end-to-end: alpaca-py na atrapie HTTP + fałszywy `claude`."""

import pytest

import main
import trader.data.market_data as md_mod
from trader.config import AlpacaCredentials
from tests.conftest import NOW, ROOT, FakeMarketData
from tests.fake_http import FakeAlpacaHTTP, patch_market_data


@pytest.fixture
def cli_env(tmp_path, monkeypatch, fake_claude):
    exe, claude_calls = fake_claude
    cfg_text = (ROOT / "config.yaml").read_text().replace("runtime/", f"{tmp_path}/runtime/")
    cfg_text = cfg_text.replace("cli_path: claude", f"cli_path: {exe}")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_LLM_API_KEY", "k")
    monkeypatch.setenv("ALPACA_LLM_SECRET_KEY", "s")
    monkeypatch.setenv("FAKE_CLAUDE_BUY", "AAPL,MSFT")

    from trader.config import load_config
    fmd = FakeMarketData(load_config(cfg_file).watchlist)
    http = FakeAlpacaHTTP(NOW, fmd)
    real = md_mod.AlpacaMarketData

    def factory(creds, feed="iex", adjustment="all"):
        m = real(AlpacaCredentials("k", "s", "https://paper-api.alpaca.markets"), feed, adjustment)
        patch_market_data(m, http)
        return m

    monkeypatch.setattr(md_mod, "AlpacaMarketData", factory)
    return cfg_file, tmp_path, http, claude_calls


def test_cli_sma_backtest_and_cache(cli_env, capsys):
    cfg_file, tmp, http, _ = cli_env
    args = ["--backtest", "--agents", "sma", "--start", "2026-10-26", "--end", "2026-10-30", "--name", "t1",
            "--config", str(cfg_file), "--env", str(tmp / "none.env")]
    assert main.main(args) == 0
    out = capsys.readouterr().out
    assert "# Backtest: t1" in out and "SMA vs SPY" in out
    assert (tmp / "runtime/backtests/t1/report.md").exists()
    n_requests = len(http.requests)
    assert main.main(args + ["--fresh"]) == 0          # drugi raz: historia z cache, bez pobierania
    assert len(http.requests) == n_requests


def test_cli_llm_replay(cli_env, capsys):
    cfg_file, tmp, _, claude_calls = cli_env
    rc = main.main(["--backtest", "--agents", "llm,sma", "--start", "2026-10-28", "--end", "2026-10-30",
                    "--name", "r1", "--parallel", "3", "--config", str(cfg_file), "--env", str(tmp / "none.env")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "LLM vs SPY" in out and "Wywołania Claude" in out
    assert len(claude_calls()) >= 11 and all(not c["has_api_key"] for c in claude_calls())


def test_cli_refuses_llm_replay_before_cutoff(cli_env, capsys):
    cfg_file, tmp, _, claude_calls = cli_env
    rc = main.main(["--backtest", "--agents", "llm", "--start", "2026-06-01", "--end", "2026-10-30",
                    "--config", str(cfg_file), "--env", str(tmp / "none.env")])
    assert rc == 2 and "niedozwolony" in capsys.readouterr().err
    assert claude_calls() == []


def test_cli_grid(cli_env, capsys):
    cfg_file, tmp, _, _ = cli_env
    rc = main.main(["--backtest-grid", "--grid", "5/15,10/30", "--folds", "2", "--start", "2026-10-26",
                    "--end", "2026-10-30", "--config", str(cfg_file), "--env", str(tmp / "none.env")])
    assert rc == 0 and "Walk-forward" in capsys.readouterr().out
