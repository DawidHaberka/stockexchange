import pytest

from trader.broker import AlpacaBroker
from trader.config import AlpacaCredentials
from trader.safety import PAPER_URL, LiveTradingForbidden, assert_paper_url


@pytest.mark.parametrize("url", [
    "https://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets/",
    "https://paper-api.alpaca.markets/v2",
    "  https://PAPER-API.alpaca.markets  ",
])
def test_paper_urls_accepted(url):
    assert assert_paper_url(url) == PAPER_URL


@pytest.mark.parametrize("url", [
    "https://api.alpaca.markets",                       # konto LIVE
    "http://paper-api.alpaca.markets",                  # bez TLS
    "https://paper-api.alpaca.markets.evil.com",
    "https://evil.com/paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets:8443",
    "https://user:pw@paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets/v2/orders",
    "https://paper-api.alpaca.markets?x=1",
    "",
    None,
])
def test_non_paper_urls_rejected(url):
    with pytest.raises(LiveTradingForbidden):
        assert_paper_url(url)


def test_broker_refuses_live_url():
    with pytest.raises(LiveTradingForbidden):
        AlpacaBroker("llm", AlpacaCredentials("k", "s", "https://api.alpaca.markets"))


def test_broker_accepts_paper_url_without_network():
    b = AlpacaBroker("llm", AlpacaCredentials("k", "s", "https://paper-api.alpaca.markets"))
    assert "paper-api.alpaca.markets" in str(b.client._base_url)


def test_main_exits_with_error_on_live_url(tmp_path, monkeypatch):
    import main

    from tests.conftest import ROOT

    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")   # konto LIVE
    monkeypatch.setenv("ALPACA_LLM_API_KEY", "a")
    monkeypatch.setenv("ALPACA_LLM_SECRET_KEY", "b")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text((ROOT / "config.yaml").read_text().replace("runtime/", str(tmp_path) + "/runtime/"))
    rc = main.main(["--dry-run", "--config", str(cfg_file), "--env", str(tmp_path / "missing.env")])
    assert rc == 2
