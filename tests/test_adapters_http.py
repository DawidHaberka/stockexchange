"""Testy PRAWDZIWYCH adapterów (alpaca-py, SDK anthropic) na atrapie HTTP — bez sieci.

Sprawdzają, że nasz kod poprawnie buduje żądania i parsuje odpowiedzi w formacie API,
czyli dokładnie to, czego nie pokrywają testy na FakeBroker.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from trader.broker import AlpacaBroker, OrderRequest
from trader.config import AlpacaCredentials, Secrets
from trader.data.market_data import AlpacaMarketData, filter_rth_hourly
from trader.runner import Runner
from tests.conftest import NOW, FakeMarketData
from tests.fake_http import FakeAlpacaHTTP, FakeClaudeHTTP, patch_broker, patch_market_data

PAPER = "https://paper-api.alpaca.markets"


def make_broker(name, fake):
    b = AlpacaBroker(name, AlpacaCredentials(f"key-{name}", f"secret-{name}", PAPER))
    patch_broker(b, fake)
    return b


def make_md(fake):
    md = AlpacaMarketData(AlpacaCredentials("key", "secret", PAPER), "iex")
    patch_market_data(md, fake)
    return md


def decide(ticker, body):
    action = "BUY" if ticker in ("AAPL", "MSFT") else "HOLD"
    return {"ticker": ticker, "action": action, "size_pct": 5 if action == "BUY" else 0,
            "confidence": 0.7, "reasoning": f"Test {ticker}."}


@pytest.fixture
def world(cfg):
    fmd = FakeMarketData(cfg.watchlist)
    llm_http = FakeAlpacaHTTP(NOW, fmd, account_number="PA3LLM0001")
    sma_http = FakeAlpacaHTTP(NOW, fmd, account_number="PA3SMA0001")
    brokers = {"llm": make_broker("llm", llm_http), "sma": make_broker("sma", sma_http)}
    md = make_md(llm_http)
    claude = FakeClaudeHTTP(decide)
    return SimpleNamespace(fmd=fmd, llm_http=llm_http, sma_http=sma_http, brokers=brokers, md=md, claude=claude)


# --------------------------------------------------------------------------- broker
def test_broker_account_clock_calendar_asset(world):
    b = world.brokers["llm"]
    acc = b.get_account()
    assert acc.account_number == "PA3LLM0001" and acc.equity == 10_000 and acc.cash == 10_000
    assert acc.status == "active" and not acc.trading_blocked
    clock = b.get_clock()
    assert clock.is_open and clock.next_close.hour == 16
    sess = b.get_session(NOW.astimezone(clock.timestamp.tzinfo).date())
    assert sess.open.hour == 9 and sess.open.minute == 30 and sess.close.hour == 16
    a = b.get_asset("AAPL")
    assert a.asset_class == "us_equity" and a.tradable and a.status == "active"


def test_broker_order_lifecycle(world):
    b = world.brokers["llm"]
    o = b.submit_order(OrderRequest("AAPL", "buy", 3, "limit", "cid-1", limit_price=200.12))
    assert o.status == "accepted" and o.order_type == "limit" and o.side == "buy" and o.qty == 3
    body = world.llm_http.requests[-1][2]
    assert body["type"] == "limit" and body["time_in_force"] == "day" and float(body["limit_price"]) == 200.12
    filled = b.get_order(o.id)
    assert filled.status == "filled" and filled.filled_qty == 3 and filled.filled_avg_price == 200.12
    pos = b.get_positions()["AAPL"]
    assert pos.qty == 3 and pos.avg_entry_price == pytest.approx(200.12)

    stop = b.submit_order(OrderRequest("AAPL", "sell", 3, "stop", "cid-2", stop_price=195.5, time_in_force="gtc"))
    body = world.llm_http.requests[-1][2]
    assert body["type"] == "stop" and body["time_in_force"] == "gtc" and float(body["stop_price"]) == 195.5
    assert [x.id for x in b.get_open_orders(["AAPL"])] == [stop.id]
    b.cancel_order(stop.id)
    assert b.get_order(stop.id).status == "canceled"
    assert b.get_open_orders() == []


def test_broker_api_error_is_raised(world):
    world.llm_http.reject_symbols.add("AAPL")
    with pytest.raises(Exception, match="insufficient buying power"):
        world.brokers["llm"].submit_order(OrderRequest("AAPL", "buy", 1, "limit", "cid", limit_price=200.0))


# --------------------------------------------------------------------------- market data
def test_market_data_bars_quotes_news(world, cfg):
    md = world.md
    hourly = md.hourly_bars(["AAPL", "MSFT"], NOW - timedelta(days=30), NOW)
    assert set(hourly) == {"AAPL", "MSFT"}
    df = hourly["AAPL"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(filter_rth_hourly(df, NOW)) == len(df) > 100
    daily = md.daily_bars(["SPY"], NOW - timedelta(days=200), NOW)
    assert len(daily["SPY"]) == 120
    q = md.latest_quotes(["AAPL"])["AAPL"]
    assert q.bid and q.ask and q.last and q.bid < q.ask
    news = md.news("AAPL", NOW - timedelta(hours=24), 5)
    assert len(news) == 2 and news[0].headline.startswith("AAPL") and news[0].created_at.tzinfo


# --------------------------------------------------------------------------- pełne przebiegi
def test_dry_run_end_to_end_real_adapters(world, cfg, db):
    r = Runner(cfg, db, world.brokers, world.md, world.claude.client(), now_fn=lambda: NOW, sleep=lambda s: None)
    s = r.run(dry_run=True)
    assert s.status == "ok", s.reason
    # żadne zlecenie nie poszło do Alpaca
    for http in (world.llm_http, world.sma_http):
        assert not [x for x in http.requests if x[0] == "POST"]
    # kształt żądań do Claude
    assert len(world.claude.bodies) == len(cfg.watchlist)
    body, headers = world.claude.bodies[0], world.claude.headers[0]
    assert body["model"] == cfg.llm.model and "temperature" not in body
    assert body["output_config"]["effort"] == cfg.llm.effort
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    # decyzje i koszty w bazie
    llm = db.query("SELECT * FROM decisions WHERE agent='llm' AND dry_run=1")
    assert len(llm) == len(cfg.watchlist) and {d["source"] for d in llm} == {"llm"}
    calls = db.query("SELECT * FROM llm_calls")
    assert all(c["request_id"] == "req_test" and c["cost_usd"] > 0 for c in calls)
    assert db.query_one("SELECT COUNT(*) AS n FROM orders WHERE status='dry_run'")["n"] >= 2


def test_live_run_end_to_end_real_adapters(world, cfg, db):
    cfg.execution.orders_enabled = True
    r = Runner(cfg, db, world.brokers, world.md, world.claude.client(), now_fn=lambda: NOW, sleep=lambda s: None)
    s = r.run(dry_run=False)
    assert s.status == "ok", s.reason
    posts = [x[2] for x in world.llm_http.requests if x[0] == "POST"]
    limits = [p for p in posts if p["type"] == "limit"]
    stops = [p for p in posts if p["type"] == "stop"]
    assert {p["symbol"] for p in limits} == {"AAPL", "MSFT"}
    assert {p["symbol"] for p in stops} == {"AAPL", "MSFT"} and all(p["time_in_force"] == "gtc" for p in stops)
    for p in stops:
        pos = world.llm_http.positions[p["symbol"]]
        assert float(p["qty"]) == float(pos["qty"])
        assert float(p["stop_price"]) < float(pos["avg_entry_price"])
    assert len(db.trades("llm")) == 2
    start = db.get_experiment_start("llm")
    assert start["start_equity"] == 10_000 and start["account_number"] == "PA3LLM0001"
    # SMA handluje wyłącznie na swoim koncie (konto LLM dostało tylko zlecenia AAPL/MSFT — patrz wyżej)
    assert [x for x in world.sma_http.requests if x[0] == "POST"]
    assert db.get_experiment_start("sma")["account_number"] == "PA3SMA0001"


def test_rejected_order_end_to_end(world, cfg, db):
    cfg.execution.orders_enabled = True
    world.llm_http.reject_symbols.add("AAPL")
    r = Runner(cfg, db, world.brokers, world.md, world.claude.client(), now_fn=lambda: NOW, sleep=lambda s: None)
    assert r.run(dry_run=False).status == "ok"
    row = db.query_one("SELECT * FROM orders WHERE agent='llm' AND symbol='AAPL' AND purpose='entry'")
    assert row["status"] == "rejected" and "insufficient buying power" in row["note"]


# --------------------------------------------------------------------------- --check
def test_check_command_offline(world, cfg, monkeypatch, capsys):
    import main
    import trader.agent.llm_agent as llm_mod

    monkeypatch.setattr(main, "build_brokers", lambda c, s, agents: {a: world.brokers[a] for a in agents})
    monkeypatch.setattr(main, "build_market_data", lambda c, s: world.md)
    monkeypatch.setattr(llm_mod, "make_anthropic_client", lambda c, k: world.claude.client())
    secrets = Secrets(PAPER, {}, "test-key")
    args = SimpleNamespace(config="config.yaml", agents="llm,sma")
    rc = main.cmd_check(cfg, secrets, args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PA3LLM0001" in out and "PA3SMA0001" in out and "WYNIK: OK" in out
    assert "Claude API: model claude-opus-5-5 dostępny" in out


def test_check_detects_same_account(world, cfg, monkeypatch, capsys):
    import main
    import trader.agent.llm_agent as llm_mod

    same = {"llm": world.brokers["llm"], "sma": world.brokers["llm"]}
    monkeypatch.setattr(main, "build_brokers", lambda c, s, agents: {a: same[a] for a in agents})
    monkeypatch.setattr(main, "build_market_data", lambda c, s: world.md)
    monkeypatch.setattr(llm_mod, "make_anthropic_client", lambda c, k: world.claude.client())
    rc = main.cmd_check(cfg, Secrets(PAPER, {}, "k"), SimpleNamespace(config="config.yaml", agents="llm,sma"))
    assert rc == 1 and "TO SAMO konto" in capsys.readouterr().out
