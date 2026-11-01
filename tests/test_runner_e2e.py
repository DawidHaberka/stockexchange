"""Pełny przebieg orkiestratora na fałszywych API: dry-run, live, limit straty, raport."""

from datetime import timedelta

import pytest

from trader.evaluation.report import build_report, collect_eod, compute_metrics
from trader.runner import Runner
from tests.conftest import NOW, FakeBroker, FakeCLI, FakeMarketData, ticker_from_prompt


def responder(prompt):
    t = ticker_from_prompt(prompt)
    action = "BUY" if t in ("AAPL", "MSFT", "NVDA") else "HOLD"
    return {"ticker": t, "action": action, "size_pct": 5 if action == "BUY" else 0,
            "confidence": 0.75, "reasoning": f"Test {t}."}


@pytest.fixture
def setup(cfg, db):
    md = FakeMarketData(cfg.watchlist)
    brokers = {"llm": FakeBroker("llm"), "sma": FakeBroker("sma")}
    client = FakeCLI(responder)

    def make(now=NOW):
        return Runner(cfg, db, brokers, md, client, now_fn=lambda: now, sleep=lambda s: None)
    return make, brokers, client, md


def test_dry_run_full_pipeline(cfg, db, setup):
    make, brokers, client, _ = setup
    s = make().run(dry_run=True)
    assert s.status == "ok" and s.mode == "dry_run"
    assert len(client.calls) == len(cfg.watchlist)              # pierwsza ocena każdego tickera
    assert all(b.submitted == [] for b in brokers.values())     # NIC nie wysłano
    n = db.query_one("SELECT COUNT(*) AS n FROM decisions WHERE dry_run = 1")["n"]
    assert n == 2 * len(cfg.watchlist)
    row = db.query_one("SELECT * FROM decisions WHERE agent='llm' LIMIT 1")
    assert row["model"] == cfg.llm.model and row["effort"] == cfg.llm.effort
    assert row["frequency"] == cfg.experiment.frequency and row["prompt_hash"] == cfg.llm.prompt_hash
    assert db.query_one("SELECT COUNT(*) AS n FROM orders WHERE status='dry_run'")["n"] >= 3
    assert db.get_experiment_start("llm") is None               # start ustawia dopiero live
    assert s.llm_calls == len(cfg.watchlist) and s.llm_cost_equiv_usd > 0


def test_orders_disabled_forces_dry_run(cfg, db, setup):
    make, brokers, _, _ = setup
    cfg.execution.orders_enabled = False
    s = make().run(dry_run=False)
    assert s.mode == "dry_run" and brokers["llm"].submitted == []


def test_live_run_places_orders_and_stops(cfg, db, setup):
    make, brokers, _, _ = setup
    cfg.execution.orders_enabled = True
    s = make().run(dry_run=False)
    assert s.status == "ok" and s.mode == "live"
    llm = brokers["llm"]
    assert {r.symbol for r in llm.submitted if r.order_type == "limit"} == {"AAPL", "MSFT", "NVDA"}
    assert {r.symbol for r in llm.submitted if r.order_type == "stop"} == {"AAPL", "MSFT", "NVDA"}
    start = db.get_experiment_start("llm")
    assert start is not None and start["start_equity"] == pytest.approx(10_000)
    assert db.get_experiment_start("sma")["start_equity"] == pytest.approx(10_000)
    for p in llm.get_positions().values():
        assert p.market_value <= 10_000 * 0.10 + 1


def test_same_slot_not_processed_twice(cfg, db, setup):
    make, _, client, _ = setup
    make().run(dry_run=True)
    s = make(NOW + timedelta(minutes=15)).run(dry_run=True)
    assert s.status == "skipped"


def test_next_hour_without_triggers_has_no_api_calls(cfg, db, setup):
    make, _, client, _ = setup
    make().run(dry_run=True)
    calls = len(client.calls)
    s = make(NOW + timedelta(hours=1)).run(dry_run=True)
    assert s.status == "ok" and len(client.calls) == calls      # te same ceny, brak newsów
    assert db.query_one("SELECT COUNT(*) AS n FROM decisions WHERE source='no_trigger'")["n"] == len(cfg.watchlist)


def test_daily_loss_limit_halts_trading(cfg, db, setup):
    make, brokers, client, _ = setup
    cfg.execution.orders_enabled = True
    brokers["llm"].set_equity(9_650)                            # -3.5% vs last_equity 10 000
    s = make().run(dry_run=False)
    assert s.agents["llm"].halted
    assert brokers["llm"].submitted == [] and len(client.calls) == 0
    assert db.is_halted("llm", "2026-11-02")[0]
    assert not s.agents["sma"].halted


def test_force_only_with_dry_run_outside_session(cfg, db, setup):
    make, brokers, _, _ = setup
    for b in brokers.values():
        b.is_open = False
    assert make().run(dry_run=True).status == "skipped"
    assert make().run(dry_run=True, force=True).status == "ok"
    cfg.execution.orders_enabled = True
    assert make().run(dry_run=False, force=True).status == "error"


def test_report_after_live_run(cfg, db, setup):
    make, brokers, _, md = setup
    cfg.execution.orders_enabled = True
    make().run(dry_run=False)
    session_date = collect_eod(cfg, db, brokers, md, NOW + timedelta(hours=7))
    res = build_report(cfg, db, session_date)
    assert "LLM" in res.markdown and "SPY buy&hold" in res.markdown and "Wynik przy cenach API" in res.markdown
    assert res.md_path.exists()
    m = compute_metrics(cfg, db)
    assert m["agents"]["llm"]["llm_cost_equiv_usd"] > 0
    assert m["agents"]["llm"]["n_trades"] == 3
    assert m["agents"]["llm"]["net_pnl_api_equiv_usd"] == pytest.approx(
        m["agents"]["llm"]["gross_pnl_usd"] - m["agents"]["llm"]["llm_cost_equiv_usd"])
    assert m["agents"]["sma"]["llm_cost_equiv_usd"] == 0


def test_report_in_dry_run_is_provisional(cfg, db, setup):
    make, brokers, _, md = setup
    make().run(dry_run=True)
    session_date = collect_eod(cfg, db, brokers, md, NOW + timedelta(hours=1))
    res = build_report(cfg, db, session_date)
    assert "tymczasowe" in res.markdown
