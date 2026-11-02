"""Backtest portfelowy: point-in-time, symulowany broker, silnik, wznawianie, replay LLM, walk-forward."""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from trader.agent.claude_cli import CLIResult
from trader.backtest.engine import BacktestEngine, BacktestPaused, BacktestSpec, LookAheadForbidden
from trader.backtest.grid import parse_pairs, run_grid, write_grid_report
from trader.backtest.history import PointInTimeMarketData
from trader.backtest.report import write_report
from trader.backtest.sim_broker import SimBroker
from trader.broker import NY, OrderRequest
from trader.evaluation import stats
from tests.conftest import FakeCLI, ticker_from_prompt
from tests.synthetic import make_store

START = date(2026, 8, 3)        # po granicy wiedzy claude-opus-5-5 (+ bufor) z config.yaml


@pytest.fixture
def store(cfg):
    return make_store(cfg.watchlist, START, 8)


@pytest.fixture
def bcfg(cfg, tmp_path):
    cfg.backtest.dir = str(tmp_path / "bt")
    return cfg


def at(d: date, h: int, m: int = 0) -> datetime:
    return datetime.combine(d, time(h, m), tzinfo=NY)


# --------------------------------------------------------------------------- point-in-time
def test_point_in_time_has_no_look_ahead(store):
    md = PointInTimeMarketData(store, spread_bps=2)
    for as_of in [at(START, 10), at(START, 13), at(START + timedelta(days=3), 15)]:
        md.as_of = as_of
        h = md.hourly_bars(["AAPL"], as_of - timedelta(days=30), as_of + timedelta(days=30))["AAPL"]
        assert (h.index + pd.Timedelta(hours=1)).max() <= pd.Timestamp(as_of)        # tylko zamknięte świece
        d = md.daily_bars(["AAPL"], as_of - timedelta(days=200), as_of + timedelta(days=30))["AAPL"]
        assert max(d.index.tz_convert(NY).date) < as_of.date()                        # bez dzisiejszej świecy
        n = md.news("AAPL", as_of - timedelta(days=30), 50)
        assert all(x.created_at <= as_of for x in n)
        q = md.latest_quotes(["AAPL"])["AAPL"]
        assert q.last == pytest.approx(float(h["close"].iloc[-1]))
        assert q.bid < q.last < q.ask


def test_news_published_after_as_of_is_invisible(store):
    md = PointInTimeMarketData(store)
    md.as_of = at(START, 11)                      # newsy syntetyczne mają godzinę 11:20
    assert not [n for n in md.news("SPY", at(START, 0), 50) if n.created_at.date() == START]
    md.as_of = at(START, 12)
    assert any(n.created_at.date() == START for n in md.news("SPY", at(START, 0), 50))


def test_early_close_session_hours(cfg):
    early = START + timedelta(days=2)
    s = make_store(["SPY"], START, 5, early_close=early)
    assert s.session_hours("SPY", early)[1].hour == 13
    assert s.session_hours("SPY", START)[1].hour == 16


# --------------------------------------------------------------------------- symulowany broker
def test_sim_broker_fill_stop_gap_and_state(store):
    b = SimBroker("llm", store, 10_000, spread_bps=2, calendar_symbol="SPY")
    b.advance_to(at(START, 10))
    b.begin_day()
    last = store.last_close("AAPL", at(START, 10))
    o = b.submit_order(OrderRequest("AAPL", "buy", 5, "limit", "c1", limit_price=round(last * 1.001, 2)))
    assert o.status == "filled" and o.filled_avg_price == pytest.approx(last * 1.0001, rel=1e-6)
    assert b.cash == pytest.approx(10_000 - 5 * last * 1.0001)
    # stop powyżej rynku zadziała na pierwszej kolejnej świecy po cenie min(stop, open)
    st = b.submit_order(OrderRequest("AAPL", "sell", 5, "stop", "c2", stop_price=last * 1.5, time_in_force="gtc"))
    saved = b.to_state()
    fills = b.advance_to(at(START, 12))
    assert fills and fills[0]["status"] == "filled"
    bar = store.bars_between("AAPL", at(START, 10), at(START, 12)).iloc[0]
    assert b.get_order(st.id).filled_avg_price == pytest.approx(min(last * 1.5, bar["open"]) * 0.9999, rel=1e-6)
    assert "AAPL" not in b.get_positions()
    # odtworzenie stanu
    b2 = SimBroker("llm", store, 1, 2, "SPY")
    b2.load_state(saved)
    assert b2.get_positions()["AAPL"].qty == 5 and len(b2.get_open_orders()) == 1


def test_sim_broker_rejects_without_cash_or_shares(store):
    b = SimBroker("llm", store, 100, 2, "SPY")
    b.advance_to(at(START, 10))
    with pytest.raises(RuntimeError, match="buying power"):
        b.submit_order(OrderRequest("AAPL", "buy", 50, "limit", "c", limit_price=10_000))
    with pytest.raises(RuntimeError, match="insufficient qty"):
        b.submit_order(OrderRequest("AAPL", "sell", 1, "limit", "c", limit_price=1))


# --------------------------------------------------------------------------- silnik (SMA)
def test_sma_backtest_runs_and_reports(bcfg, store):
    res = BacktestEngine(bcfg, store, BacktestSpec("sma", START, START + timedelta(days=20), ["sma"], fresh=True),
                         progress=lambda m: None).run()
    assert res.days == 8 and res.slots == 8 * 6
    assert len(res.equity["sma"]) == 9 and len(res.equity["spy"]) == 9   # start + 8 sesji
    db = res.extra["db"]
    assert db.query_one("SELECT COUNT(*) AS n FROM trades WHERE agent='sma'")["n"] > 0
    # wszystkie decyzje w czasie symulowanym (godziny slotów), nie w czasie rzeczywistym
    ts = [datetime.fromisoformat(r["created_at"]) for r in db.query("SELECT created_at FROM decisions")]
    assert min(ts).astimezone(NY).date() >= START and max(ts).astimezone(NY).hour <= 15
    path = write_report(res, db)
    text = path.read_text()
    assert "SMA vs SPY" in text and (res.out_dir / "equity.png").exists() and (res.out_dir / "trades.csv").exists()


def test_backtest_is_deterministic_and_resumable(bcfg, store):
    end = START + timedelta(days=20)
    full = BacktestEngine(bcfg, store, BacktestSpec("a", START, end, ["sma"], fresh=True),
                          progress=lambda m: None).run()
    # przerwany po 3 sesjach, potem wznowiony z tą samą nazwą
    BacktestEngine(bcfg, store, BacktestSpec("b", START, START + timedelta(days=2), ["sma"], fresh=True),
                   progress=lambda m: None).run()
    resumed = BacktestEngine(bcfg, store, BacktestSpec("b", START, end, ["sma"]), progress=lambda m: None).run()
    assert resumed.equity["sma"].round(6).tolist() == full.equity["sma"].round(6).tolist()
    assert resumed.equity["spy"].round(6).tolist() == full.equity["spy"].round(6).tolist()


def test_resume_refuses_other_config(bcfg, store):
    BacktestEngine(bcfg, store, BacktestSpec("c", START, START + timedelta(days=1), ["sma"], fresh=True),
                   progress=lambda m: None).run()
    bcfg.sma.fast = 10
    with pytest.raises(RuntimeError, match="innej konfiguracji"):
        BacktestEngine(bcfg, store, BacktestSpec("c", START, START + timedelta(days=5), ["sma"]),
                       progress=lambda m: None).run()


# --------------------------------------------------------------------------- replay LLM
def decide(prompt):
    t = ticker_from_prompt(prompt)
    buy = t in ("AAPL", "NVDA", "SPY")
    return {"ticker": t, "action": "BUY" if buy else "HOLD", "size_pct": 5 if buy else 0,
            "confidence": 0.7, "reasoning": f"replay {t}"}


def test_llm_replay_refuses_data_before_knowledge_cutoff(bcfg, store):
    with pytest.raises(LookAheadForbidden, match="look-ahead|niedozwolony"):
        BacktestEngine(bcfg, store, BacktestSpec("x", date(2026, 6, 1), START, ["llm"]), llm_cli=FakeCLI(decide))
    bcfg.llm.model = "claude-unknown-9"
    with pytest.raises(LookAheadForbidden, match="Brak granicy wiedzy"):
        BacktestEngine(bcfg, store, BacktestSpec("x", START, START, ["llm"]), llm_cli=FakeCLI(decide))


def test_llm_replay_runs_with_point_in_time_prompts(bcfg, store):
    cli = FakeCLI(decide)
    res = BacktestEngine(bcfg, store, BacktestSpec("llm", START, START + timedelta(days=6), ["llm", "sma"],
                                                   fresh=True), llm_cli=cli, progress=lambda m: None).run()
    assert res.days == 5 and len(cli.calls) >= len(bcfg.watchlist)
    db = res.extra["db"]
    assert db.query_one("SELECT COUNT(*) AS n FROM trades WHERE agent='llm'")["n"] > 0
    # prompt z 10:00 nie może zawierać newsa z 11:20 tego samego dnia
    first_day = [c["prompt"] for c in cli.calls if f"Time: {START:%Y-%m-%d} 10:00" in c["prompt"]]
    assert first_day and all(f"news {START}" not in p for p in first_day)
    assert "LLM vs SPY" in write_report(res, db).read_text()


def test_llm_failure_pauses_and_resume_matches_uninterrupted_run(bcfg, store):
    end = START + timedelta(days=6)
    ref = BacktestEngine(bcfg, store, BacktestSpec("ref", START, end, ["llm", "sma"], fresh=True),
                         llm_cli=FakeCLI(decide), progress=lambda m: None).run()

    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 25:
            return CLIResult(False, error="CLI: usage limit (HTTP 429)", error_status=429)
        return decide(prompt)

    with pytest.raises(BacktestPaused, match="wznowić"):
        BacktestEngine(bcfg, store, BacktestSpec("flaky", START, end, ["llm", "sma"], fresh=True),
                       llm_cli=FakeCLI(flaky), progress=lambda m: None).run()
    resumed = BacktestEngine(bcfg, store, BacktestSpec("flaky", START, end, ["llm", "sma"]),
                             llm_cli=FakeCLI(decide), progress=lambda m: None).run()
    db = resumed.extra["db"]
    assert db.query_one("SELECT COUNT(*) AS n FROM llm_calls WHERE error IS NOT NULL")["n"] == 0
    assert db.query_one("SELECT COUNT(*) AS n FROM runs WHERE status != 'ok'")["n"] == 0
    for a in ("llm", "sma", "spy"):
        assert resumed.equity[a].round(6).tolist() == ref.equity[a].round(6).tolist()


def test_parallel_llm_calls_give_identical_results(bcfg, store):
    end = START + timedelta(days=3)
    seq = BacktestEngine(bcfg, store, BacktestSpec("seq", START, end, ["llm"], fresh=True),
                         llm_cli=FakeCLI(decide), progress=lambda m: None).run()
    bcfg.llm.max_parallel = 4
    par = BacktestEngine(bcfg, store, BacktestSpec("par", START, end, ["llm"], fresh=True),
                         llm_cli=FakeCLI(decide), progress=lambda m: None).run()
    assert par.equity["llm"].round(6).tolist() == seq.equity["llm"].round(6).tolist()


# --------------------------------------------------------------------------- walk-forward / statystyki
def test_sma_grid_and_walk_forward(bcfg, tmp_path):
    store = make_store(bcfg.watchlist, START, 12)
    res = run_grid(bcfg, store, START, START + timedelta(days=20), parse_pairs("10/30,20/50"), folds=3,
                   progress=lambda m: None)
    assert set(res.curves) == {"10/30", "20/50", "SPY"}
    assert set(res.table["para"]) == {"10/30", "20/50", "SPY"} and res.table["fold"].max() == 3
    assert len(res.walk_forward) == 2 and set(res.walk_forward["wybrana_para"]) <= {"10/30", "20/50"}
    text = write_grid_report(res, tmp_path / "grid", START, START + timedelta(days=20)).read_text()
    assert "Walk-forward" in text


def _eq(returns, start=10_000.0):
    idx = pd.bdate_range("2026-01-02", periods=len(returns) + 1, tz=NY)
    return pd.Series(start * np.cumprod(np.concatenate([[1.0], 1 + np.asarray(returns)])), index=idx)


def test_stats_detects_real_edge_and_ignores_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0004, 0.01, 500)
    strong = stats.compare(_eq(base + 0.002 + rng.normal(0, 0.002, 500)), _eq(base))
    assert strong["p_bootstrap"] < 0.05 and strong["t_stat"] > 2 and "istotna" in stats.verdict(strong)
    noise = stats.compare(_eq(base + rng.normal(0, 0.01, 500)), _eq(base))
    assert noise["p_bootstrap"] > 0.05 and "nieistotna" in stats.verdict(noise)
    assert noise["days_needed_t2"] is None or noise["days_needed_t2"] > 500
    few = stats.compare(_eq(base[:3]), _eq(base[:3]))
    assert few["enough"] is False and "za mało" in stats.verdict(few)


def test_ctrl_c_rolls_back_slot_and_resume_matches(bcfg, store):
    end = START + timedelta(days=4)
    ref = BacktestEngine(bcfg, store, BacktestSpec("ref2", START, end, ["llm", "sma"], fresh=True),
                         llm_cli=FakeCLI(decide), progress=lambda m: None).run()
    calls = {"n": 0}

    def interrupt(prompt):
        calls["n"] += 1
        if calls["n"] == 15:
            raise KeyboardInterrupt
        return decide(prompt)

    with pytest.raises(BacktestPaused, match="przerwano"):
        BacktestEngine(bcfg, store, BacktestSpec("ctrlc", START, end, ["llm", "sma"], fresh=True),
                       llm_cli=FakeCLI(interrupt), progress=lambda m: None).run()
    resumed = BacktestEngine(bcfg, store, BacktestSpec("ctrlc", START, end, ["llm", "sma"]),
                             llm_cli=FakeCLI(decide), progress=lambda m: None).run()
    for a in ("llm", "sma", "spy"):
        assert resumed.equity[a].round(6).tolist() == ref.equity[a].round(6).tolist()


def test_hard_kill_mid_slot_requires_fresh(bcfg, store):
    res = BacktestEngine(bcfg, store, BacktestSpec("kill", START, START + timedelta(days=1), ["sma"], fresh=True),
                         progress=lambda m: None).run()
    res.extra["db"].conn.execute("UPDATE runs SET status = 'running' WHERE id = (SELECT MAX(id) FROM runs)")
    with pytest.raises(RuntimeError, match="--fresh"):
        BacktestEngine(bcfg, store, BacktestSpec("kill", START, START + timedelta(days=3), ["sma"]),
                       progress=lambda m: None).run()
