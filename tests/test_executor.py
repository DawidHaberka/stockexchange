"""Egzekucja: zlecenia limit, częściowe realizacje, odrzucenia, stop-lossy, dry-run."""

import pytest

from trader.execution.executor import Executor
from trader.risk.manager import OrderIntent, RiskVerdict
from tests.conftest import FakeBroker
from tests.helpers import dec, pos, snapshot


def buy_verdict(sym="AAPL", qty=4, limit=200.0, stop=192.0, atr=4.0):
    return RiskVerdict(dec(sym, "BUY", 8), "approved", "ok", OrderIntent(sym, "buy", qty, limit, stop, qty * limit, atr))


def sell_verdict(sym="AAPL", qty=4, limit=199.0):
    return RiskVerdict(dec(sym, "SELL", 100), "approved", "ok", OrderIntent(sym, "sell", qty, limit, None, qty * limit))


def make(cfg, db, broker, dry_run=False):
    h = db.register_config(cfg)
    for _ in range(3):                                  # przebiegi o id 1..3 (klucze obce)
        db.start_run("2026-11-02", None, "live", h)
    return Executor(broker, db, cfg, broker.name, dry_run=dry_run, sleep=lambda s: None)


def stops(broker, sym):
    return [o for o in broker.get_open_orders([sym]) if o.order_type == "stop"]


def test_full_fill_places_stop_for_position(cfg, db):
    b = FakeBroker()
    res = make(cfg, db, b).execute(1, [buy_verdict()], [None])[0]
    assert res.status == "filled" and res.filled_qty == 4
    st = stops(b, "AAPL")
    assert len(st) == 1 and st[0].qty == 4
    assert st[0].stop_price == pytest.approx(200.0 - 2 * 4.0)
    assert db.get_stop_level("llm", "AAPL") == pytest.approx(192.0)
    trades = db.trades("llm")
    assert len(trades) == 1 and trades[0]["qty"] == 4 and trades[0]["purpose"] == "entry"


def test_partial_fill_cancels_rest_and_protects_filled_qty(cfg, db):
    b = FakeBroker()
    b.fill_mode["AAPL"] = "partial"
    res = make(cfg, db, b).execute(1, [buy_verdict(qty=6)], [None])[0]
    assert res.status == "partially_filled" and res.filled_qty == 3
    assert b.canceled                                   # reszta anulowana
    assert stops(b, "AAPL")[0].qty == 3                 # stop tylko na zrealizowaną część
    row = db.query_one("SELECT * FROM orders WHERE purpose='entry'")
    assert row["status"] == "canceled" and row["filled_qty"] == 3 and row["fill_recorded"] == 1


def test_unfilled_order_is_canceled_without_stop(cfg, db):
    b = FakeBroker()
    b.fill_mode["AAPL"] = "none"
    res = make(cfg, db, b).execute(1, [buy_verdict()], [None])[0]
    assert res.status == "unfilled" and not stops(b, "AAPL")
    assert db.trades("llm") == []


def test_rejected_order_is_recorded(cfg, db):
    b = FakeBroker()
    b.fill_mode["AAPL"] = "reject"
    res = make(cfg, db, b).execute(1, [buy_verdict()], [None])[0]
    assert res.status == "rejected"
    row = db.query_one("SELECT * FROM orders")
    assert row["status"] == "rejected" and "buying power" in row["note"]


def test_sell_cancels_stop_first_then_closes(cfg, db):
    b = FakeBroker()
    ex = make(cfg, db, b)
    ex.execute(1, [buy_verdict(qty=4)], [None])
    assert len(stops(b, "AAPL")) == 1
    res = ex.execute(2, [sell_verdict(qty=4)], [None])[0]
    assert res.status == "filled"
    assert "AAPL" not in b.get_positions()
    assert not stops(b, "AAPL")
    assert db.get_stop_level("llm", "AAPL") is None


def test_partial_sell_restores_stop_for_remaining(cfg, db):
    b = FakeBroker()
    ex = make(cfg, db, b)
    ex.execute(1, [buy_verdict(qty=4)], [None])
    ex.execute(2, [sell_verdict(qty=1)], [None])
    st = stops(b, "AAPL")
    assert len(st) == 1 and st[0].qty == 3 and st[0].stop_price == pytest.approx(192.0)


def test_sells_execute_before_buys(cfg, db):
    b = FakeBroker(positions={"MSFT": pos("MSFT", 2, 400.0)}, cash=9_200)
    make(cfg, db, b).execute(1, [buy_verdict("AAPL"), sell_verdict("MSFT", 2, 399.0)], [None, None])
    limits = [r for r in b.submitted if r.order_type == "limit"]
    assert [r.side for r in limits] == ["sell", "buy"]


def test_dry_run_sends_nothing(cfg, db):
    b = FakeBroker()
    res = make(cfg, db, b, dry_run=True).execute(1, [buy_verdict()], [None])[0]
    assert res.status == "dry_run" and b.submitted == []
    row = db.query_one("SELECT * FROM orders")
    assert row["status"] == "dry_run" and row["dry_run"] == 1


def test_reconcile_records_stop_fill_between_runs(cfg, db):
    b = FakeBroker()
    ex = make(cfg, db, b)
    ex.execute(1, [buy_verdict(qty=4)], [None])
    b.trigger_stop("AAPL", 191.5)
    notes = ex.reconcile(2, snapshot({"AAPL": 190.0}))
    assert any("stop_loss" in n for n in notes)
    sells = [t for t in db.trades("llm") if t["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["purpose"] == "stop_loss" and sells[0]["price"] == 191.5


def test_reconcile_adds_missing_stop(cfg, db):
    b = FakeBroker(positions={"AAPL": pos("AAPL", 5, 200.0, entry=198.0)}, cash=9_000)
    notes = make(cfg, db, b).reconcile(1, snapshot({"AAPL": 200.0}, atr_1d=3.0))
    st = stops(b, "AAPL")
    assert len(st) == 1 and st[0].qty == 5 and st[0].stop_price == pytest.approx(198.0 - 2 * 3.0)
    assert notes


def test_reconcile_dry_run_does_not_modify(cfg, db):
    b = FakeBroker(positions={"AAPL": pos("AAPL", 5, 200.0)}, cash=9_000)
    notes = make(cfg, db, b, dry_run=True).reconcile(1, snapshot({"AAPL": 200.0}))
    assert notes and b.submitted == []


def test_reconcile_cancels_orphan_stop_without_position(cfg, db):
    b = FakeBroker()
    ex = make(cfg, db, b)
    ex.execute(1, [buy_verdict(qty=4)], [None])
    del b.positions["AAPL"]                      # pozycja zniknęła (np. ręczne zamknięcie)
    ex.reconcile(2, snapshot({"AAPL": 200.0}))
    assert not stops(b, "AAPL")
