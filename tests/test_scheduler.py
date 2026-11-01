from datetime import date, datetime, timedelta

import pytest

from trader.broker import NY, ClockState, SessionHours
from trader.scheduler import evaluate_schedule

DAY = date(2026, 11, 2)
SESSION = SessionHours(DAY, datetime(2026, 11, 2, 9, 30, tzinfo=NY), datetime(2026, 11, 2, 16, 0, tzinfo=NY))
EARLY_CLOSE = SessionHours(DAY, SESSION.open, datetime(2026, 11, 2, 13, 0, tzinfo=NY))


def at(h, m=0):
    return datetime(2026, 11, 2, h, m, tzinfo=NY)


def clock(is_open=True):
    return ClockState(at(10), is_open, at(9, 30) + timedelta(days=1), at(16))


def never(_):
    return False


@pytest.mark.parametrize("h, m, due, slot", [
    (9, 45, False, None),       # przed 10:00
    (10, 0, True, (10, 0)),
    (10, 14, True, (10, 0)),
    (11, 59, True, (11, 0)),
    (15, 0, True, (15, 0)),     # ostatni slot godzinowy
    (15, 29, True, (15, 0)),
    (15, 31, False, None),      # < 30 min do zamknięcia
])
def test_hourly_slots(cfg, h, m, due, slot):
    d = evaluate_schedule(at(h, m), clock(), SESSION, cfg.experiment, never)
    assert d.due is due
    if slot:
        assert d.slot_start == at(*slot)


def test_market_closed(cfg):
    assert not evaluate_schedule(at(11), clock(False), SESSION, cfg.experiment, never).due


def test_slot_processed_only_once(cfg):
    done = {at(11)}
    d = evaluate_schedule(at(11, 15), clock(), SESSION, cfg.experiment, lambda s: s in done)
    assert not d.due and "już przetworzony" in d.reason


def test_30min_frequency(cfg):
    cfg.experiment.frequency = "30min"
    assert evaluate_schedule(at(10, 40), clock(), SESSION, cfg.experiment, never).slot_start == at(10, 30)
    assert evaluate_schedule(at(15, 30), clock(), SESSION, cfg.experiment, never).slot_start == at(15, 30)


def test_15min_frequency(cfg):
    cfg.experiment.frequency = "15min"
    assert evaluate_schedule(at(12, 50), clock(), SESSION, cfg.experiment, never).slot_start == at(12, 45)


def test_early_close_day(cfg):
    assert evaluate_schedule(at(12, 15), clock(), EARLY_CLOSE, cfg.experiment, never).due
    assert evaluate_schedule(at(12, 31), clock(), EARLY_CLOSE, cfg.experiment, never).due is False
