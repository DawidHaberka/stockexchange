"""Risk manager ma ostatnie słowo nad modelem."""

from datetime import timedelta

import pytest

from trader.broker import AccountState, AssetInfo, OrderState
from trader.risk.manager import RiskManager
from tests.conftest import NOW
from tests.helpers import dec, pos, snapshot

PRICES = {"SPY": 500.0, "QQQ": 400.0, "AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0, "AMZN": 180.0,
          "GOOGL": 150.0, "META": 500.0, "JPM": 200.0, "XOM": 110.0}


@pytest.fixture
def rm(cfg):
    return RiskManager(cfg)


@pytest.fixture
def snap():
    return snapshot(PRICES, atr_1h=1.0, atr_1d=4.0)


def one(rm, decisions, portfolio, snap, **kw):
    return rm.review(decisions, portfolio, snap, NOW, **kw)


def test_buy_within_limits_is_approved_with_stop(rm, snap, portfolio_factory):
    v = one(rm, [dec("AAPL", "BUY", 5)], portfolio_factory(), snap)[0]
    assert v.status == "approved"
    it = v.intent
    limit = round(200.01 * 1.001, 2)
    assert it.side == "buy" and it.limit_price == limit
    assert it.qty == int(500 // limit)                      # 5% z 10k = 500 USD, całe akcje
    assert it.stop_price == pytest.approx(round(limit - 2 * 4.0, 2))   # 2 x ATR(1D)


def test_buy_capped_at_10_pct_of_equity(rm, snap, portfolio_factory):
    v = one(rm, [dec("NVDA", "BUY", 40)], portfolio_factory(), snap)[0]
    assert v.status == "resized"
    assert v.intent.notional <= 10_000 * 0.10 + 1e-6
    assert "10% equity" in v.reason


def test_buy_add_to_position_respects_cap(rm, snap, portfolio_factory):
    p = portfolio_factory(cash=9_200, positions={"NVDA": pos("NVDA", 8, 100.0)})   # 800 USD = 8%
    v = one(rm, [dec("NVDA", "BUY", 10)], p, snap)[0]
    assert v.status == "resized"
    assert v.intent.notional <= 200 + 1e-6


def test_buy_rejected_when_position_already_at_cap(rm, snap, portfolio_factory):
    p = portfolio_factory(cash=9_000, positions={"NVDA": pos("NVDA", 10, 100.0)})   # 1000 USD = 10%
    v = one(rm, [dec("NVDA", "BUY", 5)], p, snap)[0]
    assert v.status == "rejected"


def test_max_five_open_positions(rm, snap, portfolio_factory):
    held = {s: pos(s, 1, PRICES[s]) for s in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]}
    p = portfolio_factory(cash=8_000, positions=held)
    vs = one(rm, [dec("XOM", "BUY", 5), dec("AAPL", "BUY", 3)], p, snap)
    assert vs[0].status == "rejected" and "limit otwartych pozycji" in vs[0].reason
    assert vs[1].status == "approved"                    # dokupienie istniejącej pozycji jest OK


def test_max_positions_counts_buys_in_same_run(rm, snap, portfolio_factory):
    held = {s: pos(s, 1, PRICES[s]) for s in ["SPY", "QQQ", "AAPL"]}
    p = portfolio_factory(cash=9_000, positions=held)
    vs = one(rm, [dec("MSFT", "BUY", 5), dec("NVDA", "BUY", 5), dec("XOM", "BUY", 5)], p, snap)
    assert [v.status for v in vs] == ["approved", "approved", "rejected"]


def test_no_leverage_buy_limited_to_cash(rm, snap, portfolio_factory):
    p = portfolio_factory(equity=10_000, cash=300)
    v = one(rm, [dec("NVDA", "BUY", 10)], p, snap)[0]
    assert v.status == "resized"
    assert v.intent.notional <= 300 * 0.99 + 1e-6
    assert "bez dźwigni" in v.reason


def test_no_cash_rejected(rm, snap, portfolio_factory):
    v = one(rm, [dec("NVDA", "BUY", 5)], portfolio_factory(cash=0), snap)[0]
    assert v.status == "rejected"


def test_cash_is_shared_between_buys_in_one_run(rm, snap, portfolio_factory):
    p = portfolio_factory(equity=10_000, cash=1_000)
    vs = one(rm, [dec("NVDA", "BUY", 8), dec("XOM", "BUY", 8)], p, snap)
    total = sum(v.intent.notional for v in vs if v.intent)
    assert total <= 1_000 * 0.99 + 1e-6


def test_no_short_selling(rm, snap, portfolio_factory):
    v = one(rm, [dec("AAPL", "SELL", 100)], portfolio_factory(), snap)[0]
    assert v.status == "rejected" and "krótka sprzedaż" in v.reason


def test_sell_never_exceeds_position(rm, snap, portfolio_factory):
    p = portfolio_factory(positions={"AAPL": pos("AAPL", 7, 200.0)})
    full = one(rm, [dec("AAPL", "SELL", 100)], p, snap)[0]
    half = one(rm, [dec("AAPL", "SELL", 50)], p, snap)[0]
    assert full.intent.qty == 7 and half.intent.qty == 3
    assert full.intent.limit_price == round(199.99 * 0.999, 2)


def test_sell_rounding_to_zero_rejected(rm, snap, portfolio_factory):
    p = portfolio_factory(positions={"AAPL": pos("AAPL", 1, 200.0)})
    v = one(rm, [dec("AAPL", "SELL", 30)], p, snap)[0]
    assert v.status == "rejected"


def test_only_watchlist(rm, portfolio_factory):
    s = snapshot({**PRICES, "TSLA": 250.0})
    v = one(rm, [dec("TSLA", "BUY", 5)], portfolio_factory(), s)[0]
    assert v.status == "rejected" and "watchlisty" in v.reason


def test_options_and_crypto_rejected(cfg, snap, portfolio_factory):
    rm = RiskManager(cfg, asset_lookup=lambda s: AssetInfo(s, "us_option", True, "active"))
    v = rm.review([dec("AAPL", "BUY", 5)], portfolio_factory(), snap, NOW)[0]
    assert v.status == "rejected" and "us_option" in v.reason


def test_non_tradable_asset_rejected(cfg, snap, portfolio_factory):
    rm = RiskManager(cfg, asset_lookup=lambda s: AssetInfo(s, "us_equity", False, "inactive"))
    assert rm.review([dec("AAPL", "BUY", 5)], portfolio_factory(), snap, NOW)[0].status == "rejected"


def test_cooldown_after_trade(rm, snap, portfolio_factory):
    p = portfolio_factory(last_trade_at={"AAPL": NOW - timedelta(minutes=60)})
    v = one(rm, [dec("AAPL", "BUY", 5)], p, snap)[0]
    assert v.status == "rejected" and "cooldown" in v.reason
    p2 = portfolio_factory(last_trade_at={"AAPL": NOW - timedelta(minutes=121)})
    assert one(rm, [dec("AAPL", "BUY", 5)], p2, snap)[0].status == "approved"


def test_cooldown_applies_to_sells_too(rm, snap, portfolio_factory):
    p = portfolio_factory(positions={"AAPL": pos("AAPL", 5, 200.0)},
                          last_trade_at={"AAPL": NOW - timedelta(minutes=30)})
    assert one(rm, [dec("AAPL", "SELL", 100)], p, snap)[0].status == "rejected"


def test_halted_rejects_everything(rm, snap, portfolio_factory):
    p = portfolio_factory(positions={"AAPL": pos("AAPL", 5, 200.0)})
    vs = one(rm, [dec("AAPL", "SELL", 100), dec("XOM", "BUY", 5), dec("MSFT", "HOLD")], p, snap,
             halted=True, halt_reason="-3.5%")
    assert [v.status for v in vs] == ["rejected", "rejected", "hold"]


@pytest.mark.parametrize("equity, last, breached", [
    (9_700, 10_000, True),       # dokładnie -3%
    (9_650, 10_000, True),
    (9_701, 10_000, False),
    (10_500, 10_000, False),
])
def test_daily_loss_limit(rm, equity, last, breached):
    acc = AccountState("id", "PA1", "active", equity, last, equity, equity, False)
    assert rm.daily_loss_breached(acc)[0] is breached


def test_pending_order_blocks_new_trade(rm, snap, portfolio_factory):
    o = OrderState("o1", "c", "AAPL", "buy", "limit", 1, 0, None, "new", 200.0)
    p = portfolio_factory(open_orders=[o])
    assert one(rm, [dec("AAPL", "BUY", 5)], p, snap)[0].status == "rejected"


def test_existing_stop_order_does_not_block(rm, snap, portfolio_factory):
    o = OrderState("o1", "c", "AAPL", "sell", "stop", 5, 0, None, "new", None, 190.0)
    p = portfolio_factory(positions={"AAPL": pos("AAPL", 5, 200.0)}, open_orders=[o])
    assert one(rm, [dec("AAPL", "SELL", 100)], p, snap)[0].status == "approved"


def test_buy_without_atr_rejected_no_stop_possible(rm, portfolio_factory):
    s = snapshot(PRICES, atr_1d=None)
    v = one(rm, [dec("AAPL", "BUY", 5)], portfolio_factory(), s)[0]
    assert v.status == "rejected" and "stop-loss" in v.reason


def test_hourly_atr_stop_option(cfg, snap, portfolio_factory):
    cfg.risk.stop_loss_atr_timeframe = "1h"
    v = RiskManager(cfg).review([dec("AAPL", "BUY", 5)], portfolio_factory(), snap, NOW)[0]
    assert v.intent.stop_price == pytest.approx(round(v.intent.limit_price - 2 * 1.0, 2))


def test_qty_below_one_share_rejected(rm, portfolio_factory):
    s = snapshot({**PRICES, "META": 5_000.0})
    v = one(rm, [dec("META", "BUY", 5)], portfolio_factory(), s)[0]
    assert v.status == "rejected" and "1 akcji" in v.reason


def test_duplicate_ticker_in_one_run(rm, snap, portfolio_factory):
    vs = one(rm, [dec("AAPL", "BUY", 3), dec("AAPL", "BUY", 3)], portfolio_factory(), snap)
    assert [v.status for v in vs] == ["approved", "rejected"]


def test_trading_blocked_account(rm, snap, portfolio_factory):
    p = portfolio_factory()
    p.account.trading_blocked = True
    assert one(rm, [dec("AAPL", "BUY", 5)], p, snap)[0].status == "rejected"
