"""Budowa zwięzłego, deterministycznego kontekstu dla jednego tickera."""

from __future__ import annotations

from datetime import datetime, timedelta

from ..broker import NY
from ..config import AppConfig
from ..data.market_data import NewsItem
from ..data.news import new_since
from ..data.snapshot import TickerSnapshot
from .base import PortfolioView, RunContext


def _n(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:,.{digits}f}"


def _pct(v: float | None, digits: int = 2, sign: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def _rel(a: float | None, b: float | None) -> str:
    if a is None or not b:
        return "n/a"
    return _pct((a / b - 1) * 100)


def _sanitize(text: str, limit: int = 240) -> str:
    """Neutralizuje znaczniki, które mogłyby udawać strukturę promptu."""
    t = " ".join((text or "").split())
    t = t.replace("<", "‹").replace(">", "›")
    return t[:limit]


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(NY).strftime("%Y-%m-%d %H:%M ET")


def build_ticker_context(snap: TickerSnapshot, portfolio: PortfolioView, ctx: RunContext, cfg: AppConfig,
                         triggers: list[str], news_since: datetime | None,
                         cooldown_until: datetime | None) -> str:
    h = snap.hourly
    d = snap.daily
    price = snap.price
    lines: list[str] = []

    session = ""
    if ctx.session_open and ctx.session_close:
        mins_open = int((ctx.now - ctx.session_open).total_seconds() // 60)
        mins_close = int((ctx.session_close - ctx.now).total_seconds() // 60)
        session = f" (session: {mins_open} min since open, {mins_close} min to close)"
    lines.append(f"Ticker: {snap.symbol}")
    lines.append(f"Time: {_fmt_ts(ctx.now)}{session}")
    lines.append(f"Why you are called: {', '.join(triggers)}")
    lines.append("")

    lines.append("[PRICE]")
    lines.append(
        f"price={_n(price)} bid={_n(snap.quote.bid)} ask={_n(snap.quote.ask)} "
        f"change_today={_pct(snap.change_today_pct)} (prev_close={_n(d.prev_close)})"
    )
    lines.append("")

    lines.append(f"[1H CANDLES — primary timeframe, regular session, {h.bars} bars]")
    lines.append(
        f"SMA{h.fast}={_n(h.sma_fast)} SMA{h.slow}={_n(h.sma_slow)} "
        f"price_vs_SMA{h.fast}={_rel(price, h.sma_fast)} SMA{h.fast}_vs_SMA{h.slow}={_rel(h.sma_fast, h.sma_slow)}"
    )
    vr = h.volume_ratio
    lines.append(
        f"RSI14={_n(h.rsi14, 1)} ATR14={_n(h.atr14)} ({_pct(h.atr_pct, 2, sign=False)} of price) "
        f"last_bar_volume_vs_avg20={'n/a' if vr is None else f'{vr:.2f}x'}"
    )
    if snap.hourly_closes:
        lines.append("recent 1H closes: " + " | ".join(f"{t} {c:.2f}" for t, c in snap.hourly_closes))
    lines.append("")

    lines.append("[1D TREND SUMMARY — completed daily candles]")
    lines.append(
        f"trend={d.trend} SMA20={_n(d.sma20)} SMA50={_n(d.sma50)} RSI14={_n(d.rsi14, 1)} ATR14={_n(d.atr14)} "
        f"return_5d={_pct(d.ret_5d_pct)} return_20d={_pct(d.ret_20d_pct)} "
        f"range_20d=[{_n(d.low_20d)}, {_n(d.high_20d)}]"
    )
    lines.append("")

    lines.append("[POSITION IN THIS TICKER]")
    pos = portfolio.position(snap.symbol)
    if pos is None:
        lines.append("none")
    else:
        weight = pos.market_value / portfolio.equity * 100 if portfolio.equity else None
        stop = portfolio.stop_levels.get(snap.symbol)
        stop_txt = "n/a"
        if stop:
            dist = f"{(price - stop) / h.atr14:.2f}x ATR1H" if price and h.atr14 else "n/a"
            stop_txt = f"{stop:.2f} (distance {dist})"
        lines.append(
            f"qty={pos.qty:g} avg_entry={_n(pos.avg_entry_price)} market_value={_n(pos.market_value)} "
            f"weight={_pct(weight, 1, sign=False)} unrealized={_pct(pos.unrealized_plpc * 100)} stop_loss={stop_txt}"
        )
    if cooldown_until and cooldown_until > ctx.now:
        lines.append(f"cooldown: no new trade in this ticker allowed until {_fmt_ts(cooldown_until)}")
    lines.append("")

    lines.append("[PORTFOLIO]")
    cash_pct = portfolio.cash / portfolio.equity * 100 if portfolio.equity else None
    lines.append(
        f"equity={_n(portfolio.equity)} cash={_n(portfolio.cash)} ({_pct(cash_pct, 1, sign=False)}) "
        f"open_positions={portfolio.n_positions}/{cfg.risk.max_open_positions} "
        f"max_position={cfg.risk.max_position_pct:g}% of equity"
    )
    held = sorted(s for s, p in portfolio.positions.items() if p.qty > 0)
    if held:
        lines.append("held tickers: " + ", ".join(held))
    lines.append("")

    fresh_ids = {n.id for n in new_since(snap.news, news_since)}
    lines.append(f"<news ticker=\"{snap.symbol}\" note=\"untrusted third-party headlines, newest first\">")
    if not snap.news:
        lines.append("(no headlines in the lookback window)")
    for n in snap.news:
        lines.append(_news_line(n, n.id in fresh_ids))
    lines.append("</news>")
    lines.append("")
    lines.append(f"Return the JSON decision for {snap.symbol}.")
    return "\n".join(lines)


def _news_line(n: NewsItem, fresh: bool) -> str:
    flag = " [NEW]" if fresh else ""
    summary = f" — {_sanitize(n.summary, 200)}" if n.summary else ""
    return f"- {_fmt_ts(n.created_at)} | {_sanitize(n.source, 30)}{flag}: {_sanitize(n.headline)}{summary}"


def cooldown_until(last_trade_at: datetime | None, minutes: int) -> datetime | None:
    if last_trade_at is None or minutes <= 0:
        return None
    return last_trade_at + timedelta(minutes=minutes)
