"""Raport dzienny: equity curve LLM vs SMA vs buy-and-hold SPY od tego samego startu.

Wynik netto = zysk - koszty API (koszt API liczony tylko dla przebiegów live od startu).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..broker import NY, Broker
from ..config import AppConfig
from ..data.market_data import MarketDataSource
from ..storage.db import Database, parse_iso
from .metrics import fifo_closed_trades, summarize, win_rate_pct

log = logging.getLogger(__name__)

AGENT_LABELS = {"llm": "LLM", "sma": "SMA 20/50", "spy": "SPY buy&hold"}


# =========================================================================== zbieranie danych EOD
def collect_eod(cfg: AppConfig, db: Database, brokers: dict[str, Broker], md: MarketDataSource | None,
                now: datetime) -> str:
    """Zapisuje stan kont i zamknięcia benchmarku. Zwraca datę sesji (YYYY-MM-DD)."""
    ref = next(iter(brokers.values()))
    clock = ref.get_clock()
    today = now.astimezone(NY).date()
    session = ref.get_session(today)
    after_close = session is not None and not clock.is_open and now.astimezone(NY) >= session.close
    kind = "eod" if after_close else "report"
    session_date = today.isoformat()

    for name, broker in brokers.items():
        acc = broker.get_account()
        pos = broker.get_positions()
        db.insert_snapshot(name, None, session_date, kind, acc, pos)
        log.info("[%s] stan konta (%s): equity %.2f, cash %.2f, pozycje %d", name, kind, acc.equity, acc.cash,
                 len(pos))

    if md is not None:
        start = now - timedelta(days=30)
        row = db.query_one("SELECT MIN(session_date) AS d FROM experiment_start")
        if row and row["d"]:
            start = min(start, datetime.fromisoformat(row["d"]).replace(tzinfo=NY) - timedelta(days=5))
        bars = md.daily_bars([cfg.benchmark_symbol], start, now).get(cfg.benchmark_symbol)
        if bars is not None and not bars.empty:
            idx = bars.index if bars.index.tz is not None else bars.index.tz_localize("UTC")
            for ts, close in zip(idx.tz_convert(NY), bars["close"], strict=True):
                d = ts.date()
                if d < today or after_close:
                    db.upsert_benchmark_close(cfg.benchmark_symbol, d.isoformat(), float(close))
    return session_date


# =========================================================================== serie equity
@dataclass
class Series:
    agent: str
    equity: pd.Series
    start_equity: float | None
    provisional: bool


def agent_equity_series(db: Database, agent: str) -> Series:
    start = db.get_experiment_start(agent)
    if start is not None:
        rows = db.query("SELECT * FROM portfolio_snapshots WHERE agent = ? AND taken_at >= ? ORDER BY taken_at, id",
                        (agent, start["started_at"]))
    else:
        rows = db.query("SELECT * FROM portfolio_snapshots WHERE agent = ? ORDER BY taken_at, id", (agent,))
    if not rows and start is None:
        return Series(agent, pd.Series(dtype=float), None, True)

    # jeden punkt na sesję: ostatni stan "eod", a gdy go brak — ostatni stan z danego dnia
    by_day: dict[str, tuple[datetime, float, bool]] = {}
    for r in rows:
        is_eod = r["kind"] == "eod"
        prev = by_day.get(r["session_date"])
        if prev is None or is_eod or not prev[2]:
            by_day[r["session_date"]] = (parse_iso(r["taken_at"]), float(r["equity"]), is_eod)

    points: list[tuple[pd.Timestamp, float]] = []
    if start is not None:
        points.append((pd.Timestamp(parse_iso(start["started_at"])), float(start["start_equity"])))
    for d in sorted(by_day):
        t, eq, _ = by_day[d]
        points.append((pd.Timestamp(t), eq))
    s = pd.Series([p[1] for p in points], index=[p[0] for p in points], dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    start_eq = float(start["start_equity"]) if start is not None else (float(s.iloc[0]) if len(s) else None)
    return Series(agent, s, start_eq, start is None)


def benchmark_series(db: Database, cfg: AppConfig, ref: Series) -> Series:
    """SPY buy-and-hold kupiony za kapitał startowy agenta LLM w momencie startu eksperymentu."""
    if ref.start_equity is None or ref.equity.empty:
        return Series("spy", pd.Series(dtype=float), None, True)
    start = db.get_experiment_start(ref.agent)
    closes = {r["session_date"]: float(r["close"]) for r in
              db.query("SELECT session_date, close FROM benchmark_daily WHERE symbol = ? ORDER BY session_date",
                       (cfg.benchmark_symbol,))}
    if not closes:
        return Series("spy", pd.Series(dtype=float), ref.start_equity, True)

    start_ts = ref.equity.index[0]
    start_day = start_ts.tz_convert(NY).date().isoformat() if start_ts.tzinfo else str(start_ts.date())
    base_price = float(start["benchmark_price"]) if start is not None and start["benchmark_price"] else None
    if base_price is None:  # tryb tymczasowy (brak startu live): pierwsze dostępne zamknięcie
        cands = [d for d in sorted(closes) if d >= start_day] or sorted(closes)
        base_price = closes[cands[0]]

    values, index = [], []
    for ts in ref.equity.index:
        day = ts.tz_convert(NY).date().isoformat() if ts.tzinfo else str(ts.date())
        if ts == start_ts:
            values.append(ref.start_equity)
        else:
            px = closes.get(day)
            if px is None:
                prior = [d for d in closes if d <= day]
                if not prior:
                    continue
                px = closes[max(prior)]
            values.append(ref.start_equity * px / base_price)
        index.append(ts)
    return Series("spy", pd.Series(values, index=index, dtype=float), ref.start_equity, ref.provisional)


# =========================================================================== metryki
def api_costs(db: Database, since_iso: str | None) -> dict:
    q = ("SELECT r.mode AS mode, COUNT(*) AS n, COALESCE(SUM(c.cost_usd), 0) AS cost FROM llm_calls c "
         "JOIN runs r ON r.id = c.run_id {where} GROUP BY r.mode")
    where, params = ("WHERE c.created_at >= ?", (since_iso,)) if since_iso else ("", ())
    out = {"live": {"calls": 0, "cost_usd": 0.0}, "dry_run": {"calls": 0, "cost_usd": 0.0}}
    for r in db.query(q.format(where=where), params):
        out[r["mode"]] = {"calls": int(r["n"]), "cost_usd": float(r["cost"])}
    return out


def compute_metrics(cfg: AppConfig, db: Database) -> dict:
    rf = cfg.report.risk_free_rate_annual
    series = {a: agent_equity_series(db, a) for a in ("llm", "sma")}
    ref = series["llm"] if not series["llm"].equity.empty else series["sma"]
    series["spy"] = benchmark_series(db, cfg, ref)

    out: dict = {"agents": {}, "provisional": any(s.provisional for s in series.values())}
    for name, s in series.items():
        if s.equity.empty:
            out["agents"][name] = {"label": AGENT_LABELS[name], "available": False}
            continue
        m = summarize(s.equity, rf)
        m.update(label=AGENT_LABELS[name], available=True, provisional=s.provisional)
        start = db.get_experiment_start(name) if name != "spy" else None
        since = start["started_at"] if start is not None else None
        if name in ("llm", "sma"):
            trades = [dict(r) for r in db.trades(name) if since is None or r["filled_at"] >= since]
            closed = fifo_closed_trades(trades)
            m["n_trades"] = len(trades)
            m["n_closed_trades"] = len(closed)
            m["win_rate_pct"] = win_rate_pct(closed)
        else:
            m["n_trades"] = 0
            m["win_rate_pct"] = None
        if name == "llm":
            costs = api_costs(db, since)
            m["api_cost_usd"] = costs["live"]["cost_usd"]
            m["api_calls"] = costs["live"]["calls"]
            m["api_cost_dry_run_usd"] = costs["dry_run"]["cost_usd"]
        else:
            m["api_cost_usd"] = 0.0
        gross = (m["end_value"] - m["start_value"]) if m["start_value"] is not None else None
        m["gross_pnl_usd"] = gross
        m["net_pnl_usd"] = gross - m["api_cost_usd"] if gross is not None else None
        m["net_return_pct"] = (m["net_pnl_usd"] / m["start_value"] * 100) if gross is not None and m["start_value"] else None
        out["agents"][name] = m
    out["series"] = {n: [(ts.isoformat(), v) for ts, v in s.equity.items()] for n, s in series.items()}
    return out


# =========================================================================== render
def _fmt(v, fmt="{:,.2f}", na="—"):
    return na if v is None else fmt.format(v)


def _today_stats(db: Database, session_date: str) -> list[str]:
    lines = []
    for agent in ("llm", "sma"):
        rows = db.query(
            """SELECT d.action, d.source, d.risk_status FROM decisions d JOIN runs r ON r.id = d.run_id
               WHERE d.agent = ? AND r.session_date = ?""", (agent, session_date))
        if not rows:
            lines.append(f"- **{AGENT_LABELS[agent]}**: brak decyzji dziś")
            continue
        by_action = pd.Series([r["action"] for r in rows]).value_counts().to_dict()
        by_risk = pd.Series([r["risk_status"] for r in rows]).value_counts().to_dict()
        no_trig = sum(1 for r in rows if r["source"] == "no_trigger")
        lines.append(f"- **{AGENT_LABELS[agent]}**: decyzje {len(rows)} ({', '.join(f'{k} {v}' for k, v in by_action.items())})"
                     f"; risk: {', '.join(f'{k} {v}' for k, v in by_risk.items())}"
                     + (f"; bez wywołania API (no trigger): {no_trig}" if agent == "llm" else ""))
    c = db.query_one("""SELECT COUNT(*) AS n, COALESCE(SUM(c.cost_usd),0) AS cost,
                              COALESCE(SUM(c.input_tokens),0) AS i, COALESCE(SUM(c.output_tokens),0) AS o,
                              COALESCE(SUM(c.cache_read_tokens),0) AS cr
                       FROM llm_calls c JOIN runs r ON r.id = c.run_id WHERE r.session_date = ?""", (session_date,))
    lines.append(f"- **API Claude dziś**: {c['n']} wywołań, koszt {c['cost']:.4f} USD "
                 f"(in {c['i']:,} / out {c['o']:,} / cache read {c['cr']:,} tokenów)")
    o = db.query("""SELECT o.agent, o.status, COUNT(*) AS n FROM orders o JOIN runs r ON r.id = o.run_id
                    WHERE r.session_date = ? GROUP BY o.agent, o.status""", (session_date,))
    if o:
        lines.append("- **Zlecenia dziś**: " + "; ".join(f"{r['agent']}: {r['status']} x{r['n']}" for r in o))
    return lines


def render_markdown(cfg: AppConfig, db: Database, metrics: dict, session_date: str, png_name: str | None) -> str:
    A = metrics["agents"]
    cols = ["llm", "sma", "spy"]
    v = cfg.variant()
    start_rows = {r["agent"]: r for r in db.query("SELECT * FROM experiment_start")}
    lines = [f"# Raport dzienny — {session_date}", ""]
    lines.append(f"Eksperyment **{cfg.experiment.name}** | wariant: model `{v['model']}`, effort `{v['effort']}`, "
                 f"częstotliwość `{v['frequency']}`, prompt `{v['prompt_hash']}`, config `{v['config_hash']}`")
    if start_rows:
        lines.append("Start eksperymentu: " + "; ".join(
            f"{AGENT_LABELS[a]} {r['session_date']} z kapitałem {r['start_equity']:,.2f} USD"
            for a, r in sorted(start_rows.items())))
    if metrics.get("provisional"):
        lines += ["", "> **Wyniki tymczasowe** — eksperyment nie został jeszcze uruchomiony w trybie live "
                      "(brak zapisanego kapitału startowego). Seria liczona od pierwszego zapisanego stanu konta."]
    lines += ["", "| Metryka | " + " | ".join(AGENT_LABELS[c] for c in cols) + " |",
              "|---|" + "---:|" * len(cols)]

    def row(label, key, fmt="{:,.2f}"):
        vals = [_fmt(A[c].get(key), fmt) if A[c].get("available") else "—" for c in cols]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Kapitał startowy (USD)", "start_value")
    row("Kapitał bieżący (USD)", "end_value")
    row("Total return brutto", "total_return_pct", "{:+.2f}%")
    row("Koszt API (USD)", "api_cost_usd", "{:,.4f}")
    row("Wynik netto (USD)", "net_pnl_usd", "{:+,.2f}")
    row("Total return netto", "net_return_pct", "{:+.2f}%")
    row("Sharpe (ann.)", "sharpe", "{:.2f}")
    row("Max drawdown", "max_drawdown_pct", "{:.2f}%")
    row("Win rate", "win_rate_pct", "{:.1f}%")
    row("Liczba transakcji (realizacji)", "n_trades", "{:d}")
    row("Dni w serii", "days", "{:d}")
    if A["llm"].get("available"):
        lines.append("")
        lines.append(f"Wywołania API (live): {A['llm'].get('api_calls', 0)}; koszt wywołań w dry-run (poza wynikiem): "
                     f"{A['llm'].get('api_cost_dry_run_usd', 0):.4f} USD.")
    lines += ["", "## Dzisiaj", *_today_stats(db, session_date), ""]
    if png_name:
        lines += [f"![Equity curve]({png_name})", ""]
    lines += [
        "## Jak czytać",
        "- **Total return netto** dla LLM odejmuje koszt wszystkich wywołań Claude w przebiegach live — to jest "
        "właściwa miara, czy LLM „zarabia na siebie”.",
        "- **Sharpe** z dziennych zmian equity (annualizowany √252, rf z config). Przy kilku–kilkunastu dniach "
        "danych jest bardzo niestabilny — nie wyciągaj wniosków przed ~3 miesiącami.",
        "- **Max drawdown**: największy spadek od szczytu equity.",
        "- **Win rate**: odsetek zamkniętych sprzedaży (FIFO) z dodatnim wynikiem; bez liczby transakcji nic nie znaczy.",
        "- SPY buy&hold to hipotetyczny zakup SPY za kapitał startowy LLM w momencie startu eksperymentu.",
    ]
    return "\n".join(lines) + "\n"


def plot_equity(metrics: dict, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    styles = {"llm": dict(color="#2563eb", lw=2), "sma": dict(color="#16a34a", lw=1.6),
              "spy": dict(color="#6b7280", lw=1.6, ls="--")}
    for name, pts in metrics["series"].items():
        if not pts:
            continue
        idx = pd.to_datetime([p[0] for p in pts])
        ax.plot(idx, [p[1] for p in pts], label=AGENT_LABELS[name], marker="o", ms=3, **styles[name])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title("Equity: LLM vs SMA 20/50 vs SPY buy&hold")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


@dataclass
class ReportResult:
    session_date: str
    metrics: dict
    markdown: str
    md_path: Path
    png_path: Path | None


def build_report(cfg: AppConfig, db: Database, session_date: str) -> ReportResult:
    metrics = compute_metrics(cfg, db)
    out_dir = Path(cfg.storage.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"equity_{session_date}.png"
    has_png = plot_equity(metrics, png)
    md = render_markdown(cfg, db, metrics, session_date, png.name if has_png else None)
    md_path = out_dir / f"report_{session_date}.md"
    md_path.write_text(md, encoding="utf-8")
    db.save_report(session_date, {k: v for k, v in metrics.items() if k != "series"}, str(md_path))
    return ReportResult(session_date, metrics, md, md_path, png if has_png else None)


def metrics_json(metrics: dict) -> str:
    return json.dumps({k: v for k, v in metrics.items() if k != "series"}, ensure_ascii=False, indent=2, default=str)
