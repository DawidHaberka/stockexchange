"""Raport backtestu: metryki, istotność statystyczna, wywołania LLM, wykres i pliki CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..evaluation import stats
from ..evaluation.metrics import fifo_closed_trades, summarize, win_rate_pct
from ..storage.db import Database
from .engine import BacktestResult

LABELS = {"llm": "LLM", "sma": "SMA", "spy": "SPY buy&hold"}


def _fmt(v, fmt):
    return "—" if v is None else fmt.format(v)


def agent_metrics(res: BacktestResult, db: Database) -> dict:
    rf = res.cfg.report.risk_free_rate_annual
    out = {}
    for name, eq in res.equity.items():
        m = summarize(eq, rf)
        if name != "spy":
            trades = [dict(r) for r in db.trades(name)]
            closed = fifo_closed_trades(trades)
            m.update(n_trades=len(trades), n_closed=len(closed), win_rate_pct=win_rate_pct(closed),
                     stop_losses=sum(1 for t in trades if t["purpose"] == "stop_loss"))
        out[name] = m
    return out


def llm_stats(db: Database) -> dict:
    c = db.query_one("""SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, AVG(latency_ms) AS lat,
                               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
                        FROM llm_calls""")
    d = db.query("""SELECT source, action, risk_status, COUNT(*) AS n FROM decisions WHERE agent='llm'
                    GROUP BY source, action, risk_status""")
    return {"calls": int(c["n"]), "cost_equiv_usd": float(c["cost"]), "avg_latency_ms": c["lat"],
            "errors": int(c["errors"] or 0), "breakdown": [dict(r) for r in d]}


def plot(res: BacktestResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {"llm": dict(color="#2563eb", lw=2), "sma": dict(color="#16a34a", lw=1.6),
              "spy": dict(color="#6b7280", lw=1.6, ls="--")}
    for name, eq in res.equity.items():
        ax.plot(eq.index, eq.values, label=LABELS[name], **styles[name])
    ax.set_title(f"Backtest {res.spec.name}: {res.spec.start} – {res.spec.end}")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(res: BacktestResult, db: Database) -> Path:
    out = res.out_dir
    m = agent_metrics(res, db)
    cols = list(res.equity)
    lines = [f"# Backtest: {res.spec.name}", "",
             f"Okres **{res.spec.start} – {res.spec.end}** · sesji {res.days} · slotów {res.slots} · agenci: "
             f"{', '.join(res.spec.agents)} · kapitał startowy {res.cfg.backtest.start_equity:,.0f} USD · "
             f"spread {res.cfg.backtest.spread_bps:g} bp · korekta świec `{res.cfg.backtest.adjustment}`",
             f"Wariant: model `{res.cfg.llm.model}`, effort `{res.cfg.llm.effort}`, częstotliwość 1h, "
             f"prompt `{res.cfg.llm.prompt_hash}`, SMA {res.cfg.sma.fast}/{res.cfg.sma.slow}", ""]
    if not res.completed:
        lines += ["> **Backtest niedokończony** — wznowiony przebieg dopisze kolejne sesje.", ""]
    lines += ["| Metryka | " + " | ".join(LABELS[c] for c in cols) + " |", "|---|" + "---:|" * len(cols)]

    def row(label, key, fmt):
        lines.append(f"| {label} | " + " | ".join(_fmt(m[c].get(key), fmt) for c in cols) + " |")

    row("Kapitał końcowy (USD)", "end_value", "{:,.2f}")
    row("Total return", "total_return_pct", "{:+.2f}%")
    row("Sharpe (ann.)", "sharpe", "{:.2f}")
    row("Max drawdown", "max_drawdown_pct", "{:.2f}%")
    row("Transakcje (realizacje)", "n_trades", "{:d}")
    row("w tym stop-lossy", "stop_losses", "{:d}")
    row("Win rate (FIFO)", "win_rate_pct", "{:.1f}%")

    pairs = []
    if "llm" in res.equity:
        pairs.append(("LLM vs SPY", stats.compare(res.equity["llm"], res.equity["spy"])))
        if "sma" in res.equity:
            pairs.append(("LLM vs SMA", stats.compare(res.equity["llm"], res.equity["sma"])))
    if "sma" in res.equity:
        pairs.append(("SMA vs SPY", stats.compare(res.equity["sma"], res.equity["spy"])))
    lines += ["", "## Istotność statystyczna", "", *stats.format_table(pairs), ""]

    if "llm" in res.equity:
        s = llm_stats(db)
        lines += ["## Wywołania Claude", "",
                  f"Wywołań: **{s['calls']}**, błędów: {s['errors']}, średni czas: "
                  f"{_fmt(s['avg_latency_ms'], '{:,.0f}')} ms, równoważnik cen API: {s['cost_equiv_usd']:.2f} USD "
                  "(na subskrypcji niepobierany).", "",
                  "| Źródło | Akcja | Risk manager | Liczba |", "|---|---|---|---:|"]
        lines += [f"| {r['source']} | {r['action']} | {r['risk_status']} | {r['n']} |" for r in s["breakdown"]]
        lines.append("")

    plot(res, out / "equity.png")
    lines += ["![Equity](equity.png)", "", "## Metodologia", "",
              "- Dane „z danego momentu”: w każdym slocie agent widzi wyłącznie zamknięte świece 1H, świece 1D "
              "z poprzednich sesji i newsy opublikowane do tej chwili.",
              "- Ten sam kod co na żywo: agenci, triggery, risk manager (limity, cooldown, dzienny limit straty) "
              "i egzekucja; broker jest symulowany.",
              "- Realizacja: zlecenia po cenie ask/bid z chwili decyzji (spread jak wyżej); stop-lossy na kolejnych "
              "świecach 1H, przy luce cenowej po cenie otwarcia świecy.",
              "- Replay LLM wyłącznie po granicy wiedzy modelu (+ bufor) — bez look-ahead bias. Nagłówki newsów "
              "są w wersji końcowej (ewentualne późniejsze poprawki redakcyjne mogą być widoczne).",
              "- Istotność: test różnicy dziennych stóp zwrotu (t, bootstrap blokowy). Kilka miesięcy danych "
              "wykryje tylko bardzo dużą przewagę."]
    md = "\n".join(lines) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")

    # CSV do dalszej analizy
    pd.concat({LABELS[k]: v for k, v in res.equity.items()}, axis=1).to_csv(out / "equity.csv")
    pd.read_sql_query("SELECT * FROM trades ORDER BY filled_at, id", db.conn).to_csv(out / "trades.csv", index=False)
    pd.read_sql_query(
        """SELECT d.created_at, d.agent, d.ticker, d.action, d.size_pct, d.confidence, d.source, d.triggers,
                  d.risk_status, d.risk_reason, d.order_qty, d.limit_price, d.stop_price, d.reasoning
           FROM decisions d WHERE d.action != 'HOLD' OR d.source NOT IN ('no_trigger', 'rule')
           ORDER BY d.id""", db.conn).to_csv(out / "decisions.csv", index=False)
    return out / "report.md"
