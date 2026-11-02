"""Wrażliwość parametrów SMA i walk-forward (tylko agent SMA — bez LLM).

1. Dla każdej pary (fast, slow) pełny backtest portfelowy na tym samym okresie.
2. Okres dzielony na `folds` kolejnych części; dla każdej części: zwrot i Sharpe każdej pary.
3. Walk-forward: w części k gramy parą, która miała najlepszy Sharpe w części k-1
   (wybór wyłącznie na danych z przeszłości) — porównanie ze stałym 20/50 i z SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..evaluation.metrics import max_drawdown_pct, sharpe_ratio
from .engine import BacktestEngine, BacktestSpec
from .history import HistoricalStore


@dataclass
class GridResult:
    curves: dict[str, pd.Series]          # "10/30" -> equity (koniec sesji), "SPY" -> benchmark
    folds: list[tuple]                    # [(start_date, end_date)]
    table: pd.DataFrame                   # wiersze: (fold, para) -> return/sharpe
    walk_forward: pd.DataFrame            # fold, wybrana para, zwrot WF, zwrot 20/50, zwrot SPY
    default_pair: str


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for part in text.split(","):
        f, s = part.strip().split("/")
        pairs.append((int(f), int(s)))
    return pairs


def _eod(eq: pd.Series) -> pd.Series:
    """Equity tylko z końców sesji (bez punktu startowego), indeks = data."""
    s = eq.iloc[1:]
    return pd.Series(s.to_numpy(), index=[ts.date() for ts in s.index])


def run_grid(cfg: AppConfig, store: HistoricalStore, start, end, pairs: list[tuple[int, int]], folds: int,
             progress=print) -> GridResult:
    curves: dict[str, pd.Series] = {}
    spy = None
    for fast, slow in pairs:
        c = cfg.model_copy(deep=True)
        c.sma.fast, c.sma.slow = fast, slow
        key = f"{fast}/{slow}"
        progress(f"SMA {key}: backtest {start} – {end}")
        res = BacktestEngine(c, store, BacktestSpec(f"grid-{key.replace('/', '-')}", start, end, ["sma"],
                                                    fresh=True), db_path=":memory:",
                             progress=lambda m: None).run()
        curves[key] = _eod(res.equity["sma"])
        spy = _eod(res.equity["spy"])
    curves["SPY"] = spy

    days = list(spy.index)
    chunks = [list(a) for a in np.array_split(np.array(days, dtype=object), folds) if len(a)]
    bounds = [(ch[0], ch[-1]) for ch in chunks]

    rows = []
    for i, (a, b) in enumerate(bounds):
        for key, eq in curves.items():
            prev = eq[eq.index < a]
            base = prev.iloc[-1] if len(prev) else cfg.backtest.start_equity
            part = eq[(eq.index >= a) & (eq.index <= b)]
            full = pd.concat([pd.Series([base]), part.reset_index(drop=True)])
            rows.append({"fold": i + 1, "od": a, "do": b, "para": key,
                         "zwrot_%": (part.iloc[-1] / base - 1) * 100 if len(part) else None,
                         "sharpe": sharpe_ratio(full), "max_dd_%": max_drawdown_pct(full)})
    table = pd.DataFrame(rows)

    default = f"{cfg.sma.fast}/{cfg.sma.slow}"
    wf_rows = []
    pair_keys = [k for k in curves if k != "SPY"]
    for i in range(1, len(bounds)):
        prev = table[(table["fold"] == i) & (table["para"].isin(pair_keys))]
        best = prev.sort_values("sharpe", ascending=False, na_position="last").iloc[0]["para"]
        cur = table[table["fold"] == i + 1].set_index("para")
        wf_rows.append({"fold": i + 1, "wybrana_para": best,
                        "zwrot_WF_%": cur.loc[best, "zwrot_%"],
                        f"zwrot_{default}_%": cur.loc[default, "zwrot_%"] if default in cur.index else None,
                        "zwrot_SPY_%": cur.loc["SPY", "zwrot_%"]})
    return GridResult(curves, bounds, table, pd.DataFrame(wf_rows), default)


def write_grid_report(res: GridResult, out_dir: Path, start, end) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Wrażliwość SMA i walk-forward: {start} – {end}", "",
             "## Każda para na każdym odcinku", "",
             "```", res.table.round(2).to_string(index=False), "```", ""]
    if not res.walk_forward.empty:
        wf = res.walk_forward
        tot = lambda col: (np.prod([1 + x / 100 for x in wf[col].dropna()]) - 1) * 100  # noqa: E731
        lines += ["## Walk-forward (para wybrana na poprzednim odcinku)", "",
                  "```", wf.round(2).to_string(index=False), "```", "",
                  f"Łącznie (odcinki 2..n): WF {tot('zwrot_WF_%'):+.2f}%, "
                  f"stałe {res.default_pair} {tot(f'zwrot_{res.default_pair}_%'):+.2f}%, "
                  f"SPY {tot('zwrot_SPY_%'):+.2f}%", ""]
    lines += ["Jeśli wyniki mocno zależą od pary albo walk-forward wypada gorzej niż stała para, "
              "„najlepsze” parametry są dopasowaniem do przeszłości, a nie przewagą."]
    pd.concat(res.curves, axis=1).to_csv(out_dir / "grid_equity.csv")
    res.table.to_csv(out_dir / "grid_folds.csv", index=False)
    path = out_dir / "grid_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
