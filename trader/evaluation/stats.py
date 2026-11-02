"""Istotność statystyczna: czy różnica wyników dwóch strategii to przewaga, czy szum?

Porównujemy dzienne stopy zwrotu A i B (te same dni). Różnica d_t = r_A,t - r_B,t:
- średnia różnica, roczna nadwyżka (x252), tracking error (odch. std. x √252), information ratio,
- statystyka t = mean / (sd / √n) i p-value (przybliżenie normalne),
- bootstrap blokowy (bloki po kilka dni, odporny na autokorelację) — 95% CI rocznej nadwyżki i p-value,
- ile dni danych trzeba, żeby przy obecnej średniej i zmienności różnicy t osiągnęło 2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..broker import NY

TRADING_DAYS = 252


def daily_series(equity: pd.Series) -> pd.Series:
    """Equity -> jedna wartość na dzień sesji (ostatnia), indeks = data NY."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(equity.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY)
    s = pd.Series(equity.to_numpy(dtype=float), index=idx.date)
    return s[~s.index.duplicated(keep="last")].sort_index()


def paired_returns(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    da, db = daily_series(a), daily_series(b)
    df = pd.concat([da.rename("a"), db.rename("b")], axis=1, join="inner").sort_index()
    return df.pct_change().dropna()


def _p_normal(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def block_bootstrap_means(x: np.ndarray, n_boot: int = 5000, block: int = 5, seed: int = 0) -> np.ndarray:
    """Circular block bootstrap średniej."""
    n = len(x)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = x[idx.reshape(n_boot, -1)[:, :n]]
    return samples.mean(axis=1)


def compare(a: pd.Series, b: pd.Series, n_boot: int = 5000, block: int = 5, seed: int = 0,
            min_days: int = 5) -> dict:
    """Porównanie equity A vs B. Zwraca słownik metryk (lub {'n_days': n, 'enough': False})."""
    r = paired_returns(a, b)
    n = len(r)
    if n < min_days:
        return {"n_days": n, "enough": False}
    d = (r["a"] - r["b"]).to_numpy()
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    out = {
        "n_days": n, "enough": True,
        "mean_diff_daily_pct": mean * 100,
        "excess_annual_pct": mean * TRADING_DAYS * 100,
        "tracking_error_annual_pct": sd * math.sqrt(TRADING_DAYS) * 100,
    }
    if sd == 0:
        out.update(t_stat=None, p_value=None, info_ratio=None, ci_low_annual_pct=None, ci_high_annual_pct=None,
                   p_bootstrap=None, days_needed_t2=None)
        return out
    t = mean / (sd / math.sqrt(n))
    boots = block_bootstrap_means(d, n_boot, block, seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    out.update(
        t_stat=t, p_value=_p_normal(t), info_ratio=mean / sd * math.sqrt(TRADING_DAYS),
        ci_low_annual_pct=float(lo) * TRADING_DAYS * 100, ci_high_annual_pct=float(hi) * TRADING_DAYS * 100,
        p_bootstrap=p_boot,
        days_needed_t2=math.ceil((2 * sd / abs(mean)) ** 2) if mean != 0 else None,
    )
    return out


def verdict(c: dict) -> str:
    if not c.get("enough"):
        return f"za mało wspólnych dni ({c.get('n_days', 0)})"
    p = c.get("p_bootstrap")
    if p is None:
        return "brak zmienności różnicy"
    if p < 0.05:
        return "różnica istotna (p < 0,05)"
    need = c.get("days_needed_t2")
    extra = f"; przy obecnej przewadze potrzeba ~{need} dni" if need else ""
    return f"nieistotna — nie da się odróżnić od szumu{extra}"


def format_table(rows: list[tuple[str, dict]]) -> list[str]:
    """Tabela markdown: [(etykieta, wynik compare)]."""
    out = ["| Porównanie | Dni | Roczna nadwyżka | 95% CI (bootstrap) | t | p | Info ratio | Ocena |",
           "|---|---:|---:|---|---:|---:|---:|---|"]

    def f(v, fmt):
        return "—" if v is None else fmt.format(v)

    for label, c in rows:
        if not c.get("enough"):
            out.append(f"| {label} | {c.get('n_days', 0)} | — | — | — | — | — | {verdict(c)} |")
            continue
        ci = (f"[{c['ci_low_annual_pct']:+.1f}%, {c['ci_high_annual_pct']:+.1f}%]"
              if c.get("ci_low_annual_pct") is not None else "—")
        out.append(f"| {label} | {c['n_days']} | {c['excess_annual_pct']:+.1f}% | {ci} | {f(c.get('t_stat'), '{:.2f}')} "
                   f"| {f(c.get('p_bootstrap'), '{:.3f}')} | {f(c.get('info_ratio'), '{:.2f}')} | {verdict(c)} |")
    return out
