"""Dashboard Streamlit (tylko odczyt bazy): metryki, equity curve, log decyzji z uzasadnieniami.

Uruchom z katalogu repozytorium:
    streamlit run trader/dashboard/app.py
Na VM (dostęp przez tunel SSH):
    streamlit run trader/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from trader.config import load_config  # noqa: E402
from trader.evaluation.report import AGENT_LABELS, compute_metrics  # noqa: E402
from trader.storage.db import Database  # noqa: E402

st.set_page_config(page_title="LLM vs SMA vs SPY", layout="wide")

cfg_path = st.sidebar.text_input("Plik konfiguracji", str(ROOT / "config.yaml"))
cfg = load_config(cfg_path)
db_path = Path(cfg.storage.db_path)
if not db_path.is_absolute():
    db_path = ROOT / db_path
if not db_path.exists():
    st.warning(f"Brak bazy {db_path} — uruchom najpierw `python main.py --dry-run`.")
    st.stop()


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


db = get_db(str(db_path))


def df(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, db.conn, params=params)


if st.sidebar.button("Odśwież"):
    st.cache_data.clear()

st.title("Eksperyment: LLM vs SMA 20/50 vs SPY (paper)")
v = cfg.variant()
st.caption(f"Bieżący wariant: model `{v['model']}` · effort `{v['effort']}` · częstotliwość `{v['frequency']}` · "
           f"prompt `{v['prompt_hash']}` · config `{v['config_hash']}` · "
           f"zlecenia {'WŁĄCZONE' if cfg.execution.orders_enabled else 'WYŁĄCZONE (dry-run)'}")

# ------------------------------------------------------------------ metryki
metrics = compute_metrics(cfg, db)
if metrics.get("provisional"):
    st.info("Wyniki tymczasowe — brak startu w trybie live (kapitał startowy zostanie odczytany z konta przy "
            "pierwszym przebiegu z włączonymi zleceniami).")

rows = []
for key, label, fmt in [
    ("start_value", "Kapitał startowy (USD)", "{:,.2f}"), ("end_value", "Kapitał bieżący (USD)", "{:,.2f}"),
    ("total_return_pct", "Total return brutto", "{:+.2f}%"), ("api_cost_usd", "Koszt API (USD)", "{:,.4f}"),
    ("net_pnl_usd", "Wynik netto (USD)", "{:+,.2f}"), ("net_return_pct", "Total return netto", "{:+.2f}%"),
    ("sharpe", "Sharpe (ann.)", "{:.2f}"), ("max_drawdown_pct", "Max drawdown", "{:.2f}%"),
    ("win_rate_pct", "Win rate", "{:.1f}%"), ("n_trades", "Liczba transakcji", "{:d}"),
]:
    row = {"Metryka": label}
    for a in ("llm", "sma", "spy"):
        m = metrics["agents"][a]
        val = m.get(key) if m.get("available") else None
        row[AGENT_LABELS[a]] = "—" if val is None else fmt.format(val)
    rows.append(row)
st.subheader("Metryki")
st.dataframe(pd.DataFrame(rows).set_index("Metryka"), width="stretch")

# ------------------------------------------------------------------ equity
st.subheader("Equity curve")
frames = []
for a, pts in metrics["series"].items():
    if pts:
        frames.append(pd.Series([p[1] for p in pts], index=pd.to_datetime([p[0] for p in pts]), name=AGENT_LABELS[a]))
if frames:
    st.line_chart(pd.concat(frames, axis=1).sort_index().ffill())
else:
    st.write("Brak danych equity.")

# ------------------------------------------------------------------ decyzje
st.subheader("Log decyzji")
c1, c2, c3, c4 = st.columns(4)
agent_f = c1.multiselect("Agent", ["llm", "sma"], default=["llm", "sma"])
action_f = c2.multiselect("Akcja", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
hide_silent = c3.checkbox("Ukryj HOLD bez wywołania API (no_trigger / reguła)", value=True)
include_dry = c4.checkbox("Pokaż dry-run", value=True)
limit = st.slider("Liczba wierszy", 50, 2000, 300, step=50)

dec = df("""SELECT d.id, d.created_at, d.agent, d.ticker, d.action, d.size_pct, d.confidence, d.source,
                   d.risk_status, d.risk_reason, d.order_qty, d.limit_price, d.stop_price, d.reasoning,
                   d.triggers, d.validation_error, d.dry_run, d.model, d.effort, d.frequency, d.prompt_hash,
                   d.config_hash, c.cost_usd
            FROM decisions d LEFT JOIN llm_calls c ON c.id = d.llm_call_id
            ORDER BY d.id DESC LIMIT ?""", (limit * 5,))
if not dec.empty:
    dec = dec[dec["agent"].isin(agent_f) & dec["action"].isin(action_f)]
    if hide_silent:
        dec = dec[~((dec["action"] == "HOLD") & dec["source"].isin(["no_trigger", "rule", "halted"]))]
    if not include_dry:
        dec = dec[dec["dry_run"] == 0]
    dec = dec.head(limit)
    st.dataframe(dec.drop(columns=["reasoning"]), width="stretch", hide_index=True)
    st.markdown("**Uzasadnienia** (najnowsze na górze)")
    for _, r in dec.head(50).iterrows():
        title = (f"{r['created_at']} · {r['agent']} · {r['ticker']} · {r['action']} {r['size_pct']:.0f}% "
                 f"(conf {r['confidence']:.2f}) → {r['risk_status']}{' · DRY' if r['dry_run'] else ''}")
        with st.expander(title):
            st.write(r["reasoning"])
            st.caption(f"źródło: {r['source']} · triggery: {r['triggers']} · risk: {r['risk_reason']}")
            if r["validation_error"]:
                st.error(f"Walidacja: {r['validation_error']}")
else:
    st.write("Brak decyzji.")

# ------------------------------------------------------------------ koszty API
st.subheader("Koszty Claude API")
costs = df("""SELECT substr(c.created_at, 1, 10) AS day, r.mode, COUNT(*) AS calls, SUM(c.cost_usd) AS cost_usd,
                     SUM(c.input_tokens) AS input_tokens, SUM(c.output_tokens) AS output_tokens,
                     SUM(c.cache_read_tokens) AS cache_read_tokens
              FROM llm_calls c JOIN runs r ON r.id = c.run_id GROUP BY day, r.mode ORDER BY day""")
if not costs.empty:
    st.bar_chart(costs.pivot_table(index="day", columns="mode", values="cost_usd", aggfunc="sum").fillna(0))
    st.dataframe(costs, width="stretch", hide_index=True)
variants = df("""SELECT c.model_requested, c.model_served, c.effort, c.prompt_hash, COUNT(*) AS calls,
                        ROUND(SUM(c.cost_usd), 4) AS cost_usd, ROUND(AVG(c.latency_ms)) AS avg_latency_ms,
                        SUM(CASE WHEN c.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
                 FROM llm_calls c GROUP BY 1, 2, 3, 4""")
if not variants.empty:
    st.markdown("Wg wariantu")
    st.dataframe(variants, width="stretch", hide_index=True)

# ------------------------------------------------------------------ zlecenia / transakcje
st.subheader("Zlecenia")
st.dataframe(df("SELECT * FROM orders ORDER BY id DESC LIMIT 300"), width="stretch", hide_index=True)
st.subheader("Transakcje (realizacje)")
st.dataframe(df("SELECT * FROM trades ORDER BY id DESC LIMIT 300"), width="stretch", hide_index=True)

with st.expander("Przebiegi i wersje konfiguracji"):
    st.dataframe(df("SELECT * FROM runs ORDER BY id DESC LIMIT 200"), width="stretch", hide_index=True)
    st.dataframe(df("SELECT config_hash, created_at, model, effort, frequency, prompt_hash FROM config_versions "
                    "ORDER BY created_at DESC"), width="stretch", hide_index=True)
