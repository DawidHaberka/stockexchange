-- Schemat bazy SQLite eksperymentu. Wszystkie znaczniki czasu w ISO-8601 (UTC).

CREATE TABLE IF NOT EXISTS config_versions (
    config_hash   TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    model         TEXT NOT NULL,
    effort        TEXT NOT NULL,
    frequency     TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    config_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    session_date TEXT NOT NULL,
    slot_start   TEXT,
    mode         TEXT NOT NULL,            -- live | dry_run
    status       TEXT NOT NULL,            -- running | ok | error
    message      TEXT,
    config_hash  TEXT NOT NULL REFERENCES config_versions(config_hash)
);
CREATE INDEX IF NOT EXISTS idx_runs_slot ON runs(slot_start, mode, status);

-- Kapitał startowy czytany z konta przy pierwszym uruchomieniu live.
CREATE TABLE IF NOT EXISTS experiment_start (
    agent           TEXT PRIMARY KEY,      -- llm | sma
    started_at      TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    start_equity    REAL NOT NULL,
    account_number  TEXT,
    benchmark_symbol TEXT NOT NULL,
    benchmark_price REAL                   -- cena SPY w momencie startu
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER REFERENCES runs(id),
    ticker                TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    model_requested       TEXT NOT NULL,
    model_served          TEXT,
    effort                TEXT,
    prompt_hash           TEXT,
    input_tokens          INTEGER DEFAULT 0,
    output_tokens         INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens     INTEGER DEFAULT 0,
    cost_usd              REAL DEFAULT 0,
    latency_ms            INTEGER,
    stop_reason           TEXT,
    request_id            TEXT,
    error                 TEXT,
    user_prompt           TEXT,
    raw_response          TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs(id),
    agent         TEXT NOT NULL,           -- llm | sma
    ticker        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    action        TEXT NOT NULL,           -- BUY | SELL | HOLD (decyzja agenta)
    size_pct      REAL NOT NULL,
    confidence    REAL NOT NULL,
    reasoning     TEXT NOT NULL,
    source        TEXT NOT NULL,           -- llm | no_trigger | invalid | api_error | refusal | rule | halted
    triggers      TEXT,                    -- JSON lista triggerów
    llm_call_id   INTEGER REFERENCES llm_calls(id),
    validation_error TEXT,
    risk_status   TEXT,                    -- approved | resized | rejected | hold
    risk_reason   TEXT,
    order_side    TEXT,
    order_qty     INTEGER,
    limit_price   REAL,
    stop_price    REAL,
    notional      REAL,
    dry_run       INTEGER NOT NULL,
    -- wersja konfiguracji (wariant eksperymentu)
    config_hash   TEXT NOT NULL,
    model         TEXT,
    effort        TEXT,
    frequency     TEXT NOT NULL,
    prompt_hash   TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent, created_at);

CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER REFERENCES runs(id),
    agent             TEXT NOT NULL,
    decision_id       INTEGER REFERENCES decisions(id),
    alpaca_order_id   TEXT UNIQUE,
    client_order_id   TEXT,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,       -- buy | sell
    order_type        TEXT NOT NULL,       -- limit | stop
    purpose           TEXT NOT NULL,       -- entry | exit | stop_loss | external
    qty               REAL NOT NULL,
    limit_price       REAL,
    stop_price        REAL,
    status            TEXT NOT NULL,
    filled_qty        REAL DEFAULT 0,
    filled_avg_price  REAL,
    submitted_at      TEXT,
    updated_at        TEXT,
    note              TEXT,
    dry_run           INTEGER NOT NULL,
    fill_recorded     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_orders_agent ON orders(agent, symbol, status);

-- Transakcje = (częściowe) realizacje zleceń.
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent           TEXT NOT NULL,
    order_id        INTEGER REFERENCES orders(id),
    alpaca_order_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    filled_at       TEXT NOT NULL,
    purpose         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent, symbol, filled_at);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent           TEXT NOT NULL,
    run_id          INTEGER REFERENCES runs(id),
    taken_at        TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    kind            TEXT NOT NULL,         -- run_start | run_end | eod
    equity          REAL NOT NULL,
    last_equity     REAL,
    cash            REAL NOT NULL,
    positions_value REAL NOT NULL,
    positions_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_snap_agent ON portfolio_snapshots(agent, session_date, kind);

CREATE TABLE IF NOT EXISTS risk_state (
    agent        TEXT NOT NULL,
    session_date TEXT NOT NULL,
    halted       INTEGER NOT NULL,
    reason       TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (agent, session_date)
);

-- Stan per ticker do wyliczania triggerów LLM.
CREATE TABLE IF NOT EXISTS ticker_state (
    agent          TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    last_run_at    TEXT,
    last_price     REAL,
    last_llm_at    TEXT,
    last_llm_price REAL,
    PRIMARY KEY (agent, symbol)
);

-- Aktualny poziom stop-lossa dla pozycji (źródło prawdy: zlecenie stop w Alpaca).
CREATE TABLE IF NOT EXISTS stop_levels (
    agent       TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    stop_price  REAL NOT NULL,
    atr_used    REAL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (agent, symbol)
);

CREATE TABLE IF NOT EXISTS benchmark_daily (
    symbol       TEXT NOT NULL,
    session_date TEXT NOT NULL,
    close        REAL NOT NULL,
    PRIMARY KEY (symbol, session_date)
);

CREATE TABLE IF NOT EXISTS daily_reports (
    session_date TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    report_path  TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
