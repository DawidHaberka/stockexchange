#!/usr/bin/env python3
"""Eksperymentalny agent tradingowy LLM vs SMA vs SPY — WYŁĄCZNIE Alpaca Paper.

Przykłady:
  python main.py --check                 # połączenia z oboma kontami, dane, Claude Code CLI
  python main.py --dry-run               # pełny przebieg bez wysyłania zleceń
  python main.py --dry-run --force       # j.w., ale ignoruje harmonogram (np. poza sesją)
  python main.py --run                   # przebieg wg harmonogramu (zlecenia tylko gdy orders_enabled)
  python main.py --report                # raport: LLM vs SMA vs SPY
  python main.py --backtest --agents sma                 # backtest portfelowy SMA vs SPY (ten sam kod co live)
  python main.py --backtest --agents llm,sma --start 2026-07-14 --parallel 4   # replay LLM po granicy wiedzy
  python main.py --backtest-grid --grid 10/30,20/50,50/200 --folds 4         # wrażliwość SMA + walk-forward
  python main.py --backtest-sma          # szybki backtest SMA per ticker (backtesting.py)
  streamlit run trader/dashboard/app.py  # dashboard
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

from pydantic import ValidationError

from trader.config import ConfigError, load_config, load_secrets
from trader.logging_setup import setup_logging
from trader.safety import LiveTradingForbidden, assert_paper_url

log = logging.getLogger("main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM paper-trading experiment (Alpaca Paper only)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run", action="store_true", help="przebieg wg harmonogramu (cron co 15 min)")
    g.add_argument("--check", action="store_true", help="sprawdź połączenia i konfigurację")
    g.add_argument("--report", action="store_true", help="raport dzienny LLM vs SMA vs SPY")
    g.add_argument("--backtest-sma", action="store_true", help="szybki backtest SMA per ticker (backtesting.py)")
    g.add_argument("--backtest", action="store_true", help="backtest portfelowy (replay) — sma i/lub llm")
    g.add_argument("--backtest-grid", action="store_true", help="wrażliwość parametrów SMA + walk-forward")
    p.add_argument("--dry-run", action="store_true", help="wszystko poza wysłaniem zleceń (implikuje --run)")
    p.add_argument("--force", action="store_true", help="tylko z --dry-run: ignoruj harmonogram/godziny sesji")
    p.add_argument("--agents", default="llm,sma", help="lista agentów, np. llm lub llm,sma")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--symbols", default=None, help="backtest: lista tickerów (domyślnie watchlista)")
    p.add_argument("--days", type=int, default=365, help="backtest: liczba dni historii")
    p.add_argument("--cash", type=float, default=10_000.0, help="backtest-sma: kapitał startowy")
    p.add_argument("--start", default=None, help="backtest: data początkowa YYYY-MM-DD")
    p.add_argument("--end", default=None, help="backtest: data końcowa YYYY-MM-DD (domyślnie wczoraj)")
    p.add_argument("--name", default=None, help="backtest: nazwa (katalog wyników; ta sama nazwa = wznowienie)")
    p.add_argument("--fresh", action="store_true", help="backtest: zacznij od nowa (usuwa zapisany stan)")
    p.add_argument("--parallel", type=int, default=None, help="backtest LLM: równoległe wywołania claude")
    p.add_argument("--grid", default="10/30,20/50,50/200", help="backtest-grid: pary fast/slow")
    p.add_argument("--folds", type=int, default=4, help="backtest-grid: liczba odcinków walk-forward")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    commands = (args.run, args.check, args.report, args.backtest_sma, args.backtest, args.backtest_grid)
    if args.dry_run and not any(commands):
        args.run = True
    if not any((args.run, *commands[1:])):
        p.error("podaj jedną z opcji: --run, --dry-run, --check, --report, --backtest, --backtest-grid, "
                "--backtest-sma")
    if args.dry_run and not args.run:
        p.error("--dry-run dotyczy tylko przebiegu (--run)")
    if args.force and not args.dry_run:
        p.error("--force jest dozwolone tylko razem z --dry-run")
    return args


# --------------------------------------------------------------------------- builders
def build_brokers(cfg, secrets, agents):
    from trader.broker import make_broker

    return {a: make_broker(a, secrets.accounts.get(a)) for a in agents}


def build_market_data(cfg, secrets):
    from trader.data.market_data import AlpacaMarketData

    creds = secrets.accounts.get("llm") or next(iter(secrets.accounts.values()), None)
    if creds is None:
        raise ConfigError("Brak kluczy Alpaca w .env")
    return AlpacaMarketData(creds, cfg.data.feed)


def build_llm_cli(cfg):
    from pathlib import Path

    from trader.agent.claude_cli import ClaudeCLI

    # pusty katalog roboczy: Claude Code nie wczyta CLAUDE.md ani ustawień projektu
    return ClaudeCLI(cfg.llm, Path(cfg.storage.db_path).parent / "llm_workdir")


def open_db(cfg):
    from trader.storage.db import Database

    return Database(cfg.storage.db_path)


# --------------------------------------------------------------------------- commands
def cmd_run(cfg, secrets, args) -> int:
    from trader.runner import Runner

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    brokers = build_brokers(cfg, secrets, agents)
    llm_cli = build_llm_cli(cfg) if "llm" in agents else None
    db = open_db(cfg)
    runner = Runner(cfg, db, brokers, build_market_data(cfg, secrets), llm_cli)
    summary = runner.run(dry_run=args.dry_run, force=args.force, agents=agents)
    print_run_summary(summary)
    return 0 if summary.status in ("ok", "skipped") else 1


def print_run_summary(s) -> None:
    print()
    print(f"=== PODSUMOWANIE PRZEBIEGU: {s.status.upper()} ({s.mode}) — {s.reason}")
    if s.run_id is None:
        return
    print(f"run_id={s.run_id}  wywołania Claude: {s.llm_calls}  "
          f"(równoważnik cen API: {s.llm_cost_equiv_usd:.4f} USD — nie jest pobierany przy subskrypcji)")
    if s.data_problems:
        print("Problemy z danymi: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in s.data_problems.items()))
    for name, r in s.agents.items():
        print()
        chg = f"{r.daily_change_pct:+.2f}%" if r.daily_change_pct is not None else "n/a"
        print(f"--- Agent {name.upper()}: equity {r.equity_start} -> {r.equity_end} | zmiana dzienna {chg}"
              f"{' | HANDEL WSTRZYMANY' if r.halted else ''}{' | BŁĄD: ' + r.error if r.error else ''}")
        for n in r.reconcile_notes:
            print(f"  rekoncyliacja: {n}")
        for v in r.verdicts:
            d = v.agent_decision.decision
            trig = f" [{', '.join(v.agent_decision.triggers)}]" if v.agent_decision.triggers else ""
            print(f"  {d.ticker:<6} {d.action.value:<4} {d.size_pct:5.1f}% conf={d.confidence:.2f} "
                  f"src={v.agent_decision.source:<10} risk={v.status:<8}{trig}")
            if d.action.value != "HOLD" or v.agent_decision.source not in ("no_trigger", "rule", "halted"):
                print(f"         uzasadnienie: {d.reasoning}")
            if v.status != "hold":
                print(f"         risk: {v.reason}")
        for e in r.executions:
            print(f"  zlecenie: {e.side.upper()} {e.requested_qty} {e.symbol} -> {e.status}"
                  f" (zrealizowano {e.filled_qty:g} @ {e.avg_price})")


def cmd_check(cfg, secrets, args) -> int:
    from trader.broker import NY
    from trader.storage.db import utcnow

    ok = True
    print(f"Konfiguracja: {args.config} | wariant {cfg.variant()}")
    print(f"ALPACA_BASE_URL: {assert_paper_url(secrets.alpaca_base_url)} (paper OK)")
    print(f"execution.orders_enabled = {cfg.execution.orders_enabled}"
          + ("  (zlecenia WYŁĄCZONE — każdy przebieg = dry-run)" if not cfg.execution.orders_enabled else ""))
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    equities = {}
    brokers = {}
    for a in agents:
        try:
            b = build_brokers(cfg, secrets, [a])[a]
            acc = b.get_account()
            pos = b.get_positions()
            orders = b.get_open_orders()
            brokers[a] = b
            equities[a] = acc.equity
            print(f"[{a}] konto {acc.account_number} status={acc.status} equity={acc.equity:,.2f} "
                  f"cash={acc.cash:,.2f} buying_power={acc.buying_power:,.2f} last_equity={acc.last_equity:,.2f} "
                  f"blokada={acc.trading_blocked} pozycje={len(pos)} otwarte_zlecenia={len(orders)}")
            if acc.trading_blocked:
                ok = False
        except Exception as exc:
            ok = False
            print(f"[{a}] BŁĄD połączenia: {type(exc).__name__}: {exc}")
    if len(equities) == 2 and abs(equities["llm"] - equities["sma"]) > 1:
        print(f"UWAGA: konta mają różne equity ({equities['llm']:,.2f} vs {equities['sma']:,.2f}) — "
              "porównanie wymaga tego samego kapitału startowego.")
    if len(equities) == 2 and len({b.get_account().account_number for b in brokers.values()}) == 1:
        ok = False
        print("BŁĄD: oba agenty wskazują na TO SAMO konto — potrzebne są dwa osobne konta paper.")

    if brokers:
        b = next(iter(brokers.values()))
        clock = b.get_clock()
        sess = b.get_session(utcnow().astimezone(NY).date())
        print(f"Zegar: open={clock.is_open} teraz={clock.timestamp:%Y-%m-%d %H:%M %Z} "
              f"next_open={clock.next_open:%Y-%m-%d %H:%M} next_close={clock.next_close:%Y-%m-%d %H:%M}"
              + (f" | sesja dziś {sess.open:%H:%M}-{sess.close:%H:%M} NY" if sess else " | dziś brak sesji"))
    try:
        md = build_market_data(cfg, secrets)
        now = utcnow()
        sym = cfg.benchmark_symbol
        h = md.hourly_bars([sym], now - timedelta(days=10), now).get(sym)
        d = md.daily_bars([sym], now - timedelta(days=30), now).get(sym)
        q = md.latest_quotes([sym]).get(sym)
        n = md.news(sym, now - timedelta(hours=cfg.data.news_lookback_hours), 5)
        print(f"Dane ({cfg.data.feed}): {sym} świece 1H={0 if h is None else len(h)} 1D={0 if d is None else len(d)} "
              f"quote bid={q.bid} ask={q.ask} last={q.last} | newsy 24h: {len(n)}")
    except Exception as exc:
        ok = False
        print(f"Dane: BŁĄD {type(exc).__name__}: {exc}")
    if "llm" in agents:
        ok = check_claude_cli(cfg) and ok
    print("WYNIK: OK" if ok else "WYNIK: SĄ PROBLEMY (patrz wyżej)")
    return 0 if ok else 1


def check_claude_cli(cfg) -> bool:
    """Claude Code CLI: instalacja, logowanie subskrypcją i jedno testowe wywołanie (effort=low)."""
    import json

    from trader.agent.schemas import DECISION_JSON_SCHEMA, parse_decision

    cli = build_llm_cli(cfg)
    version = cli.version()
    if not version:
        print(f"Claude Code: BŁĄD — nie znaleziono '{cfg.llm.cli_path}'. Zainstaluj Claude Code (README).")
        return False
    auth = cli.auth_status() or {}
    method = str(auth.get("authMethod", ""))
    print(f"Claude Code: {version} | zalogowany={auth.get('loggedIn')} metoda={method or 'n/a'}")
    if not auth.get("loggedIn"):
        print("Claude Code: BŁĄD — niezalogowany. Uruchom `claude auth login` (konto z subskrypcją Claude).")
        return False
    if "api" in method.lower() and "key" in method.lower():
        print("Claude Code: BŁĄD — logowanie kluczem API (płatne). Zaloguj się kontem z subskrypcją: "
              "`claude auth logout` i `claude auth login`.")
        return False
    res = cli.ask("Ticker: TEST\nThis is a connectivity test. Return HOLD for TEST with size_pct 0, "
                  "confidence 0 and reasoning 'test'.", DECISION_JSON_SCHEMA, effort="low")
    if not res.ok:
        print(f"Claude Code: BŁĄD testowego wywołania modelu {cfg.llm.model}: {res.error}")
        return False
    raw = json.dumps(res.structured) if res.structured is not None else res.text
    _, err = parse_decision(raw, "TEST")
    if err:
        print(f"Claude Code: BŁĄD — odpowiedź nie przeszła walidacji: {err}")
        return False
    print(f"Claude Code: testowe wywołanie OK — model {res.model_served}, {res.duration_ms} ms, "
          f"równoważnik cen API {res.cost_equiv_usd:.4f} USD (przy subskrypcji nie jest pobierany)")
    return True


def cmd_report(cfg, secrets, args) -> int:
    from trader.evaluation.report import build_report, collect_eod
    from trader.storage.db import utcnow

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    db = open_db(cfg)
    brokers = build_brokers(cfg, secrets, agents)
    session_date = collect_eod(cfg, db, brokers, build_market_data(cfg, secrets), utcnow())
    res = build_report(cfg, db, session_date)
    print(res.markdown)
    print(f"Zapisano: {res.md_path}" + (f", {res.png_path}" if res.png_path else ""))
    return 0


def cmd_backtest_sma(cfg, secrets, args) -> int:
    import pandas as pd

    from trader.evaluation.backtest_sma import backtest_watchlist, results_table
    from trader.storage.db import utcnow

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    results = backtest_watchlist(cfg, build_market_data(cfg, secrets), utcnow(), args.days, symbols, args.cash)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(results_table(results).round(2).to_string(index=False))
    print("\nUwaga: backtest dotyczy wyłącznie agenta SMA. Agent LLM nie jest backtestowany (look-ahead bias).")
    return 0


def _backtest_window(cfg, args, llm: bool):
    from datetime import date, timedelta

    from trader.broker import NY
    from trader.storage.db import utcnow

    end = date.fromisoformat(args.end) if args.end else utcnow().astimezone(NY).date() - timedelta(days=1)
    if args.start:
        start = date.fromisoformat(args.start)
    elif llm:
        start = cfg.backtest.llm_not_before(cfg.llm.model) or end
    else:
        start = end - timedelta(days=365)
    return start, end


def _load_store(cfg, secrets, start, end, with_news, slow):
    import math

    from trader.backtest.history import load_history
    from trader.data.market_data import AlpacaMarketData

    creds = secrets.accounts.get("llm") or next(iter(secrets.accounts.values()), None)
    if creds is None:
        raise ConfigError("Brak kluczy Alpaca w .env (potrzebne do pobrania historii)")
    md = AlpacaMarketData(creds, cfg.data.feed, cfg.backtest.adjustment)
    symbols = list(dict.fromkeys([*cfg.watchlist, cfg.benchmark_symbol]))
    # SMA na 1H potrzebuje `slow` zamkniętych świec (~6,5 na sesję) + zapas na weekendy/święta
    hourly_days = max(cfg.data.hourly_lookback_days, math.ceil(slow / 6.5 * 1.6) + 7)
    return load_history(md, symbols, start, end, hourly_days, cfg.data.daily_lookback_days, with_news,
                        cfg.data.news_lookback_hours, cfg.backtest.cache_dir, cfg.data.feed, cfg.backtest.adjustment)


def cmd_backtest(cfg, secrets, args) -> int:
    from trader.backtest.engine import BacktestEngine, BacktestPaused, BacktestSpec, LookAheadForbidden, check_llm_window
    from trader.backtest.report import write_report

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    llm = "llm" in agents
    start, end = _backtest_window(cfg, args, llm)
    if llm:
        try:
            check_llm_window(cfg, start)
        except LookAheadForbidden as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 2
        if args.parallel:
            cfg.llm.max_parallel = args.parallel
    name = args.name or f"{'-'.join(agents)}_{start}_{end}"
    store = _load_store(cfg, secrets, start, end, llm, cfg.sma.slow)
    spec = BacktestSpec(name, start, end, agents, fresh=args.fresh)
    engine = BacktestEngine(cfg, store, spec, build_llm_cli(cfg) if llm else None)
    try:
        res = engine.run()
    except BacktestPaused as exc:
        print(f"\nPAUZA: {exc}")
        return 3
    path = write_report(res, res.extra["db"])
    print(path.read_text(encoding="utf-8"))
    print(f"Wyniki: {res.out_dir} (czas {res.elapsed_s:.0f} s)")
    return 0


def cmd_backtest_grid(cfg, secrets, args) -> int:
    from pathlib import Path

    from trader.backtest.grid import parse_pairs, run_grid, write_grid_report

    start, end = _backtest_window(cfg, args, llm=False)
    pairs = parse_pairs(args.grid)
    store = _load_store(cfg, secrets, start, end, False, max(s for _, s in pairs))
    res = run_grid(cfg, store, start, end, pairs, args.folds)
    path = write_grid_report(res, Path(cfg.backtest.dir) / (args.name or f"grid_{start}_{end}"), start, end)
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        setup_logging(cfg.storage.log_dir, args.verbose)
        secrets = load_secrets(cfg, args.env)
        assert_paper_url(secrets.alpaca_base_url)   # twardy bezpiecznik przed czymkolwiek innym
        if args.check:
            return cmd_check(cfg, secrets, args)
        if args.report:
            return cmd_report(cfg, secrets, args)
        if args.backtest_sma:
            return cmd_backtest_sma(cfg, secrets, args)
        if args.backtest:
            return cmd_backtest(cfg, secrets, args)
        if args.backtest_grid:
            return cmd_backtest_grid(cfg, secrets, args)
        return cmd_run(cfg, secrets, args)
    except LiveTradingForbidden as exc:
        logging.getLogger("main").critical(str(exc))
        print(f"\nSTOP: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ValidationError) as exc:
        print(f"Błąd konfiguracji: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
