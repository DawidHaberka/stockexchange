#!/usr/bin/env python3
"""Eksperymentalny agent tradingowy LLM vs SMA vs SPY — WYŁĄCZNIE Alpaca Paper.

Przykłady:
  python main.py --check                 # połączenia z oboma kontami, dane, Claude API
  python main.py --dry-run               # pełny przebieg bez wysyłania zleceń
  python main.py --dry-run --force       # j.w., ale ignoruje harmonogram (np. poza sesją)
  python main.py --run                   # przebieg wg harmonogramu (zlecenia tylko gdy orders_enabled)
  python main.py --report                # raport: LLM vs SMA vs SPY
  python main.py --backtest-sma          # backtest agenta SMA na historycznych świecach 1H
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
    g.add_argument("--backtest-sma", action="store_true", help="backtest agenta SMA (tylko SMA!)")
    p.add_argument("--dry-run", action="store_true", help="wszystko poza wysłaniem zleceń (implikuje --run)")
    p.add_argument("--force", action="store_true", help="tylko z --dry-run: ignoruj harmonogram/godziny sesji")
    p.add_argument("--agents", default="llm,sma", help="lista agentów, np. llm lub llm,sma")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--symbols", default=None, help="backtest: lista tickerów (domyślnie watchlista)")
    p.add_argument("--days", type=int, default=365, help="backtest: liczba dni historii")
    p.add_argument("--cash", type=float, default=10_000.0, help="backtest: kapitał startowy")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.dry_run and not (args.run or args.check or args.report or args.backtest_sma):
        args.run = True
    if not (args.run or args.check or args.report or args.backtest_sma):
        p.error("podaj jedną z opcji: --run, --dry-run, --check, --report, --backtest-sma")
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


def open_db(cfg):
    from trader.storage.db import Database

    return Database(cfg.storage.db_path)


# --------------------------------------------------------------------------- commands
def cmd_run(cfg, secrets, args) -> int:
    from trader.agent.llm_agent import make_anthropic_client
    from trader.runner import Runner

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    brokers = build_brokers(cfg, secrets, agents)
    llm_client = make_anthropic_client(cfg, secrets.anthropic_api_key) if "llm" in agents else None
    db = open_db(cfg)
    runner = Runner(cfg, db, brokers, build_market_data(cfg, secrets), llm_client)
    summary = runner.run(dry_run=args.dry_run, force=args.force, agents=agents)
    print_run_summary(summary)
    return 0 if summary.status in ("ok", "skipped") else 1


def print_run_summary(s) -> None:
    print()
    print(f"=== PODSUMOWANIE PRZEBIEGU: {s.status.upper()} ({s.mode}) — {s.reason}")
    if s.run_id is None:
        return
    print(f"run_id={s.run_id}  wywołania Claude: {s.llm_calls}  koszt: {s.llm_cost_usd:.4f} USD")
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
        try:
            from trader.agent.llm_agent import make_anthropic_client

            client = make_anthropic_client(cfg, secrets.anthropic_api_key)
            m = client.models.retrieve(cfg.llm.model)
            priced = cfg.pricing_for(cfg.llm.model) is not None
            print(f"Claude API: model {m.id} dostępny{'' if priced else ' — UWAGA: brak cennika w config.yaml'}")
        except Exception as exc:
            ok = False
            print(f"Claude API: BŁĄD {type(exc).__name__}: {exc}")
    print("WYNIK: OK" if ok else "WYNIK: SĄ PROBLEMY (patrz wyżej)")
    return 0 if ok else 1


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


def cmd_backtest(cfg, secrets, args) -> int:
    import pandas as pd

    from trader.evaluation.backtest_sma import backtest_watchlist, results_table
    from trader.storage.db import utcnow

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    results = backtest_watchlist(cfg, build_market_data(cfg, secrets), utcnow(), args.days, symbols, args.cash)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(results_table(results).round(2).to_string(index=False))
    print("\nUwaga: backtest dotyczy wyłącznie agenta SMA. Agent LLM nie jest backtestowany (look-ahead bias).")
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
            return cmd_backtest(cfg, secrets, args)
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
