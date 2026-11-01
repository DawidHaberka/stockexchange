"""Wykonanie zleceń przez Alpaca Paper: zlecenia limit, częściowe realizacje, odrzucenia, stop-lossy.

Przebieg dla zatwierdzonej decyzji:
  SELL: anuluj stop-loss tickera -> limit sell -> czekaj na realizację (timeout) -> anuluj
        resztę -> jeśli coś zostało w pozycji, odtwórz stop dla pozostałej ilości.
  BUY:  limit buy -> czekaj na realizację (timeout) -> anuluj resztę -> jeśli cokolwiek
        się zrealizowało: stop-loss GTC dla CAŁEJ pozycji = średnia cena wejścia - k x ATR.
Stop-lossy są zleceniami po stronie Alpaca, więc działają także między uruchomieniami.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from ..broker import Broker, OrderRequest, OrderState
from ..config import AppConfig
from ..data.snapshot import MarketSnapshot
from ..risk.manager import RiskVerdict, round_price
from ..storage.db import Database, iso, utcnow

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    symbol: str
    side: str
    requested_qty: int
    status: str            # filled | partially_filled | unfilled | rejected | dry_run | error
    filled_qty: float = 0.0
    avg_price: float | None = None
    order_db_id: int | None = None
    note: str = ""


class Executor:
    def __init__(self, broker: Broker, db: Database, cfg: AppConfig, agent: str, dry_run: bool,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic):
        self.broker = broker
        self.db = db
        self.cfg = cfg
        self.agent = agent
        self.dry_run = dry_run
        self.sleep = sleep
        self.monotonic = monotonic

    # ================================================================== helpers
    def _client_id(self, run_id: int | None, symbol: str, purpose: str) -> str:
        return f"{self.agent}-{run_id or 0}-{symbol}-{purpose}-{uuid.uuid4().hex[:8]}"

    def _record_order(self, run_id, decision_id, req: OrderRequest, purpose: str, state: OrderState | None,
                      status: str, note: str | None = None) -> int:
        return self.db.insert_order({
            "run_id": run_id, "agent": self.agent, "decision_id": decision_id,
            "alpaca_order_id": state.id if state else None, "client_order_id": req.client_order_id,
            "symbol": req.symbol, "side": req.side, "order_type": req.order_type, "purpose": purpose,
            "qty": req.qty, "limit_price": req.limit_price, "stop_price": req.stop_price,
            "status": status, "filled_qty": state.filled_qty if state else 0,
            "filled_avg_price": state.filled_avg_price if state else None,
            "submitted_at": iso(state.submitted_at) if state and state.submitted_at else iso(utcnow()),
            "updated_at": iso(utcnow()), "note": note, "dry_run": 1 if self.dry_run else 0,
            "fill_recorded": 0 if (state and not state.is_terminal) else 1,
        })

    def _settle(self, order_db_id: int, state: OrderState, purpose: str) -> None:
        """Aktualizuje zlecenie w bazie i — gdy terminalne — zapisuje transakcję (realizację)."""
        self.db.update_order(order_db_id, {
            "status": state.status, "filled_qty": state.filled_qty, "filled_avg_price": state.filled_avg_price,
            "updated_at": iso(utcnow()), "fill_recorded": 1 if state.is_terminal else 0,
        })
        if state.is_terminal and state.filled_qty > 0 and state.filled_avg_price:
            self.db.insert_trade({
                "agent": self.agent, "order_id": order_db_id, "alpaca_order_id": state.id,
                "symbol": state.symbol, "side": state.side, "qty": state.filled_qty,
                "price": state.filled_avg_price, "filled_at": iso(state.filled_at or utcnow()), "purpose": purpose,
            })

    def _wait_for_terminal(self, order: OrderState, timeout_s: float) -> OrderState:
        deadline = self.monotonic() + timeout_s
        while not order.is_terminal and self.monotonic() < deadline:
            self.sleep(self.cfg.execution.poll_interval_s)
            order = self.broker.get_order(order.id)
        return order

    def _cancel_and_settle(self, order: OrderState) -> OrderState:
        """Anuluje niezrealizowaną resztę i czeka, aż zlecenie będzie terminalne."""
        if order.is_terminal:
            return order
        try:
            self.broker.cancel_order(order.id)
        except Exception as exc:  # mogło się właśnie zrealizować
            log.warning("[%s] anulowanie %s nie powiodło się: %s", self.agent, order.id, exc)
        for _ in range(12):
            order = self.broker.get_order(order.id)
            if order.is_terminal:
                break
            self.sleep(self.cfg.execution.poll_interval_s)
        return order

    def _open_stop_orders(self, symbol: str) -> list[OrderState]:
        return [o for o in self.broker.get_open_orders([symbol])
                if o.symbol == symbol and o.order_type == "stop" and o.side == "sell"]

    def _cancel_stops(self, symbol: str) -> None:
        for o in self._open_stop_orders(symbol):
            final = self._cancel_and_settle(o)
            row = self.db.get_order_by_alpaca_id(o.id)
            if row is not None:
                self.db.update_order(row["id"], {"status": final.status, "filled_qty": final.filled_qty,
                                                 "filled_avg_price": final.filled_avg_price,
                                                 "updated_at": iso(utcnow()), "fill_recorded": 1})
                if final.filled_qty > 0 and final.filled_avg_price:
                    self.db.insert_trade({
                        "agent": self.agent, "order_id": row["id"], "alpaca_order_id": final.id,
                        "symbol": symbol, "side": "sell", "qty": final.filled_qty, "price": final.filled_avg_price,
                        "filled_at": iso(final.filled_at or utcnow()), "purpose": "stop_loss",
                    })

    def place_stop(self, run_id, symbol: str, qty: int, stop_price: float, atr: float | None) -> bool:
        """Stawia stop-loss GTC dla `qty` akcji. Zwraca True, jeśli zlecenie przyjęte."""
        stop_price = round_price(stop_price)
        req = OrderRequest(symbol=symbol, side="sell", qty=qty, order_type="stop", stop_price=stop_price,
                           time_in_force="gtc", client_order_id=self._client_id(run_id, symbol, "stop"))
        if self.dry_run:
            log.info("[%s] DRY-RUN: postawiłbym stop-loss %s x%d @ %.2f", self.agent, symbol, qty, stop_price)
            return True
        try:
            state = self.broker.submit_order(req)
        except Exception as exc:
            log.error("[%s] UWAGA: stop-loss dla %s odrzucony (%s) — pozycja bez ochrony do następnego "
                      "uruchomienia", self.agent, symbol, exc)
            self._record_order(run_id, None, req, "stop_loss", None, "rejected", str(exc)[:500])
            return False
        self._record_order(run_id, None, req, "stop_loss", state, state.status)
        self.db.set_stop_level(self.agent, symbol, stop_price, atr)
        log.info("[%s] stop-loss %s x%d @ %.2f (GTC)", self.agent, symbol, qty, stop_price)
        return True

    # ================================================================== execute
    def execute(self, run_id: int | None, verdicts: list[RiskVerdict], decision_ids: list[int | None]) -> list[ExecutionResult]:
        """Wykonuje zatwierdzone zlecenia: najpierw SELL (zwalniają gotówkę), potem BUY."""
        pairs = [(v, did) for v, did in zip(verdicts, decision_ids, strict=True) if v.is_order]
        pairs.sort(key=lambda p: 0 if p[0].intent.side == "sell" else 1)
        results = []
        for verdict, decision_id in pairs:
            try:
                results.append(self._execute_one(run_id, verdict, decision_id))
            except Exception as exc:
                log.exception("[%s] błąd wykonania %s", self.agent, verdict.intent.symbol)
                results.append(ExecutionResult(verdict.intent.symbol, verdict.intent.side, verdict.intent.qty,
                                               "error", note=str(exc)[:300]))
        return results

    def _execute_one(self, run_id, verdict: RiskVerdict, decision_id) -> ExecutionResult:
        it = verdict.intent
        purpose = "entry" if it.side == "buy" else "exit"
        req = OrderRequest(symbol=it.symbol, side=it.side, qty=it.qty, order_type="limit",
                           limit_price=it.limit_price, time_in_force="day",
                           client_order_id=self._client_id(run_id, it.symbol, purpose))
        reasoning = verdict.agent_decision.decision.reasoning

        if self.dry_run:
            oid = self._record_order(run_id, decision_id, req, purpose, None, "dry_run",
                                     f"stop={it.stop_price}" if it.stop_price else None)
            log.info("[%s] DRY-RUN: %s %d %s @ limit %.2f%s | uzasadnienie: %s", self.agent, it.side.upper(),
                     it.qty, it.symbol, it.limit_price,
                     f", stop {it.stop_price:.2f}" if it.stop_price else "", reasoning)
            return ExecutionResult(it.symbol, it.side, it.qty, "dry_run", order_db_id=oid)

        if it.side == "sell":
            self._cancel_stops(it.symbol)   # akcje zablokowane przez stop muszą być wolne

        try:
            state = self.broker.submit_order(req)
        except Exception as exc:
            oid = self._record_order(run_id, decision_id, req, purpose, None, "rejected", str(exc)[:500])
            log.error("[%s] zlecenie %s %d %s ODRZUCONE: %s", self.agent, it.side, it.qty, it.symbol, exc)
            if it.side == "sell":
                self._restore_stop_after_sell(run_id, it.symbol)
            return ExecutionResult(it.symbol, it.side, it.qty, "rejected", order_db_id=oid, note=str(exc)[:300])

        oid = self._record_order(run_id, decision_id, req, purpose, state, state.status)
        log.info("[%s] wysłano %s %d %s @ limit %.2f (id=%s) | uzasadnienie: %s", self.agent, it.side.upper(),
                 it.qty, it.symbol, it.limit_price, state.id, reasoning)

        state = self._wait_for_terminal(state, self.cfg.execution.fill_timeout_s)
        if not state.is_terminal:
            log.info("[%s] %s: timeout — anuluję niezrealizowaną resztę (%g/%d zrealizowane)", self.agent,
                     it.symbol, state.filled_qty, it.qty)
            state = self._cancel_and_settle(state)
        self._settle(oid, state, purpose)

        if state.status == "rejected":
            result_status = "rejected"
        elif state.filled_qty >= it.qty - 1e-9:
            result_status = "filled"
        elif state.filled_qty > 0:
            result_status = "partially_filled"
        else:
            result_status = "unfilled"
        log.info("[%s] %s %s: %s, zrealizowano %g/%d @ %s", self.agent, it.side.upper(), it.symbol, result_status,
                 state.filled_qty, it.qty, state.filled_avg_price)

        if it.side == "buy" and state.filled_qty > 0:
            self._protect_position_after_buy(run_id, it.symbol, it.atr_for_stop)
        elif it.side == "sell":
            self._restore_stop_after_sell(run_id, it.symbol)

        return ExecutionResult(it.symbol, it.side, it.qty, result_status, state.filled_qty, state.filled_avg_price,
                               oid, note=state.status)

    def _protect_position_after_buy(self, run_id, symbol: str, atr: float | None) -> None:
        pos = self.broker.get_positions().get(symbol)
        if pos is None or pos.qty < 1:
            return
        if not atr:
            log.error("[%s] brak ATR dla %s — nie mogę wyliczyć stop-lossa!", self.agent, symbol)
            return
        self._cancel_stops(symbol)
        stop = pos.avg_entry_price - self.cfg.risk.stop_loss_atr_mult * atr
        self.place_stop(run_id, symbol, int(math.floor(pos.qty + 1e-9)), stop, atr)

    def _restore_stop_after_sell(self, run_id, symbol: str) -> None:
        pos = self.broker.get_positions().get(symbol)
        if pos is None or pos.qty < 1:
            self.db.delete_stop_level(self.agent, symbol)
            return
        level = self.db.get_stop_level(self.agent, symbol)
        if level is None:
            log.error("[%s] brak zapisanego poziomu stop dla %s — uzupełni go rekoncyliacja", self.agent, symbol)
            return
        self.place_stop(run_id, symbol, int(math.floor(pos.qty + 1e-9)), level, None)

    # ================================================================== reconcile
    def reconcile(self, run_id: int | None, snapshot: MarketSnapshot) -> list[str]:
        """Synchronizacja z Alpaca na początku przebiegu.

        1. rozlicza zlecenia z bazy, które zrealizowały się między uruchomieniami (np. stop-lossy),
        2. anuluje osierocone zlecenia limit z poprzednich przebiegów,
        3. pilnuje, żeby każda pozycja miała stop-loss na pełną ilość, a stopy bez pozycji znikały.
        W dry-run tylko raportuje, co by zrobiła.
        """
        notes: list[str] = []
        # 1) rozliczenie śledzonych zleceń
        for row in self.db.tracked_unsettled_orders(self.agent):
            try:
                st = self.broker.get_order(row["alpaca_order_id"])
            except Exception as exc:
                notes.append(f"nie udało się pobrać zlecenia {row['alpaca_order_id']}: {exc}")
                continue
            if st.is_terminal:
                self._settle(row["id"], st, row["purpose"])
                if st.filled_qty > 0:
                    notes.append(f"zrealizowane między uruchomieniami: {row['purpose']} {st.side} "
                                 f"{st.filled_qty:g} {st.symbol} @ {st.filled_avg_price}")
                    if row["purpose"] == "stop_loss":
                        self.db.delete_stop_level(self.agent, st.symbol)

        positions = self.broker.get_positions()
        open_orders = self.broker.get_open_orders()

        # 2) osierocone zlecenia limit (np. po awarii w trakcie przebiegu)
        for o in open_orders:
            if o.order_type != "stop":
                notes.append(f"osierocone zlecenie {o.side} {o.symbol} ({o.status}) — anuluję")
                if not self.dry_run:
                    final = self._cancel_and_settle(o)
                    row = self.db.get_order_by_alpaca_id(o.id)
                    if row is not None:
                        self._settle(row["id"], final, row["purpose"])

        # 3) stop-lossy
        stops_by_symbol: dict[str, list[OrderState]] = {}
        for o in open_orders:
            if o.order_type == "stop" and o.side == "sell":
                stops_by_symbol.setdefault(o.symbol, []).append(o)

        for sym in stops_by_symbol:
            if sym not in positions or positions[sym].qty < 1:
                notes.append(f"stop-loss {sym} bez pozycji — anuluję")
                if not self.dry_run:
                    self._cancel_stops(sym)
                    self.db.delete_stop_level(self.agent, sym)

        for sym, pos in positions.items():
            qty = int(math.floor(pos.qty + 1e-9))
            if qty < 1:
                continue
            stops = stops_by_symbol.get(sym, [])
            covered = sum(o.qty for o in stops)
            if stops and abs(covered - qty) < 1e-9:
                if self.db.get_stop_level(self.agent, sym) is None and stops[0].stop_price:
                    self.db.set_stop_level(self.agent, sym, stops[0].stop_price, None)
                continue
            level = self.db.get_stop_level(self.agent, sym)
            atr = None
            if level is None:
                snap = snapshot.get(sym)
                atr = snap.atr_for(self.cfg.risk.stop_loss_atr_timeframe) if snap else None
                if not atr:
                    notes.append(f"UWAGA: {sym} bez stop-lossa i bez ATR — nie mogę go wyliczyć")
                    continue
                level = pos.avg_entry_price - self.cfg.risk.stop_loss_atr_mult * atr
            notes.append(f"{sym}: stop-loss pokrywa {covered:g}/{qty} akcji — ustawiam stop {round_price(level):.2f} "
                         f"na {qty} akcji")
            if not self.dry_run:
                self._cancel_stops(sym)
                self.place_stop(run_id, sym, qty, level, atr)

        for n in notes:
            log.info("[%s] rekoncyliacja: %s", self.agent, n)
        return notes
