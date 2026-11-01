"""Atrapa serwera Alpaca na poziomie HTTP (requests.Session.request) i Claude (httpx2.MockTransport).

Pozwala przetestować PRAWDZIWE adaptery (alpaca-py + nasze AlpacaBroker / AlpacaMarketData,
SDK anthropic + LLMAgent) bez sieci: payloady mają format z dokumentacji API.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import httpx2
import requests

from trader.broker import NY


def _resp(url: str, status: int, payload=None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r.url = url
    r.reason = "OK" if status < 400 else "Error"
    r._content = b"" if payload is None else json.dumps(payload).encode()
    r.headers["Content-Type"] = "application/json"
    return r


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class FakeAlpacaHTTP:
    """Jedno konto paper + dane rynkowe. Zlecenia limit realizują się przy pierwszym odczycie."""

    def __init__(self, now: datetime, md, equity: float = 10_000.0, account_number: str = "PA3TEST0001",
                 last_equity: float | None = None, is_open: bool = True):
        self.now = now
        self.md = md                     # tests.conftest.FakeMarketData — źródło świec
        self.cash = equity
        self.last_equity = equity if last_equity is None else last_equity
        self.account_number = account_number
        self.is_open = is_open
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.reject_symbols: set[str] = set()
        self.requests: list[tuple[str, str, dict]] = []

    # ------------------------------------------------------------------ router
    def request(self, method, url, **opts):
        params = opts.get("params") or {}
        body = opts.get("json") or {}
        self.requests.append((method, url, body or params))
        u = urlparse(url)
        host, path = u.hostname, u.path
        if host == "paper-api.alpaca.markets":
            return self._trading(method, url, path, params, body)
        if host == "data.alpaca.markets":
            return self._data(url, path, params)
        return _resp(url, 404, {"message": f"unexpected host {host}"})

    # ------------------------------------------------------------------ trading
    def _equity(self) -> float:
        return self.cash + sum(float(p["market_value"]) for p in self.positions.values())

    def _trading(self, method, url, path, params, body):
        if path == "/v2/account":
            eq = self._equity()
            return _resp(url, 200, {
                "id": "0d9a1a4c-0000-4000-8000-000000000001", "account_number": self.account_number,
                "status": "ACTIVE", "currency": "USD", "cash": f"{self.cash:.2f}", "equity": f"{eq:.2f}",
                "last_equity": f"{self.last_equity:.2f}", "buying_power": f"{self.cash * 2:.2f}",
                "non_marginable_buying_power": f"{self.cash:.2f}", "multiplier": "2",
                "trading_blocked": False, "account_blocked": False, "transfers_blocked": False,
                "pattern_day_trader": False, "shorting_enabled": True, "daytrade_count": 0,
                "created_at": "2026-10-01T00:00:00Z",
            })
        if path == "/v2/clock":
            n = self.now.astimezone(NY)
            return _resp(url, 200, {
                "timestamp": n.isoformat(), "is_open": self.is_open,
                "next_open": (n.replace(hour=9, minute=30) + timedelta(days=1)).isoformat(),
                "next_close": n.replace(hour=16, minute=0).isoformat(),
            })
        if path == "/v2/calendar":
            d = params.get("start")
            return _resp(url, 200, [{"date": d, "open": "09:30", "close": "16:00", "session_open": "0400",
                                     "session_close": "2000", "settlement_date": d}])
        if path.startswith("/v2/assets/"):
            sym = path.rsplit("/", 1)[1]
            return _resp(url, 200, {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, sym)), "class": "us_equity",
                                    "exchange": "NASDAQ", "symbol": sym, "name": sym, "status": "active",
                                    "tradable": True, "marginable": True, "shortable": True,
                                    "easy_to_borrow": True, "fractionable": True, "attributes": []})
        if path == "/v2/positions":
            return _resp(url, 200, list(self.positions.values()))
        if path == "/v2/orders" and method == "POST":
            return self._submit(url, body)
        if path == "/v2/orders" and method == "GET":
            status = params.get("status", "open")
            syms = set(str(params.get("symbols", "")).split(",")) - {""}
            out = [o for o in self.orders.values()
                   if (status != "open" or o["status"] in ("new", "accepted", "partially_filled"))
                   and (not syms or o["symbol"] in syms)]
            return _resp(url, 200, out)
        m = re.fullmatch(r"/v2/orders/([0-9a-f-]+)", path)
        if m and method == "GET":
            return _resp(url, 200, self._progress(self.orders[m.group(1)]))
        if m and method == "DELETE":
            o = self.orders[m.group(1)]
            if o["status"] in ("new", "accepted", "partially_filled"):
                o["status"] = "canceled"
                o["canceled_at"] = _ts(self.now)
            return _resp(url, 204)
        return _resp(url, 404, {"code": 40410000, "message": f"not found {method} {path}"})

    def _submit(self, url, body):
        sym = body["symbol"]
        if sym in self.reject_symbols:
            return _resp(url, 403, {"code": 40310000, "message": "insufficient buying power"})
        oid = str(uuid.uuid4())
        o = {
            "id": oid, "client_order_id": body.get("client_order_id") or oid,
            "created_at": _ts(self.now), "updated_at": _ts(self.now), "submitted_at": _ts(self.now),
            "filled_at": None, "expired_at": None, "canceled_at": None, "failed_at": None, "replaced_at": None,
            "replaced_by": None, "replaces": None, "asset_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, sym)),
            "symbol": sym, "asset_class": "us_equity", "notional": None, "qty": str(body["qty"]),
            "filled_qty": "0", "filled_avg_price": None, "order_class": "", "order_type": body["type"],
            "type": body["type"], "side": body["side"], "time_in_force": body["time_in_force"],
            "limit_price": str(body["limit_price"]) if body.get("limit_price") is not None else None,
            "stop_price": str(body["stop_price"]) if body.get("stop_price") is not None else None,
            "status": "accepted", "extended_hours": False, "legs": None, "trail_percent": None,
            "trail_price": None, "hwm": None, "position_intent": None,
        }
        self.orders[oid] = o
        return _resp(url, 200, o)

    def _progress(self, o):
        if o["type"] == "limit" and o["status"] == "accepted":
            qty, price = float(o["qty"]), float(o["limit_price"])
            o.update(status="filled", filled_qty=o["qty"], filled_avg_price=o["limit_price"],
                     filled_at=_ts(self.now), updated_at=_ts(self.now))
            sym = o["symbol"]
            p = self.positions.get(sym)
            if o["side"] == "buy":
                self.cash -= qty * price
                old_q = float(p["qty"]) if p else 0.0
                old_avg = float(p["avg_entry_price"]) if p else 0.0
                new_q = old_q + qty
                avg = (old_q * old_avg + qty * price) / new_q
                self.positions[sym] = {
                    "asset_id": o["asset_id"], "symbol": sym, "exchange": "NASDAQ", "asset_class": "us_equity",
                    "asset_marginable": True, "qty": f"{new_q:g}", "qty_available": "0",
                    "avg_entry_price": f"{avg:.4f}", "side": "long", "market_value": f"{new_q * price:.2f}",
                    "cost_basis": f"{new_q * avg:.2f}", "unrealized_pl": "0", "unrealized_plpc": "0",
                    "unrealized_intraday_pl": "0", "unrealized_intraday_plpc": "0",
                    "current_price": f"{price:.2f}", "lastday_price": f"{price:.2f}", "change_today": "0",
                }
            else:
                self.cash += qty * price
                new_q = float(p["qty"]) - qty
                if new_q <= 0:
                    del self.positions[sym]
                else:
                    p.update(qty=f"{new_q:g}", market_value=f"{new_q * price:.2f}")
        return o

    # ------------------------------------------------------------------ data
    def _bars_json(self, df):
        return [{"t": _ts(ts.to_pydatetime()), "o": float(r.open), "h": float(r.high), "l": float(r.low),
                 "c": float(r.close), "v": float(r.volume), "n": 100, "vw": float(r.close)}
                for ts, r in df.iterrows()]

    def _data(self, url, path, params):
        syms = [s for s in str(params.get("symbols", "")).split(",") if s]
        if path == "/v2/stocks/bars":
            src = self.md._hourly if str(params.get("timeframe")) == "1Hour" else self.md._daily
            return _resp(url, 200, {"bars": {s: self._bars_json(src[s]) for s in syms if s in src},
                                    "next_page_token": None})
        if path == "/v2/stocks/quotes/latest":
            q = self.md.latest_quotes(syms)
            return _resp(url, 200, {"quotes": {s: {"t": _ts(self.now), "ax": "V", "ap": q[s].ask, "as": 1,
                                                   "bx": "V", "bp": q[s].bid, "bs": 1, "c": ["R"], "z": "C"}
                                               for s in syms}})
        if path == "/v2/stocks/trades/latest":
            q = self.md.latest_quotes(syms)
            return _resp(url, 200, {"trades": {s: {"t": _ts(self.now), "x": "V", "p": q[s].last, "s": 100,
                                                   "c": ["@"], "i": 1, "z": "C"} for s in syms}})
        if path == "/v1beta1/news":
            sym = syms[0] if syms else ""
            items = [{"id": 1000 + i, "headline": f"{sym} headline {i}", "author": "test",
                      "created_at": _ts(self.now - timedelta(hours=2 + i)),
                      "updated_at": _ts(self.now - timedelta(hours=2 + i)), "summary": f"Summary {i}",
                      "content": "", "url": f"https://example.com/{sym}/{i}", "images": [], "symbols": [sym],
                      "source": "benzinga"} for i in range(2)]
            return _resp(url, 200, {"news": items, "next_page_token": None})
        return _resp(url, 404, {"message": f"unexpected data path {path}"})


def patch_broker(broker, fake: FakeAlpacaHTTP) -> None:
    broker.client._session.request = fake.request


def patch_market_data(md, fake: FakeAlpacaHTTP) -> None:
    md._stock._session.request = fake.request
    md._news._session.request = fake.request


# --------------------------------------------------------------------------- Claude (HTTP)
class FakeClaudeHTTP:
    """Odpowiada jak POST /v1/messages; zapisuje ciała żądań i nagłówki."""

    def __init__(self, decide):
        self.decide = decide            # (ticker, body) -> dict decyzji lub str surowej odpowiedzi
        self.bodies: list[dict] = []
        self.headers: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path.startswith("/v1/models/"):
            mid = request.url.path.rsplit("/", 1)[1]
            return httpx2.Response(200, json={"type": "model", "id": mid, "display_name": mid,
                                              "created_at": "2026-09-01T00:00:00Z"})
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(dict(request.headers))
        prompt = body["messages"][0]["content"]
        ticker = prompt.split("\n", 1)[0].replace("Ticker:", "").strip()
        out = self.decide(ticker, body)
        text = out if isinstance(out, str) else json.dumps(out)
        return httpx2.Response(200, json={
            "id": f"msg_{uuid.uuid4().hex[:12]}", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1500, "output_tokens": 700, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 900},
        }, headers={"request-id": "req_test"})

    def client(self):
        import anthropic

        return anthropic.Anthropic(api_key="test-key", base_url="https://api.anthropic.com", max_retries=0,
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(self.handler)))
