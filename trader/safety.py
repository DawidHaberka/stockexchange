"""Twarde bezpieczniki: ten program obsługuje WYŁĄCZNIE Alpaca Paper API.

Każdy klient tradingowy jest tworzony dopiero po przejściu `assert_paper_url`.
Nie istnieje żadna ścieżka kodu, która tworzyłaby klienta konta live.
"""

from __future__ import annotations

from urllib.parse import urlparse

PAPER_HOST = "paper-api.alpaca.markets"
PAPER_URL = f"https://{PAPER_HOST}"


class LiveTradingForbidden(RuntimeError):
    """Próba użycia endpointu innego niż Alpaca Paper API."""


def assert_paper_url(url: str | None) -> str:
    """Zwraca znormalizowany URL paper albo rzuca LiveTradingForbidden.

    Akceptowane: dokładnie https://paper-api.alpaca.markets (opcjonalnie z "/"
    lub "/v2" na końcu). Wszystko inne — w tym api.alpaca.markets (live),
    http://, inne hosty, porty, dane logowania w URL — zatrzymuje program.
    """
    if not url or not isinstance(url, str):
        raise LiveTradingForbidden("Brak ALPACA_BASE_URL — wymagany paper endpoint.")

    parsed = urlparse(url.strip())
    problems = []
    if parsed.scheme != "https":
        problems.append(f"schemat {parsed.scheme!r} zamiast 'https'")
    if (parsed.hostname or "").lower() != PAPER_HOST:
        problems.append(f"host {parsed.hostname!r} zamiast {PAPER_HOST!r}")
    if parsed.port not in (None, 443):
        problems.append(f"niedozwolony port {parsed.port}")
    if parsed.username or parsed.password:
        problems.append("dane logowania w URL")
    if parsed.path.rstrip("/") not in ("", "/v2"):
        problems.append(f"niedozwolona ścieżka {parsed.path!r}")
    if parsed.query or parsed.fragment:
        problems.append("parametry/fragment w URL")

    if problems:
        raise LiveTradingForbidden(
            "ODMOWA: ALPACA_BASE_URL nie jest endpointem Alpaca Paper "
            f"({'; '.join(problems)}). Ten program obsługuje wyłącznie paper trading "
            f"({PAPER_URL})."
        )
    return PAPER_URL
