"""Walidacja JSON decyzji: niepoprawna odpowiedź = HOLD."""

import json

import pytest

from trader.agent.schemas import DECISION_JSON_SCHEMA, Action, Decision, parse_decision

VALID = {"ticker": "AAPL", "action": "BUY", "size_pct": 5, "confidence": 0.72, "reasoning": "Silny trend."}


def test_valid_decision():
    d, err = parse_decision(json.dumps(VALID), "AAPL")
    assert err is None
    assert d.action == Action.BUY and d.size_pct == 5 and d.confidence == 0.72


def test_hold_size_is_zeroed():
    d, err = parse_decision(json.dumps({**VALID, "action": "HOLD", "size_pct": 7}), "AAPL")
    assert err is None and d.action == Action.HOLD and d.size_pct == 0


@pytest.mark.parametrize("raw, fragment", [
    ("", "pusta"),
    ("not json", "JSON"),
    ("```json\n" + json.dumps(VALID) + "\n```", "JSON"),
    (json.dumps([VALID]), "obiektem"),
    (json.dumps({**VALID, "action": "STRONG_BUY"}), "action"),
    (json.dumps({**VALID, "action": "buy"}), "action"),
    (json.dumps({**VALID, "confidence": 1.5}), "confidence"),
    (json.dumps({**VALID, "confidence": -0.1}), "confidence"),
    (json.dumps({**VALID, "size_pct": 150}), "size_pct"),
    (json.dumps({**VALID, "size_pct": -1}), "size_pct"),
    (json.dumps({**VALID, "size_pct": 0}), "size_pct > 0"),
    (json.dumps({k: v for k, v in VALID.items() if k != "reasoning"}), "reasoning"),
    (json.dumps({**VALID, "reasoning": "   "}), "reasoning"),
    (json.dumps({**VALID, "leverage": 3}), "leverage"),
    (json.dumps({**VALID, "ticker": "MSFT"}), "ticker"),
    (json.dumps({**VALID, "size_pct": "dużo"}), "size_pct"),
])
def test_invalid_responses_become_hold(raw, fragment):
    d, err = parse_decision(raw, "AAPL")
    assert d.action == Action.HOLD
    assert d.ticker == "AAPL"
    assert d.size_pct == 0
    assert err is not None and fragment.lower() in err.lower()


def test_ticker_case_insensitive():
    d, err = parse_decision(json.dumps({**VALID, "ticker": "aapl"}), "AAPL")
    assert err is None and d.ticker == "AAPL"


def test_json_schema_matches_model_fields():
    assert set(DECISION_JSON_SCHEMA["required"]) == set(Decision.model_fields)
    assert DECISION_JSON_SCHEMA["additionalProperties"] is False
    assert DECISION_JSON_SCHEMA["properties"]["action"]["enum"] == ["BUY", "SELL", "HOLD"]
