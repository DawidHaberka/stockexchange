"""Ściśle ustrukturyzowana decyzja agenta + walidacja odpowiedzi modelu.

Zasada: każda odpowiedź, która nie przejdzie walidacji pydantic, staje się HOLD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    ticker: str = Field(min_length=1, max_length=10)
    action: Action
    size_pct: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1, max_length=4000)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("reasoning")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("reasoning nie może być pusty")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Decision":
        if self.action in (Action.BUY, Action.SELL) and self.size_pct <= 0:
            raise ValueError(f"{self.action.value} wymaga size_pct > 0")
        return self


# JSON Schema przekazywany do structured outputs (output_config.format).
# Zakresy liczbowe sprawdza pydantic (druga warstwa walidacji).
DECISION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "size_pct": {"type": "number"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["ticker", "action", "size_pct", "confidence", "reasoning"],
    "additionalProperties": False,
}


def hold(ticker: str, reason: str) -> Decision:
    return Decision(ticker=ticker, action=Action.HOLD, size_pct=0, confidence=0, reasoning=reason)


def parse_decision(raw: str | None, expected_ticker: str) -> tuple[Decision, str | None]:
    """Parsuje surowy tekst odpowiedzi. Zwraca (decyzja, błąd). Przy błędzie decyzja = HOLD."""
    expected = expected_ticker.strip().upper()
    if raw is None or not raw.strip():
        err = "pusta odpowiedź modelu"
        return hold(expected, f"HOLD (niepoprawna odpowiedź: {err})"), err
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        err = f"niepoprawny JSON: {exc.msg} (poz. {exc.pos})"
        return hold(expected, f"HOLD (niepoprawna odpowiedź: {err})"), err
    if not isinstance(data, dict):
        err = "JSON nie jest obiektem"
        return hold(expected, f"HOLD (niepoprawna odpowiedź: {err})"), err
    try:
        decision = Decision.model_validate(data)
    except ValidationError as exc:
        err = "walidacja: " + "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'obiekt'}: {e['msg']}" for e in exc.errors()
        )
        return hold(expected, f"HOLD (niepoprawna odpowiedź: {err})"), err
    if decision.ticker != expected:
        err = f"ticker w odpowiedzi ({decision.ticker}) != oczekiwany ({expected})"
        return hold(expected, f"HOLD (niepoprawna odpowiedź: {err})"), err
    if decision.action == Action.HOLD and decision.size_pct != 0:
        decision = decision.model_copy(update={"size_pct": 0.0})
    return decision, None


@dataclass
class AgentDecision:
    """Decyzja agenta wraz z metadanymi pochodzenia."""

    decision: Decision
    source: str  # llm | no_trigger | invalid | api_error | refusal | rule | halted | no_data
    triggers: list[str] = field(default_factory=list)
    llm_call_id: int | None = None
    validation_error: str | None = None

    @property
    def ticker(self) -> str:
        return self.decision.ticker

    @property
    def action(self) -> Action:
        return self.decision.action
