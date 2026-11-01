"""Ładowanie i walidacja konfiguracji (config.yaml) oraz sekretów (.env)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NY_TZ_NAME = "America/New_York"

FREQUENCY_MINUTES = {"1h": 60, "30min": 30, "15min": 15}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_Strict):
    name: str
    frequency: Literal["1h", "30min", "15min"] = "1h"
    first_run_after_open_min: int = Field(30, ge=0)
    last_run_before_close_min: int = Field(30, ge=0)

    @property
    def frequency_minutes(self) -> int:
        return FREQUENCY_MINUTES[self.frequency]


class AccountsCfg(_Strict):
    llm: str = "ALPACA_LLM"
    sma: str = "ALPACA_SMA"


class DataCfg(_Strict):
    feed: Literal["iex", "sip"] = "iex"
    hourly_lookback_days: int = Field(30, ge=10)
    daily_lookback_days: int = Field(200, ge=80)
    news_lookback_hours: int = Field(24, ge=1)
    max_news_per_ticker: int = Field(5, ge=0, le=50)
    hourly_closes_in_context: int = Field(8, ge=1, le=50)


class TriggerCfg(_Strict):
    on_first_evaluation: bool = True
    on_new_news: bool = True
    price_move_atr_mult: float = Field(1.0, gt=0)
    price_reference: Literal["last_run", "last_llm_call"] = "last_run"
    near_stop_atr_mult: float = Field(1.0, ge=0)


class LLMCfg(_Strict):
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(16000, ge=256)
    timeout_s: float = Field(300, gt=0)
    max_retries: int = Field(2, ge=0)
    refusal_fallback: bool = True
    system_prompt: str

    @field_validator("system_prompt")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt nie może być pusty")
        return v

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]


class Pricing(_Strict):
    input: float
    output: float
    cache_write: float
    cache_read: float


class SMACfg(_Strict):
    fast: int = Field(20, ge=2)
    slow: int = Field(50, ge=3)
    target_size_pct: float = Field(10.0, gt=0, le=100)

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> "SMACfg":
        if self.fast >= self.slow:
            raise ValueError("sma.fast musi być mniejsze niż sma.slow")
        return self


class RiskCfg(_Strict):
    max_position_pct: float = Field(10.0, gt=0, le=100)
    max_open_positions: int = Field(5, ge=1)
    stop_loss_atr_mult: float = Field(2.0, gt=0)
    stop_loss_atr_timeframe: Literal["1d", "1h"] = "1d"
    daily_loss_limit_pct: float = Field(3.0, gt=0)
    cooldown_minutes: int = Field(120, ge=0)
    cash_buffer_pct: float = Field(1.0, ge=0, lt=100)


class ExecutionCfg(_Strict):
    orders_enabled: bool = False
    limit_offset_bps: float = Field(10, ge=0, le=500)
    fill_timeout_s: float = Field(120, ge=0)
    poll_interval_s: float = Field(5, gt=0)


class StorageCfg(_Strict):
    db_path: str = "runtime/trader.db"
    reports_dir: str = "runtime/reports"
    log_dir: str = "runtime/logs"


class ReportCfg(_Strict):
    risk_free_rate_annual: float = 0.0


class AppConfig(_Strict):
    experiment: ExperimentCfg
    watchlist: list[str]
    benchmark_symbol: str = "SPY"
    accounts: AccountsCfg = AccountsCfg()
    data: DataCfg = DataCfg()
    triggers: TriggerCfg = TriggerCfg()
    llm: LLMCfg
    pricing_usd_per_mtok: dict[str, Pricing] = {}
    sma: SMACfg = SMACfg()
    risk: RiskCfg = RiskCfg()
    execution: ExecutionCfg = ExecutionCfg()
    storage: StorageCfg = StorageCfg()
    report: ReportCfg = ReportCfg()

    @field_validator("watchlist")
    @classmethod
    def _normalize_watchlist(cls, v: list[str]) -> list[str]:
        out = []
        for s in v:
            s = s.strip().upper()
            if not s or not s.replace(".", "").isalnum():
                raise ValueError(f"Niepoprawny ticker na watchliście: {s!r}")
            if s not in out:
                out.append(s)
        if not out:
            raise ValueError("watchlist nie może być pusta")
        return out

    @model_validator(mode="after")
    def _benchmark_upper(self) -> "AppConfig":
        self.benchmark_symbol = self.benchmark_symbol.strip().upper()
        return self

    # --- wersjonowanie wariantu eksperymentu -------------------------------
    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def variant(self) -> dict:
        """Pola identyfikujące wariant eksperymentu — zapisywane przy każdej decyzji."""
        return {
            "model": self.llm.model,
            "effort": self.llm.effort,
            "frequency": self.experiment.frequency,
            "prompt_hash": self.llm.prompt_hash,
            "config_hash": self.config_hash,
        }

    def pricing_for(self, model: str) -> Pricing | None:
        return self.pricing_usd_per_mtok.get(model)


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str
    base_url: str


@dataclass(frozen=True)
class Secrets:
    alpaca_base_url: str
    accounts: dict[str, AlpacaCredentials]
    anthropic_api_key: str | None


class ConfigError(RuntimeError):
    pass


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Brak pliku konfiguracji: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


def load_secrets(cfg: AppConfig, env_file: str | Path | None = ".env") -> Secrets:
    """Czyta klucze z .env (lub środowiska). Nie loguje wartości kluczy."""
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    base_url = os.environ.get("ALPACA_BASE_URL", "").strip()
    if not base_url:
        raise ConfigError("Brak ALPACA_BASE_URL w .env")

    accounts: dict[str, AlpacaCredentials] = {}
    for agent_name, prefix in (("llm", cfg.accounts.llm), ("sma", cfg.accounts.sma)):
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        secret = os.environ.get(f"{prefix}_SECRET_KEY", "").strip()
        if key and secret:
            accounts[agent_name] = AlpacaCredentials(key, secret, base_url)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    return Secrets(alpaca_base_url=base_url, accounts=accounts, anthropic_api_key=anthropic_key)
