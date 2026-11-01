"""Wywołanie Claude przez Claude Code CLI w trybie headless (`claude -p`) — bez płatnego API.

Claude Code działa na koncie z subskrypcją Claude (Pro/Max), zalogowanym na maszynie,
na której chodzi cron. Każde wywołanie:
  - zastępuje domyślny prompt systemowy naszym (`--system-prompt`),
  - wyłącza wszystkie narzędzia (`--tools ""`): model tylko ocenia podany kontekst,
  - wymusza format odpowiedzi schematem JSON (`--json-schema`) -> pole `structured_output`,
  - nie zapisuje sesji (`--no-session-persistence`),
  - dostaje prompt przez stdin, a wynik zwraca jako jeden obiekt JSON (`--output-format json`).

Zmienne ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN są usuwane ze środowiska procesu, żeby
Claude Code na pewno korzystał z subskrypcji, a nie z płatnego klucza API.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import LLMCfg

log = logging.getLogger(__name__)

STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


@dataclass
class CLIResult:
    ok: bool
    structured: dict | None = None
    text: str | None = None
    error: str | None = None
    error_status: int | None = None
    model_served: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_equiv_usd: float = 0.0       # równoważnik ceny API (z CLI) — NIE jest pobierany przy subskrypcji
    duration_ms: int | None = None
    stop_reason: str | None = None
    session_id: str | None = None

    @property
    def retryable(self) -> bool:
        return (not self.ok) and (self.error_status in RETRYABLE_STATUS or self.error_status is None)


def parse_cli_output(stdout: str, returncode: int, stderr: str = "") -> CLIResult:
    """Parsuje wynik `claude -p --output-format json`."""
    try:
        data = json.loads(stdout) if stdout and stdout.strip() else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        tail = (stderr or stdout or "").strip()[-400:]
        return CLIResult(False, error=f"nieczytelny wynik CLI (kod {returncode}): {tail or 'brak wyjścia'}")

    usage = data.get("usage") or {}
    model_usage = data.get("modelUsage") or {}
    served = None
    if model_usage:
        served = max(model_usage.items(), key=lambda kv: (kv[1] or {}).get("outputTokens", 0))[0]
    res = CLIResult(
        ok=not data.get("is_error", False) and returncode == 0 and data.get("subtype", "success") == "success",
        structured=data.get("structured_output") if isinstance(data.get("structured_output"), dict) else None,
        text=data.get("result") if isinstance(data.get("result"), str) else None,
        error_status=data.get("api_error_status"),
        model_served=served,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cost_equiv_usd=float(data.get("total_cost_usd") or 0.0),
        duration_ms=data.get("duration_ms"),
        stop_reason=data.get("stop_reason"),
        session_id=data.get("uuid") or data.get("session_id"),
    )
    if not res.ok:
        detail = res.text or data.get("subtype") or data.get("terminal_reason") or "błąd"
        res.error = f"CLI: {detail}" + (f" (HTTP {res.error_status})" if res.error_status else "")
    return res


class ClaudeCLI:
    def __init__(self, cfg: LLMCfg, workdir: str | Path,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.runner = runner
        self.sleep = sleep

    # ------------------------------------------------------------------ helpers
    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        for k in STRIPPED_ENV:
            env.pop(k, None)
        return env

    def build_cmd(self, schema: dict, model: str | None = None, effort: str | None = None,
                  system_prompt: str | None = None) -> list[str]:
        return [
            self.cfg.cli_path, "-p",
            "--output-format", "json",
            "--model", model or self.cfg.model,
            "--effort", effort or self.cfg.effort,
            "--system-prompt", system_prompt if system_prompt is not None else self.cfg.system_prompt,
            "--json-schema", json.dumps(schema, separators=(",", ":")),
            "--tools", "",
            "--no-session-persistence",
            *self.cfg.cli_extra_args,
        ]

    def _run(self, args: list[str], stdin: str | None, timeout: float) -> subprocess.CompletedProcess:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.runner(args, input=stdin, capture_output=True, text=True, timeout=timeout,
                           cwd=str(self.workdir), env=self.env())

    # ------------------------------------------------------------------ API
    def ask(self, prompt: str, schema: dict, model: str | None = None, effort: str | None = None,
            system_prompt: str | None = None) -> CLIResult:
        cmd = self.build_cmd(schema, model, effort, system_prompt)
        attempts = self.cfg.max_retries + 1
        res = CLIResult(False, error="nie wykonano")
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                proc = self._run(cmd, prompt, self.cfg.timeout_s)
            except FileNotFoundError:
                return CLIResult(False, error=f"nie znaleziono programu '{self.cfg.cli_path}' — zainstaluj Claude Code")
            except subprocess.TimeoutExpired:
                res = CLIResult(False, error=f"timeout {self.cfg.timeout_s:.0f}s")
            else:
                res = parse_cli_output(proc.stdout, proc.returncode, proc.stderr)
                if res.duration_ms is None:
                    res.duration_ms = int((time.monotonic() - t0) * 1000)
            if res.ok or not res.retryable or attempt == attempts:
                return res
            log.warning("Claude CLI: %s — ponawiam (%d/%d)", res.error, attempt, attempts - 1)
            self.sleep(min(30.0, 5.0 * attempt))
        return res

    def version(self) -> str | None:
        try:
            p = self._run([self.cfg.cli_path, "--version"], None, 30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    def auth_status(self) -> dict | None:
        try:
            p = self._run([self.cfg.cli_path, "auth", "status", "--json"], None, 30)
            return json.loads(p.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None
