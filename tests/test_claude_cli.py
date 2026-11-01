"""ClaudeCLI: prawdziwe uruchomienie podprocesu, ale zamiast `claude` działa fałszywy program,
który zapisuje argumenty / stdin / środowisko i odpowiada w formacie `claude -p --output-format json`
(format sprawdzony na prawdziwym Claude Code)."""

import json

import pytest

from trader.agent.claude_cli import ClaudeCLI, parse_cli_output
from trader.agent.schemas import DECISION_JSON_SCHEMA


def make_cli(cfg, exe, tmp_path, **over):
    cfg.llm.cli_path = str(exe)
    for k, v in over.items():
        setattr(cfg.llm, k, v)
    return ClaudeCLI(cfg.llm, tmp_path / "workdir", sleep=lambda s: None)


def test_success_parsing_and_command_line(cfg, fake_claude, tmp_path):
    exe, calls = fake_claude
    cli = make_cli(cfg, exe, tmp_path)
    res = cli.ask("Ticker: AAPL\nkontekst", DECISION_JSON_SCHEMA)
    assert res.ok and res.structured["ticker"] == "AAPL" and res.structured["action"] == "HOLD"
    assert res.model_served == cfg.llm.model and res.cost_equiv_usd == pytest.approx(0.0123)
    assert res.input_tokens == 3 and res.cache_creation_tokens == 2700 and res.output_tokens == 150
    assert res.duration_ms == 1234 and res.session_id == "u-1"

    c = calls()[0]
    a = c["args"]
    assert a[0] == "-p"
    assert a[a.index("--output-format") + 1] == "json"
    assert a[a.index("--model") + 1] == cfg.llm.model
    assert a[a.index("--effort") + 1] == cfg.llm.effort
    assert a[a.index("--system-prompt") + 1] == cfg.llm.system_prompt
    assert json.loads(a[a.index("--json-schema") + 1]) == DECISION_JSON_SCHEMA
    assert a[a.index("--tools") + 1] == ""                        # model bez narzędzi
    assert "--no-session-persistence" in a and "--safe-mode" in a and "--strict-mcp-config" in a
    assert c["stdin"] == "Ticker: AAPL\nkontekst"                 # prompt przez stdin
    assert c["cwd"] == str(tmp_path / "workdir")                  # pusty katalog roboczy
    assert not c["has_api_key"] and not c["has_auth_token"]       # NIGDY płatne API


def test_effort_override(cfg, fake_claude, tmp_path):
    exe, calls = fake_claude
    make_cli(cfg, exe, tmp_path).ask("Ticker: X", DECISION_JSON_SCHEMA, effort="low")
    a = calls()[0]["args"]
    assert a[a.index("--effort") + 1] == "low"


def test_model_error_is_reported(cfg, fake_claude, tmp_path, monkeypatch):
    exe, calls = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "404")
    res = make_cli(cfg, exe, tmp_path).ask("Ticker: X", DECISION_JSON_SCHEMA)
    assert not res.ok and res.error_status == 404 and "selected model" in res.error
    assert len(calls()) == 1                                       # 404 nie jest ponawiany


def test_rate_limit_is_retried(cfg, fake_claude, tmp_path, monkeypatch):
    exe, calls = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "429_then_ok")
    res = make_cli(cfg, exe, tmp_path, max_retries=1).ask("Ticker: X", DECISION_JSON_SCHEMA)
    assert res.ok and len(calls()) == 2


def test_garbage_output_is_error(cfg, fake_claude, tmp_path, monkeypatch):
    exe, _ = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    res = make_cli(cfg, exe, tmp_path, max_retries=0).ask("Ticker: X", DECISION_JSON_SCHEMA)
    assert not res.ok and "nieczytelny wynik CLI" in res.error and "Segmentation fault" in res.error


def test_timeout_is_error(cfg, fake_claude, tmp_path, monkeypatch):
    exe, _ = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    res = make_cli(cfg, exe, tmp_path, timeout_s=1, max_retries=0).ask("Ticker: X", DECISION_JSON_SCHEMA)
    assert not res.ok and "timeout" in res.error


def test_missing_binary(cfg, tmp_path):
    cfg.llm.cli_path = str(tmp_path / "nie-ma-claude")
    res = ClaudeCLI(cfg.llm, tmp_path / "w").ask("Ticker: X", DECISION_JSON_SCHEMA)
    assert not res.ok and "nie znaleziono programu" in res.error


def test_version_and_auth(cfg, fake_claude, tmp_path):
    exe, _ = fake_claude
    cli = make_cli(cfg, exe, tmp_path)
    assert cli.version() == "9.9.9 (Claude Code)"
    assert cli.auth_status()["loggedIn"] is True


def test_parse_real_cli_success_payload():
    """Fragment prawdziwego wyniku `claude -p --output-format json --json-schema ...`."""
    raw = json.dumps({
        "type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
        "stop_reason": "tool_use", "num_turns": 2, "duration_ms": 1497, "total_cost_usd": 0.0006206,
        "result": "{\"ticker\":\"TEST\",\"action\":\"HOLD\",\"size_pct\":0,\"confidence\":0.5,\"reasoning\":\"test\"}",
        "structured_output": {"ticker": "TEST", "action": "HOLD", "size_pct": 0, "confidence": 0.5,
                              "reasoning": "test"},
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 2722, "cache_read_input_tokens": 0,
                  "output_tokens": 152, "output_tokens_details": {"thinking_tokens": 0}},
        "modelUsage": {"claude-haiku-5-5": {"inputTokens": 2, "outputTokens": 152, "costUSD": 0.0006206}},
        "uuid": "f34cadab-6d1f-4088-9a46-aa8ca5033d93",
    })
    res = parse_cli_output(raw, 0)
    assert res.ok and res.structured["action"] == "HOLD" and res.model_served == "claude-haiku-5-5"
    assert res.cost_equiv_usd == pytest.approx(0.0006206) and res.output_tokens == 152


def test_parse_real_cli_error_payload():
    raw = json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 404,
                      "result": "There's an issue with the selected model (claude-nonexistent-9).",
                      "terminal_reason": "api_error", "total_cost_usd": 0, "usage": {}, "modelUsage": {}})
    res = parse_cli_output(raw, 1)
    assert not res.ok and res.error_status == 404 and "selected model" in res.error and not res.retryable
