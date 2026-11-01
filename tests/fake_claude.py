"""Fałszywy program `claude` (Claude Code CLI) do testów — odpowiada w formacie `claude -p --output-format json`
(sprawdzonym na prawdziwym Claude Code), zapisuje argumenty, stdin, katalog roboczy i obecność kluczy API."""

import json
import stat
import sys

import pytest

FAKE = r'''#!{python}
import json, os, sys, time
args = sys.argv[1:]
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
log = os.environ["FAKE_CLAUDE_LOG"]
if args == ["--version"]:
    print("9.9.9 (Claude Code)"); sys.exit(0)
if args[:2] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": True, "authMethod": os.environ.get("FAKE_AUTH", "claude.ai")}})); sys.exit(0)
stdin = sys.stdin.read()
with open(log, "a") as fh:
    fh.write(json.dumps({{"args": args, "stdin": stdin, "cwd": os.getcwd(),
                         "has_api_key": "ANTHROPIC_API_KEY" in os.environ,
                         "has_auth_token": "ANTHROPIC_AUTH_TOKEN" in os.environ}}) + "\n")
calls = sum(1 for _ in open(log))
if mode == "sleep":
    time.sleep(5)
if mode == "garbage":
    print("Segmentation fault"); sys.exit(139)
if mode == "404" or (mode == "429_then_ok" and calls == 1) or mode == "429":
    status = 404 if mode == "404" else 429
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": True, "api_error_status": status,
                      "result": "There's an issue with the selected model" if status == 404 else "Usage limit reached",
                      "total_cost_usd": 0, "usage": {{}}, "modelUsage": {{}}, "terminal_reason": "api_error"}}))
    sys.exit(1)
ticker = stdin.split("\n", 1)[0].replace("Ticker:", "").strip()
buy = ticker in os.environ.get("FAKE_CLAUDE_BUY", "").split(",")
out = {{"ticker": ticker, "action": "BUY" if buy else "HOLD", "size_pct": 5 if buy else 0,
       "confidence": 0.7 if buy else 0.4, "reasoning": "test " + ticker}}
model = args[args.index("--model") + 1]
print(json.dumps({{
    "type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
    "result": json.dumps(out), "structured_output": out, "stop_reason": "tool_use", "num_turns": 2,
    "duration_ms": 1234, "total_cost_usd": 0.0123, "uuid": "u-1", "session_id": "s-1",
    "usage": {{"input_tokens": 3, "cache_creation_input_tokens": 2700, "cache_read_input_tokens": 0,
               "output_tokens": 150}},
    "modelUsage": {{model: {{"inputTokens": 3, "outputTokens": 150, "costUSD": 0.0123}}}},
}}))
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-passed")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-be-passed")

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return exe, calls
