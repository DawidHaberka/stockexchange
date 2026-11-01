#!/usr/bin/env bash
# Wrapper dla crona: katalog repo, venv, PATH z Claude Code, blokada przed nakładaniem się przebiegów, log.
# Użycie: deploy/run.sh --run | deploy/run.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."
# cron ma okrojony PATH — dodaj typowe lokalizacje programu `claude`
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:/usr/local/bin:$PATH"
mkdir -p runtime/logs
exec /usr/bin/flock -n runtime/.run.lock .venv/bin/python main.py "$@" >> runtime/logs/cron.log 2>&1
