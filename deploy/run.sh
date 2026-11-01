#!/usr/bin/env bash
# Wrapper dla crona: katalog repo, venv, blokada przed nakładaniem się przebiegów, log.
# Użycie: deploy/run.sh --run | deploy/run.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runtime/logs
exec /usr/bin/flock -n runtime/.run.lock .venv/bin/python main.py "$@" >> runtime/logs/cron.log 2>&1
