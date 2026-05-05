#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"

cd "$ROOT_DIR"
python -m optimal_samples_system.mobile_api --host "$HOST" --port "$PORT"
