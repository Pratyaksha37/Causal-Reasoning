#!/usr/bin/env bash
# One project entry point; all commands are available through `./run.sh --help`.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Create Python 3.11 .venv, then install: pip install -e '.[dev]'" >&2
  exit 2
fi
if [ "${1:-}" = "test" ] || [ "${1:-}" = "tests" ]; then
  shift
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  exec "$PYTHON" -m pytest "$@"
fi
if [ "${1:-}" = "--from-saved-predictions" ] || [ "${1:-}" = "--max-pairs" ] || [ "${1:-}" = "--max-eval-pairs" ]; then
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  echo "Deprecated bare extraction flags: use ./run.sh extract $*" >&2
  exec "$PYTHON" run_pipeline.py "$@"
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"
if [ "$#" -eq 0 ]; then
  if [ -f "$PROJECT_ROOT/data/brfss/raw/LLCP2015XPT.zip" ]; then
    exec "$PYTHON" -m src run --data data/brfss/raw/LLCP2015XPT.zip
  fi
  exec "$PYTHON" -m src --help
fi
exec "$PYTHON" -m src "$@"
