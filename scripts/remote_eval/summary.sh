#!/usr/bin/env bash
# Print aggregate and per-task metrics for a remote evaluation run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_EVAL_HOME="${REMOTE_EVAL_HOME:-${REPO_ROOT}/.remote_eval}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/remote_eval}"
EVAL_PYTHON="${REMOTE_EVAL_HOME}/envs/simpler/bin/python"

if [[ ! -x "$EVAL_PYTHON" ]]; then
    EVAL_PYTHON="$(command -v python3 || true)"
fi
[[ -n "$EVAL_PYTHON" ]] || { echo "Python 3 is required" >&2; exit 1; }

exec "$EVAL_PYTHON" "${SCRIPT_DIR}/summarize_results.py" --results-dir "$RESULTS_DIR" "$@"
