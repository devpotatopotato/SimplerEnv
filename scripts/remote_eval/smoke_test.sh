#!/usr/bin/env bash
# Verify headless rendering and the full protocol with the stationary mock policy.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_EVAL_HOME="${REMOTE_EVAL_HOME:-${REPO_ROOT}/.remote_eval}"
EVAL_PYTHON="${REMOTE_EVAL_HOME}/envs/simpler/bin/python"
EVAL_GPU="${EVAL_GPU:-1}"
POLICY_PORT="${POLICY_PORT:-8000}"
RUN_TAG="smoke-$(date -u +%Y%m%dT%H%M%SZ)"
RUNTIME_DIR="${REPO_ROOT}/results/remote_eval_smoke/_launcher/${RUN_TAG}"
RUNTIME_CONFIG="${RUNTIME_DIR}/smoke.json"
SERVER_PID=""

[[ -x "$EVAL_PYTHON" ]] || { echo "Run setup.sh --models eval first" >&2; exit 1; }
mkdir -p "$RUNTIME_DIR"

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$EVAL_PYTHON" - "${REPO_ROOT}/configs/remote_eval/widowx_smoke.json" "$RUNTIME_CONFIG" "$EVAL_GPU" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
config["env_kwargs"]["renderer_kwargs"]["device"] = f"cuda:{sys.argv[3]}"
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
PY

# Match run.sh: keep all GPUs visible and select the physical renderer index.
env -u CUDA_VISIBLE_DEVICES "$EVAL_PYTHON" - "$EVAL_GPU" <<'PY'
import simpler_env
import sys

env = simpler_env.make(
    "widowx_spoon_on_towel",
    max_episode_steps=2,
    renderer_kwargs={"offscreen_only": True, "device": f"cuda:{sys.argv[1]}"},
)
obs, _ = env.reset(seed=0)
image = obs["image"]["3rd_view_camera"]["rgb"]
print("Headless rendering OK:", image.shape, image.dtype)
env.close()
PY

"$EVAL_PYTHON" -u -m policy_servers.mock_server \
    --host 127.0.0.1 --port "$POLICY_PORT" >"${RUNTIME_DIR}/mock_server.log" 2>&1 &
SERVER_PID=$!

for _ in {1..30}; do
    if curl --max-time 2 --fail --silent "http://127.0.0.1:${POLICY_PORT}/healthz" >/dev/null; then
        break
    fi
    sleep 1
done
curl --max-time 2 --fail --silent "http://127.0.0.1:${POLICY_PORT}/healthz" >/dev/null || {
    tail -n 80 "${RUNTIME_DIR}/mock_server.log" >&2
    exit 1
}

env -u CUDA_VISIBLE_DEVICES "$EVAL_PYTHON" -u -m simpler_env.evaluation.remote_evaluator \
    --config "$RUNTIME_CONFIG" \
    --server-url "http://127.0.0.1:${POLICY_PORT}" \
    --run-name "$RUN_TAG"

echo "Smoke test passed. Output: ${REPO_ROOT}/results/remote_eval_smoke/${RUN_TAG}"
