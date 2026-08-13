#!/usr/bin/env bash
# Sequentially evaluate adapted π0.5, VPP, and Cosmos 3 Edge policies.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_EVAL_HOME="${REMOTE_EVAL_HOME:-${REPO_ROOT}/.remote_eval}"
SOURCE_ROOT="${SOURCE_ROOT:-${REMOTE_EVAL_HOME}/sources}"
ENV_ROOT="${ENV_ROOT:-${REMOTE_EVAL_HOME}/envs}"

ENV_FILE="${SCRIPT_DIR}/models.env"
BASE_CONFIG="${REPO_ROOT}/configs/remote_eval/widowx_bridge.json"
SELECTED_MODELS="pi05,vpp,cosmos3"
POLICY_GPU="${POLICY_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
POLICY_PORT="${POLICY_PORT:-8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/remote_eval}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN=0
CONTINUE_ON_ERROR=0
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage: run.sh [options]

Runs one model at a time on POLICY_GPU and SimplerEnv on EVAL_GPU.

Options:
  --env-file PATH       Checkpoint/config variables (default: models.env)
  --config PATH         Evaluation JSON (default: widowx_bridge.json)
  --models LIST         Comma-separated pi05,vpp,cosmos3 (default: all three)
  --run-tag NAME        Shared result/log suffix
  --dry-run             Validate and print commands without starting models
  --continue-on-error   Continue with the next model after a failed evaluation
  -h, --help            Show this help

Environment overrides:
  POLICY_GPU=0, EVAL_GPU=1, POLICY_PORT=8000
  SERVER_START_TIMEOUT=1800, RESULTS_DIR=<repo>/results/remote_eval

Required checkpoint values are documented in models.env.example. Only adapted
canonical checkpoints are accepted by this fair-comparison launcher.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

while (($#)); do
    case "$1" in
        --env-file)
            (($# >= 2)) || die "--env-file requires a path"
            ENV_FILE="$2"
            shift 2
            ;;
        --config)
            (($# >= 2)) || die "--config requires a path"
            BASE_CONFIG="$2"
            shift 2
            ;;
        --models)
            (($# >= 2)) || die "--models requires a list"
            SELECTED_MODELS="$2"
            shift 2
            ;;
        --run-tag)
            (($# >= 2)) || die "--run-tag requires a value"
            RUN_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE; copy models.env.example and fill its paths"
# shellcheck source=/dev/null
source "$ENV_FILE"

EVAL_PYTHON="${ENV_ROOT}/simpler/bin/python"
PI05_ROOT="${SOURCE_ROOT}/openpi"
VPP_ROOT="${SOURCE_ROOT}/video-prediction-policy"
COSMOS_ROOT="${SOURCE_ROOT}/cosmos-framework"
PI05_PYTHON="${PI05_ROOT}/.venv/bin/python"
VPP_PYTHON="${ENV_ROOT}/vpp/bin/python"
COSMOS_PYTHON="${COSMOS_ROOT}/.venv/bin/python"

[[ -x "$EVAL_PYTHON" ]] || die "evaluator environment is missing; run setup.sh"
[[ -f "$BASE_CONFIG" ]] || die "evaluation config does not exist: $BASE_CONFIG"
[[ "$POLICY_GPU" != "$EVAL_GPU" ]] || die "POLICY_GPU and EVAL_GPU must be different"
command -v curl >/dev/null || die "curl is required for policy-server health checks"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
((POLICY_GPU >= 0 && POLICY_GPU < GPU_COUNT)) || die "POLICY_GPU=$POLICY_GPU is invalid for $GPU_COUNT GPUs"
((EVAL_GPU >= 0 && EVAL_GPU < GPU_COUNT)) || die "EVAL_GPU=$EVAL_GPU is invalid for $GPU_COUNT GPUs"

MODELS=()
for model in ${SELECTED_MODELS//,/ }; do
    case "$model" in
        pi05|vpp|cosmos3) MODELS+=("$model") ;;
        *) die "unknown model: $model" ;;
    esac
done
((${#MODELS[@]} > 0)) || die "no models selected"

require_value() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "$name is missing in $ENV_FILE"
    [[ "${!name}" != *"/path/to/"* && "${!name}" != your_* ]] || die "$name still contains a placeholder"
}

require_path() {
    local name="$1"
    require_value "$name"
    [[ -e "${!name}" ]] || die "$name does not exist: ${!name}"
}

validate_model() {
    case "$1" in
        pi05)
            [[ -x "$PI05_PYTHON" ]] || die "OpenPI environment is missing; run setup.sh --models pi05"
            require_value PI05_CONFIG_NAME
            require_value PI05_CHECKPOINT
            if [[ "$PI05_CHECKPOINT" != *"://"* ]]; then
                [[ -e "$PI05_CHECKPOINT" ]] || die "PI05_CHECKPOINT does not exist: $PI05_CHECKPOINT"
            fi
            ;;
        vpp)
            [[ -x "$VPP_PYTHON" ]] || die "VPP environment is missing; run setup.sh --models vpp"
            [[ -d "$VPP_ROOT" ]] || die "VPP source is missing: $VPP_ROOT"
            require_path VPP_CONFIG
            require_path VPP_VIDEO_MODEL_PATH
            require_path VPP_TEXT_ENCODER_PATH
            require_path VPP_ACTION_CHECKPOINT
            ;;
        cosmos3)
            [[ -x "$COSMOS_PYTHON" ]] || die "Cosmos environment is missing; run setup.sh --models cosmos3"
            [[ -d "$COSMOS_ROOT" ]] || die "Cosmos source is missing: $COSMOS_ROOT"
            require_path COSMOS_CHECKPOINT
            require_value COSMOS_DOMAIN_NAME
            ;;
    esac
}

for model in "${MODELS[@]}"; do
    validate_model "$model"
done

LOG_DIR="${RESULTS_DIR}/_launcher/${RUN_TAG}"
RUNTIME_CONFIG="${LOG_DIR}/evaluation_gpu${EVAL_GPU}.json"
mkdir -p "$LOG_DIR"

# Snapshot the evaluation config with the selected physical renderer GPU.
"$EVAL_PYTHON" - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$EVAL_GPU" "$RESULTS_DIR" <<'PY'
import json
import pathlib
import sys

source, destination, gpu, output_dir = sys.argv[1:]
with open(source, encoding="utf-8") as stream:
    config = json.load(stream)
config.setdefault("env_kwargs", {}).setdefault("renderer_kwargs", {})["offscreen_only"] = True
config["env_kwargs"]["renderer_kwargs"]["device"] = f"cuda:{gpu}"
config["output_dir"] = str(pathlib.Path(output_dir).resolve())
pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

cleanup_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        note "stopping policy server PID $SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

port_is_open() {
    "$EVAL_PYTHON" - "$POLICY_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_for_server() {
    local pid="$1"
    local log_file="$2"
    local elapsed=0
    while ((elapsed < SERVER_START_TIMEOUT)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '\nPolicy server exited during startup. Last log lines:\n' >&2
            tail -n 80 "$log_file" >&2 || true
            return 1
        fi
        if curl --max-time 2 --fail --silent "http://127.0.0.1:${POLICY_PORT}/healthz" >/dev/null; then
            return 0
        fi
        if ((elapsed % 15 == 0)); then
            printf '[%s] waiting for policy server (%ss/%ss)\n' \
                "$(date -u +%H:%M:%S)" "$elapsed" "$SERVER_START_TIMEOUT"
        fi
        sleep 1
        ((elapsed += 1))
    done
    printf 'Policy server did not become healthy within %ss. Last log lines:\n' "$SERVER_START_TIMEOUT" >&2
    tail -n 80 "$log_file" >&2 || true
    return 1
}

build_server_command() {
    local model="$1"
    SERVER_COMMAND=()
    case "$model" in
        pi05)
            SERVER_COMMAND=(
                "$PI05_PYTHON" -u -m policy_servers.pi05.server
                --config-name "$PI05_CONFIG_NAME"
                --checkpoint "$PI05_CHECKPOINT"
                --device cuda:0
                --output-mode canonical
                --confirm-canonical-adapted
                --adaptation-dataset bridge_widowx
                --adaptation-method "${PI05_ADAPTATION_METHOD:-action_head_or_adapter}"
                --host 127.0.0.1 --port "$POLICY_PORT"
            )
            ;;
        vpp)
            SERVER_COMMAND=(
                "$VPP_PYTHON" -u -m policy_servers.vpp.server
                --vpp-root "$VPP_ROOT"
                --config "$VPP_CONFIG"
                --video-model-path "$VPP_VIDEO_MODEL_PATH"
                --text-encoder-path "$VPP_TEXT_ENCODER_PATH"
                --action-checkpoint "$VPP_ACTION_CHECKPOINT"
                --device cuda:0
                --output-mode canonical
                --confirm-canonical-adapted
                --adaptation-dataset bridge_widowx
                --adaptation-method "${VPP_ADAPTATION_METHOD:-frozen_video_backbone_action_head}"
                --host 127.0.0.1 --port "$POLICY_PORT"
            )
            ;;
        cosmos3)
            SERVER_COMMAND=(
                "$COSMOS_PYTHON" -u -m policy_servers.cosmos3.server
                --cosmos-root "$COSMOS_ROOT"
                --checkpoint "$COSMOS_CHECKPOINT"
                --domain-name "$COSMOS_DOMAIN_NAME"
                --confirm-cartesian-adapted
                --gripper-output "${COSMOS_GRIPPER_OUTPUT:-open_fraction}"
                --adaptation-dataset bridge_widowx
                --adaptation-method "${COSMOS_ADAPTATION_METHOD:-frozen_backbone_action_adapter}"
                --action-chunk-size 16
                --host 127.0.0.1 --port "$POLICY_PORT"
            )
            ;;
    esac
}

run_model() {
    local model="$1"
    local server_log="${LOG_DIR}/${model}_server.log"
    local evaluator_log="${LOG_DIR}/${model}_evaluator.log"
    local run_name="${model}-${RUN_TAG}"
    build_server_command "$model"

    note "$model server command"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$POLICY_GPU"
    printf '%q ' "${SERVER_COMMAND[@]}"
    printf '\n'
    note "$model evaluator command"
    printf '%q ' "$EVAL_PYTHON" -u -m simpler_env.evaluation.remote_evaluator \
        --config "$RUNTIME_CONFIG" --server-url "http://127.0.0.1:${POLICY_PORT}" --run-name "$run_name"
    printf '\n'
    ((DRY_RUN == 0)) || return 0

    if port_is_open; then
        die "TCP port $POLICY_PORT is already in use"
    fi

    note "starting $model on physical GPU $POLICY_GPU; log: $server_log"
    if [[ "$model" == "cosmos3" ]]; then
        env CUDA_VISIBLE_DEVICES="$POLICY_GPU" LD_LIBRARY_PATH="" \
            "${SERVER_COMMAND[@]}" >"$server_log" 2>&1 &
    else
        env CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
            "${SERVER_COMMAND[@]}" >"$server_log" 2>&1 &
    fi
    SERVER_PID=$!
    wait_for_server "$SERVER_PID" "$server_log"

    note "evaluating $model with SAPIEN on physical GPU $EVAL_GPU"
    # Keep all GPUs visible: runtime config explicitly uses cuda:EVAL_GPU.
    if env -u CUDA_VISIBLE_DEVICES "$EVAL_PYTHON" -u -m simpler_env.evaluation.remote_evaluator \
        --config "$RUNTIME_CONFIG" \
        --server-url "http://127.0.0.1:${POLICY_PORT}" \
        --run-name "$run_name" 2>&1 | tee "$evaluator_log"; then
        note "$model evaluation completed"
        cleanup_server
        return 0
    fi

    local status=${PIPESTATUS[0]}
    printf 'Evaluation failed for %s with status %s. Logs: %s and %s\n' \
        "$model" "$status" "$server_log" "$evaluator_log" >&2
    cleanup_server
    return "$status"
}

FAILURES=()
for model in "${MODELS[@]}"; do
    if ! run_model "$model"; then
        FAILURES+=("$model")
        ((CONTINUE_ON_ERROR == 1)) || exit 1
    fi
done

if ((${#FAILURES[@]})); then
    printf 'Completed with failures: %s\n' "${FAILURES[*]}" >&2
    exit 1
fi

note "all requested evaluations completed"
printf 'Results: %s\nLauncher logs: %s\n' "$RESULTS_DIR" "$LOG_DIR"
