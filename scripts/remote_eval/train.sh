#!/usr/bin/env bash
# Prepare common Bridge data and declare all three SimplerEnv policy regimes.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_EVAL_HOME="${REMOTE_EVAL_HOME:-${REPO_ROOT}/.remote_eval}"
SOURCE_ROOT="${SOURCE_ROOT:-${REMOTE_EVAL_HOME}/sources}"
ENV_ROOT="${ENV_ROOT:-${REMOTE_EVAL_HOME}/envs}"
DATA_HOME="${BRIDGE_DATA_HOME:-${REMOTE_EVAL_HOME}/data/bridge}"
TRAINING_HOME="${TRAINING_HOME:-${REMOTE_EVAL_HOME}/training}"
MODEL_HOME="${MODEL_HOME:-${REMOTE_EVAL_HOME}/models}"

OPENPI_ROOT="${SOURCE_ROOT}/openpi"
VPP_ROOT="${SOURCE_ROOT}/video-prediction-policy"
COSMOS_ROOT="${SOURCE_ROOT}/cosmos-framework"
OPENPI_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
VPP_PYTHON="${ENV_ROOT}/vpp/bin/python"
MODELS_ENV="${SCRIPT_DIR}/models.env"

SELECTED_MODELS="pi05,vpp,cosmos3"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-gs://gresearch/robotics/bridge/0.1.0}"
BRIDGE_TASK_FILTER="${BRIDGE_TASK_FILTER:-${REPO_ROOT}/configs/remote_training/bridge_tasks.json}"
BRIDGE_VALIDATION_FRACTION="${BRIDGE_VALIDATION_FRACTION:-0.1}"
BRIDGE_MAX_EPISODES="${BRIDGE_MAX_EPISODES:-0}"
BRIDGE_MAX_EPISODES_PER_TASK="${BRIDGE_MAX_EPISODES_PER_TASK:-0}"
BRIDGE_IMAGE_WRITER_THREADS="${BRIDGE_IMAGE_WRITER_THREADS:-16}"

PI05_TRAIN_STEPS="${PI05_TRAIN_STEPS:-30000}"
PI05_BATCH_SIZE="${PI05_BATCH_SIZE:-32}"
PI05_SAVE_INTERVAL="${PI05_SAVE_INTERVAL:-1000}"
PI05_NORM_MAX_FRAMES="${PI05_NORM_MAX_FRAMES:-0}"
VPP_TRAIN_STEPS="${VPP_TRAIN_STEPS:-50000}"
VPP_BATCH_SIZE="${VPP_BATCH_SIZE:-2}"
VPP_GRADIENT_ACCUMULATION="${VPP_GRADIENT_ACCUMULATION:-4}"
VPP_SAVE_INTERVAL="${VPP_SAVE_INTERVAL:-1000}"
VPP_MAX_VAL_BATCHES="${VPP_MAX_VAL_BATCHES:-16}"
VPP_SAMPLE_STRIDE="${VPP_SAMPLE_STRIDE:-1}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
VPP_TRAIN_PORT="${VPP_TRAIN_PORT:-29673}"

DRY_RUN=0
DATA_ONLY=0

usage() {
    cat <<'EOF'
Usage: train.sh [options]

Builds one common Bridge/WidowX dataset, adapts π0.5 and VPP, configures the
released Cosmos3-Edge native Bridge domain, writes an auditable manifest,
updates models.env, and validates run.sh.

Options:
  --models LIST  Comma-separated pi05,vpp,cosmos3 (default: all three)
  --data-only    Convert/validate the common dataset, then stop
  --dry-run      Print commands without downloads, conversion, or training
  -h, --help     Show this help

Useful environment overrides:
  TRAIN_GPUS=0,1
  BRIDGE_SOURCE=gs://gresearch/robotics/bridge/0.1.0
  BRIDGE_MAX_EPISODES=0       0 means all matching episodes
  BRIDGE_MAX_EPISODES_PER_TASK=0  0 means no per-task cap
  BRIDGE_IMAGE_WRITER_THREADS=16  CPU image-writing concurrency
  PI05_TRAIN_STEPS=30000      PI05_BATCH_SIZE=32
  VPP_TRAIN_STEPS=50000       VPP_BATCH_SIZE=2
  VPP_GRADIENT_ACCUMULATION=4 TRAIN_NUM_WORKERS=4
  TRAIN_SEED=42

Interrupted model training resumes from last.pt/Orbax checkpoints. A complete
dataset is reused. An incomplete dataset conversion is rejected rather than
silently treated as complete.

The default command uses two GPUs. One-GPU training and shared-GPU evaluation
are also supported:
  TRAIN_GPUS=0,1 ./scripts/remote_eval/train.sh
  TRAIN_GPUS=0 ./scripts/remote_eval/train.sh
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    ((DRY_RUN == 1)) || "$@"
}

while (($#)); do
    case "$1" in
        --models)
            (($# >= 2)) || die "--models requires a value"
            SELECTED_MODELS="$2"
            shift 2
            ;;
        --data-only)
            DATA_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
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

wants_model() {
    [[ ",${SELECTED_MODELS}," == *",$1,"* ]]
}

for model in ${SELECTED_MODELS//,/ }; do
    case "$model" in
        pi05|vpp|cosmos3) ;;
        *) die "unknown model: $model" ;;
    esac
done
[[ -n "$SELECTED_MODELS" ]] || die "no models selected"

command -v uv >/dev/null || die "uv is required; run setup.sh first"
[[ -d "$OPENPI_ROOT" && -x "$OPENPI_PYTHON" ]] || die "OpenPI is missing; run setup.sh --models pi05"
if ((DATA_ONLY == 0)) && wants_model vpp; then
    [[ -d "$VPP_ROOT" && -x "$VPP_PYTHON" ]] || die "VPP is missing; run setup.sh --models vpp"
fi
if ((DATA_ONLY == 0)) && wants_model cosmos3; then
    [[ -d "$COSMOS_ROOT" && -x "${COSMOS_ROOT}/.venv/bin/python" ]] || \
        die "Cosmos is missing; run setup.sh --models cosmos3"
fi
[[ -f "$BRIDGE_TASK_FILTER" ]] || die "missing Bridge task filter: $BRIDGE_TASK_FILTER"

IFS=',' read -r -a GPU_LIST <<<"$TRAIN_GPUS"
((${#GPU_LIST[@]} > 0)) || die "TRAIN_GPUS is empty"
for gpu in "${GPU_LIST[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index in TRAIN_GPUS: $gpu"
done
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq "${#GPU_LIST[@]}" ]] || \
    die "TRAIN_GPUS contains duplicate GPU indices"
if ((DATA_ONLY == 0 && DRY_RUN == 0)); then
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
    PHYSICAL_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    for gpu in "${GPU_LIST[@]}"; do
        ((gpu < PHYSICAL_GPU_COUNT)) || die "GPU $gpu does not exist ($PHYSICAL_GPU_COUNT visible GPUs)"
    done
fi
if ((DATA_ONLY == 0)) && wants_model pi05; then
    ((PI05_BATCH_SIZE % ${#GPU_LIST[@]} == 0)) || \
        die "PI05_BATCH_SIZE must be divisible by the number of TRAIN_GPUS"
fi

mkdir -p "$DATA_HOME" "$TRAINING_HOME" "$MODEL_HOME"

note "installing training-only dependencies into the isolated model environments"
run_command env GIT_LFS_SKIP_SMUDGE=1 uv sync --project "$OPENPI_ROOT" --frozen --group rlds
run_command uv pip install --python "$OPENPI_PYTHON" -e "$REPO_ROOT" --no-deps
if ((DATA_ONLY == 0)) && wants_model vpp; then
    run_command uv pip install --python "$VPP_PYTHON" "pyarrow==20.0.0"
    run_command uv pip install --python "$VPP_PYTHON" -e "$REPO_ROOT" --no-deps
fi

TRAIN_REPO_ID="local/simpler_bridge_train"
VAL_REPO_ID="local/simpler_bridge_val"
TRAIN_DATASET="${DATA_HOME}/${TRAIN_REPO_ID}"
VAL_DATASET="${DATA_HOME}/${VAL_REPO_ID}"
DATA_MANIFEST="${DATA_HOME}/simpler_bridge_manifest.json"

if [[ -f "${TRAIN_DATASET}/meta/info.json" && -f "${VAL_DATASET}/meta/info.json" && -f "$DATA_MANIFEST" ]]; then
    note "using complete canonical Bridge dataset at $DATA_HOME"
else
    [[ ! -e "$TRAIN_DATASET" && ! -e "$VAL_DATASET" ]] || \
        die "incomplete dataset exists under $DATA_HOME; inspect it and move it aside before retrying"
    note "converting Bridge episodes on CPU/network/disk; GPUs begin at π0.5 training"
    DATA_COMMAND=(
        env CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 HF_LEROBOT_HOME="$DATA_HOME"
        "$OPENPI_PYTHON" -m simpler_training.bridge_data
        --source "$BRIDGE_SOURCE"
        --output-home "$DATA_HOME"
        --task-filter "$BRIDGE_TASK_FILTER"
        --validation-fraction "$BRIDGE_VALIDATION_FRACTION"
        --fps 5
        --image-writer-threads "$BRIDGE_IMAGE_WRITER_THREADS"
    )
    if ((BRIDGE_MAX_EPISODES > 0)); then
        DATA_COMMAND+=(--max-episodes "$BRIDGE_MAX_EPISODES")
    fi
    if ((BRIDGE_MAX_EPISODES_PER_TASK > 0)); then
        DATA_COMMAND+=(--max-episodes-per-task "$BRIDGE_MAX_EPISODES_PER_TASK")
    fi
    run_command "${DATA_COMMAND[@]}"
fi

if ((DATA_ONLY == 1)); then
    note "data preparation complete"
    printf 'Manifest: %s\n' "$DATA_MANIFEST"
    exit 0
fi

PI05_OUTPUT="${TRAINING_HOME}/pi05"
PI05_FINAL="${PI05_OUTPUT}/checkpoints/pi05_simpler_bridge_lora/bridge_lora/$((PI05_TRAIN_STEPS - 1))"
VPP_OUTPUT="${TRAINING_HOME}/vpp"
VPP_VIDEO_MODEL="${MODEL_HOME}/vpp/svd-robot"
VPP_TEXT_ENCODER="${MODEL_HOME}/vpp/clip-vit-base-patch32"

if wants_model pi05; then
    note "training π0.5 LoRA on GPUs $TRAIN_GPUS"
    PI05_COMMAND=(
        env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" HF_LEROBOT_HOME="$DATA_HOME" WANDB_MODE="${WANDB_MODE:-disabled}"
        "$OPENPI_PYTHON" -m simpler_training.openpi_train
        --openpi-root "$OPENPI_ROOT"
        --repo-id "$TRAIN_REPO_ID"
        --output-root "$PI05_OUTPUT"
        --exp-name bridge_lora
        --train-steps "$PI05_TRAIN_STEPS"
        --batch-size "$PI05_BATCH_SIZE"
        --num-workers "$TRAIN_NUM_WORKERS"
        --save-interval "$PI05_SAVE_INTERVAL"
        --seed "$TRAIN_SEED"
    )
    if ((PI05_NORM_MAX_FRAMES > 0)); then
        PI05_COMMAND+=(--norm-max-frames "$PI05_NORM_MAX_FRAMES")
    fi
    run_command "${PI05_COMMAND[@]}"
fi

if wants_model vpp; then
    note "downloading VPP's frozen public backbones when absent"
    run_command "$VPP_PYTHON" -m simpler_training.hf_download yjguo/svd-robot "$VPP_VIDEO_MODEL"
    run_command "$VPP_PYTHON" -m simpler_training.hf_download \
        openai/clip-vit-base-patch32 "$VPP_TEXT_ENCODER"

    note "training the VPP action model on GPUs $TRAIN_GPUS"
    VPP_LAUNCH=(
        env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" WANDB_MODE="${WANDB_MODE:-disabled}"
        "$VPP_PYTHON" -m accelerate.commands.launch
        --num_processes "${#GPU_LIST[@]}"
        --main_process_port "$VPP_TRAIN_PORT"
    )
    if ((${#GPU_LIST[@]} > 1)); then
        VPP_LAUNCH+=(--multi_gpu)
    fi
    VPP_LAUNCH+=(
        -m simpler_training.vpp_train
        --vpp-root "$VPP_ROOT"
        --train-dataset "$TRAIN_DATASET"
        --val-dataset "$VAL_DATASET"
        --template-config "${REPO_ROOT}/configs/remote_training/vpp_simpler_bridge.yaml"
        --video-model-path "$VPP_VIDEO_MODEL"
        --text-encoder-path "$VPP_TEXT_ENCODER"
        --output-dir "$VPP_OUTPUT"
        --max-steps "$VPP_TRAIN_STEPS"
        --batch-size "$VPP_BATCH_SIZE"
        --gradient-accumulation "$VPP_GRADIENT_ACCUMULATION"
        --num-workers "$TRAIN_NUM_WORKERS"
        --save-interval "$VPP_SAVE_INTERVAL"
        --max-val-batches "$VPP_MAX_VAL_BATCHES"
        --sample-stride "$VPP_SAMPLE_STRIDE"
        --seed "$TRAIN_SEED"
    )
    run_command "${VPP_LAUNCH[@]}"
fi

if wants_model cosmos3; then
    note "using Cosmos3-Edge's released native bridge_orig_lerobot action domain"
    printf '%s\n' \
        "Cosmos is not locally post-trained: the official Edge recipe is not a validated two-GPU recipe." \
        "Its results will be labeled native_pretrained_bridge, separately from π0.5/VPP shared adaptation."
fi

if ((DRY_RUN == 1)); then
    note "dry run complete; models.env was not changed"
    exit 0
fi

TRAINING_MANIFEST="${TRAINING_HOME}/training_manifest.json"
MANIFEST_ARGS=(
    --dataset-manifest "$DATA_MANIFEST"
    --output "$TRAINING_MANIFEST"
    --seed "$TRAIN_SEED"
)
UPDATE_ARGS=("TRAINED_MODELS=${SELECTED_MODELS}" "TRAINING_MANIFEST=${TRAINING_MANIFEST}")
if wants_model pi05; then
    [[ -d "$PI05_FINAL" ]] || die "missing final π0.5 checkpoint: $PI05_FINAL"
    UPDATE_ARGS+=(
        "PI05_CONFIG_NAME=pi05_simpler_bridge_lora"
        "PI05_CHECKPOINT=${PI05_FINAL}"
        "PI05_ADAPTATION_METHOD=lora"
        "PI05_COMPARISON_GROUP=shared_bridge_adaptation"
    )
    MANIFEST_ARGS+=(--artifact "pi05=${PI05_OUTPUT}/artifacts.json")
fi
if wants_model vpp; then
    [[ -f "${VPP_OUTPUT}/final.pt" && -f "${VPP_OUTPUT}/config.yaml" ]] || \
        die "missing final VPP artifacts under $VPP_OUTPUT"
    UPDATE_ARGS+=(
        "VPP_CONFIG=${VPP_OUTPUT}/config.yaml"
        "VPP_VIDEO_MODEL_PATH=${VPP_VIDEO_MODEL}"
        "VPP_TEXT_ENCODER_PATH=${VPP_TEXT_ENCODER}"
        "VPP_ACTION_CHECKPOINT=${VPP_OUTPUT}/final.pt"
        "VPP_ADAPTATION_METHOD=frozen_video_backbone_action_head"
        "VPP_COMPARISON_GROUP=shared_bridge_adaptation"
    )
    MANIFEST_ARGS+=(--artifact "vpp=${VPP_OUTPUT}/artifacts.json")
fi
if wants_model cosmos3; then
    UPDATE_ARGS+=(
        "COSMOS_CHECKPOINT=Cosmos3-Edge"
        "COSMOS_DOMAIN_NAME=bridge_orig_lerobot"
        "COSMOS_ADAPTATION_DATASET=upstream_bridge_original"
        "COSMOS_ADAPTATION_METHOD=native_bridge_action_head"
        "COSMOS_COMPARISON_GROUP=native_pretrained_bridge"
    )
    MANIFEST_ARGS+=(--native "cosmos3=Cosmos3-Edge")
fi

[[ -f "$MODELS_ENV" ]] || cp "${SCRIPT_DIR}/models.env.example" "$MODELS_ENV"
run_command "$OPENPI_PYTHON" -m simpler_training.training_manifest "${MANIFEST_ARGS[@]}"
run_command "$OPENPI_PYTHON" -m simpler_training.models_env "$MODELS_ENV" "${UPDATE_ARGS[@]}"

note "validating the evaluation launcher against the produced artifacts"
POLICY_GPU="${GPU_LIST[0]}"
EVAL_GPU="${GPU_LIST[1]:-${GPU_LIST[0]}}"
if [[ "$POLICY_GPU" == "$EVAL_GPU" ]]; then
    note "validating shared-GPU evaluation on physical GPU $POLICY_GPU"
fi
POLICY_GPU="$POLICY_GPU" EVAL_GPU="$EVAL_GPU" \
    "${SCRIPT_DIR}/run.sh" --models "$SELECTED_MODELS" --dry-run

note "training pipeline complete"
printf '\nRun the adapted evaluation with:\n  cd %q\n  POLICY_GPU=%q EVAL_GPU=%q %q\n' \
    "$REPO_ROOT" "$POLICY_GPU" "$EVAL_GPU" "${SCRIPT_DIR}/run.sh"
