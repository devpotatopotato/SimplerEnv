#!/usr/bin/env bash
# Set up four isolated uv environments for the SimplerEnv remote evaluation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_EVAL_HOME="${REMOTE_EVAL_HOME:-${REPO_ROOT}/.remote_eval}"
SOURCE_ROOT="${SOURCE_ROOT:-${REMOTE_EVAL_HOME}/sources}"
ENV_ROOT="${ENV_ROOT:-${REMOTE_EVAL_HOME}/envs}"

# These revisions are the upstream interfaces against which the adapters were audited.
OPENPI_REVISION="${OPENPI_REVISION:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
VPP_REVISION="${VPP_REVISION:-75cf37ad0f537627e1a3aa9a04bab70324ec27ff}"
COSMOS_REVISION="${COSMOS_REVISION:-103c5d1687d290b050e4890f48ff7a38b12742ef}"
COSMOS_CUDA_GROUP="${COSMOS_CUDA_GROUP:-cu130-train}"
VPP_TORCH_VERSION="${VPP_TORCH_VERSION:-2.7.1}"
VPP_TORCHVISION_VERSION="${VPP_TORCHVISION_VERSION:-0.22.1}"
VPP_TORCH_INDEX="${VPP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
ALLOW_SOURCE_REVISION_MISMATCH="${ALLOW_SOURCE_REVISION_MISMATCH:-0}"
VPP_REQUIREMENTS_FILE=""

SELECTED_MODELS="all"

usage() {
    cat <<'EOF'
Usage: setup.sh [--models LIST]

Options:
  --models LIST  Comma-separated: eval,pi05,vpp,cosmos3, or all (default).
  -h, --help     Show this help.

Environment overrides:
  REMOTE_EVAL_HOME        Environments and sources (default: <repo>/.remote_eval)
  COSMOS_CUDA_GROUP       cu130-train (default) or cu128-train
  VPP_TORCH_VERSION       Blackwell-capable PyTorch version (default: 2.7.1)
  VPP_TORCHVISION_VERSION Matching torchvision version (default: 0.22.1)
  ALLOW_SOURCE_REVISION_MISMATCH=1  Keep an existing checkout at another revision

The script never downloads adapted policy checkpoints. Fill models.env after setup.
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
        --models)
            (($# >= 2)) || die "--models requires a value"
            SELECTED_MODELS="$2"
            shift 2
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
    local requested="$1"
    [[ ",${SELECTED_MODELS}," == *",all,"* || ",${SELECTED_MODELS}," == *",${requested},"* ]]
}

for selected in ${SELECTED_MODELS//,/ }; do
    case "$selected" in
        all|eval|pi05|vpp|cosmos3) ;;
        *) die "unknown model in --models: $selected" ;;
    esac
done

command -v git >/dev/null || die "git is required"
command -v curl >/dev/null || die "curl is required"

if ! command -v uv >/dev/null; then
    note "uv is not installed; installing it with the official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null || die "uv installation completed but uv is not on PATH"

mkdir -p "$SOURCE_ROOT" "$ENV_ROOT"

clone_at_revision() {
    local name="$1"
    local url="$2"
    local destination="$3"
    local revision="$4"
    local recurse="${5:-0}"

    if [[ -d "${destination}/.git" ]]; then
        local current_revision
        current_revision="$(git -C "$destination" rev-parse HEAD)"
        if [[ "$current_revision" != "$revision" ]]; then
            if [[ "$ALLOW_SOURCE_REVISION_MISMATCH" == "1" ]]; then
                note "$name exists at $current_revision; retaining it by request"
            else
                die "$name exists at $current_revision, expected $revision. Set ALLOW_SOURCE_REVISION_MISMATCH=1 to retain it."
            fi
        else
            note "$name source already exists at the pinned revision"
        fi
        return
    fi
    [[ ! -e "$destination" ]] || die "$destination exists but is not a Git checkout"

    note "cloning $name"
    if [[ "$recurse" == "1" ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules "$url" "$destination"
    else
        git clone "$url" "$destination"
    fi
    git -C "$destination" checkout --detach "$revision"
    if [[ "$recurse" == "1" ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" submodule update --init --recursive
    fi
}

install_project_adapter() {
    local python_bin="$1"
    uv pip install --python "$python_bin" -e "$REPO_ROOT" --no-deps
}

setup_evaluator() {
    local env_dir="${ENV_ROOT}/simpler"
    local python_bin="${env_dir}/bin/python"
    note "creating the SimplerEnv evaluator environment"
    uv venv --python 3.10 --allow-existing "$env_dir"
    uv pip install --python "$python_bin" "numpy==1.24.4"
    uv pip install --python "$python_bin" -e "${REPO_ROOT}/ManiSkill2_real2sim"
    uv pip install --python "$python_bin" -e "$REPO_ROOT" --no-deps
    uv pip install --python "$python_bin" "mediapy==1.2.0" matplotlib
    "$python_bin" -m unittest discover -s "${REPO_ROOT}/tests" -v
    "$python_bin" -c 'import simpler_env, sapien; print("SimplerEnv/SAPIEN import OK")'
}

setup_pi05() {
    local source_dir="${SOURCE_ROOT}/openpi"
    clone_at_revision openpi https://github.com/Physical-Intelligence/openpi.git "$source_dir" "$OPENPI_REVISION" 1
    note "syncing the OpenPI environment"
    GIT_LFS_SKIP_SMUDGE=1 uv sync --project "$source_dir" --locked
    install_project_adapter "${source_dir}/.venv/bin/python"
    "${source_dir}/.venv/bin/python" -m policy_servers.pi05.server --help >/dev/null
}

setup_vpp() {
    local source_dir="${SOURCE_ROOT}/video-prediction-policy"
    local env_dir="${ENV_ROOT}/vpp"
    local python_bin="${env_dir}/bin/python"
    clone_at_revision VPP https://github.com/roboterax/video-prediction-policy.git "$source_dir" "$VPP_REVISION"
    note "creating the VPP environment"
    uv venv --python 3.10 --allow-existing "$env_dir"
    # Upstream pins torch 2.0.1, which cannot execute on Blackwell. Install a
    # CUDA 12.8 Blackwell-capable build and omit only that old requirements pin.
    uv pip install --python "$python_bin" --index-url "$VPP_TORCH_INDEX" \
        "torch==${VPP_TORCH_VERSION}" "torchvision==${VPP_TORCHVISION_VERSION}"
    uv pip install --python "$python_bin" "numpy==1.26.4"
    VPP_REQUIREMENTS_FILE="$(mktemp "${REMOTE_EVAL_HOME}/vpp-requirements.XXXXXX.txt")"
    trap '[[ -z "$VPP_REQUIREMENTS_FILE" ]] || rm -f -- "$VPP_REQUIREMENTS_FILE"' EXIT
    sed '/^[[:space:]]*torch==/d' "${source_dir}/requirements.txt" >"$VPP_REQUIREMENTS_FILE"
    uv pip install --python "$python_bin" -r "$VPP_REQUIREMENTS_FILE"
    rm -f -- "$VPP_REQUIREMENTS_FILE"
    VPP_REQUIREMENTS_FILE=""
    trap - EXIT
    install_project_adapter "$python_bin"
    "$python_bin" -m policy_servers.vpp.server --help >/dev/null
}

setup_cosmos3() {
    local source_dir="${SOURCE_ROOT}/cosmos-framework"
    clone_at_revision Cosmos https://github.com/NVIDIA/cosmos-framework.git "$source_dir" "$COSMOS_REVISION"
    note "syncing Cosmos with dependency group ${COSMOS_CUDA_GROUP}"
    uv sync --project "$source_dir" --locked --all-extras --group "$COSMOS_CUDA_GROUP"
    install_project_adapter "${source_dir}/.venv/bin/python"
    "$source_dir/.venv/bin/python" -m policy_servers.cosmos3.server --help >/dev/null
}

if wants_model eval; then
    setup_evaluator
fi
if wants_model pi05; then
    setup_pi05
fi
if wants_model vpp; then
    setup_vpp
fi
if wants_model cosmos3; then
    setup_cosmos3
fi

if [[ ! -e "${SCRIPT_DIR}/models.env" ]]; then
    cp "${SCRIPT_DIR}/models.env.example" "${SCRIPT_DIR}/models.env"
    note "created ${SCRIPT_DIR}/models.env"
fi

cat <<EOF

Setup complete.

Next:
  1. Edit ${SCRIPT_DIR}/models.env with your adapted checkpoint paths.
  2. Test rendering: ${SCRIPT_DIR}/smoke_test.sh
  3. Run all models: ${SCRIPT_DIR}/run.sh

Environments and upstream sources are under:
  ${REMOTE_EVAL_HOME}
EOF
