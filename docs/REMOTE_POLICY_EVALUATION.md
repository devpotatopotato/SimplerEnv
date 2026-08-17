# Headless evaluation of π0.5, VPP, and Cosmos3-Edge

This pipeline evaluates all three policies on the SimplerEnv WidowX Bridge
tasks without a desktop or X server. Policy inference and simulation are
separate processes with isolated dependencies. A two-GPU run uses one RTX PRO
6000 for the active policy and one for SAPIEN/Vulkan. A one-GPU run keeps the
same process boundary but shares one card between inference and rendering;
models run sequentially in both modes.

## What is comparable

SimplerEnv's WidowX controller takes seven **control values**, not seven robot
joints:

```text
[Δx, Δy, Δz] metres
+ [axis_x·angle, axis_y·angle, axis_z·angle] radians
+ gripper, where +1=open and -1=closed
```

Every server must return this `eef_delta_axis_angle_gripper_v1` contract. The
evaluator applies the same 5 cm translation and 0.25 radian rotation limits,
executes one action, and replans at 5 Hz.

The model training regimes are deliberately reported separately:

- π0.5 is LoRA-adapted on the local canonical Bridge split.
- VPP keeps its public SVD/CLIP backbones and trains its action policy on the
  same local split.
- `nvidia/Cosmos3-Edge` uses the released native `bridge_orig_lerobot` action
  domain. It is not locally post-trained because NVIDIA's Edge post-training
  recipe is not a validated two-GPU recipe. Its score shares the evaluation
  protocol, but not the π0.5/VPP adaptation-data budget.

Do not substitute `Cosmos3-Edge-Policy-DROID`. DROID emits absolute Franka
joint targets, for which there is no model-independent WidowX Cartesian
conversion. The native base Edge checkpoint is suitable because it includes a
Bridge 10-D action domain. The adapter applies its official quantile statistics,
5 Hz conditioning, ego-view prompt, backward-framewise pose convention, and
Bridge flange/OpenCV tool transform before returning canonical deltas.

The upstream interfaces are pinned to audited revisions of
[OpenPI](https://github.com/Physical-Intelligence/openpi),
[VPP](https://github.com/roboterax/video-prediction-policy), and
[Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The base model
is [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge).

## Standard evaluation protocol

`configs/remote_eval/widowx_bridge.json` reproduces SimplerEnv's Bridge visual
matching protocol:

- four tasks;
- fixed object episode IDs 0–23, for 24 variations per task and 96 trials per
  model;
- 60-step limits for spoon, carrot, and cube stacking;
- a 120-step limit for eggplant-in-basket;
- the prepackaged Bridge scenes, overlays, camera, robot initialization,
  5 Hz control, and 500 Hz simulation;
- one policy seed (`0`) independent of the environment seed and object ID.

The fixed 96-trial protocol is the primary result. To study stochastic
variation, copy the config and set `policy_seeds` to multiple predeclared values
for **every** model. This forms the Cartesian product of object IDs and policy
seeds; do not add seeds only for a poorly performing model.

The summary reports micro and per-task success, Wilson 95% intervals, safety
clipping, inference/round-trip latency, and paired bootstrap intervals over
identical task/object/policy-seed trials. It warns when a pair belongs to
different training groups.

## Architecture

```text
GPU EVAL_GPU                                      GPU POLICY_GPU

SAPIEN/Vulkan environment
        │ RGB + proprioception
        ▼
RemotePolicy ── HTTP/JSON ─────────────────────► one isolated policy server
        ▲                                         │ native preprocessing
        │ canonical action chunk                  │ checkpoint inference
        └─────────────────────────────────────────┘
        │ common execution horizon and safety
        ▼
WidowX controller ──► JSONL, metadata, summary, videos
```

Key boundaries:

- `simpler_protocol/` owns the dependency-light schema and transport.
- `policy_servers/` owns model loading and model-specific transformations. It
  never creates a simulator.
- `simpler_env/evaluation/remote_evaluator.py` owns tasks, fixed variations,
  independent seeds, success, safety, videos, and metrics.
- `scripts/remote_eval/train.sh` owns data/adaptation artifacts and the training
  manifest; `run.sh` owns process/GPU orchestration.

## One-time setup

On the server:

```bash
git clone https://github.com/devpotatopotato/SimplerEnv.git
cd SimplerEnv
./scripts/remote_eval/setup.sh 2>&1 | tee setup-all.log
EVAL_GPU=4 ./scripts/remote_eval/smoke_test.sh
```

`smoke_test.sh` defaults to GPU 0, so the override is only needed when testing
a different physical card.

`setup.sh` creates isolated uv environments under `.remote_eval/`, clones the
pinned upstream sources, installs Blackwell-compatible dependencies, and copies
`models.env.example` to `models.env` only when the latter does not exist. It
does not overwrite completed checkpoints or an existing `models.env`.

The evaluator environment pins `setuptools==80.9.0` because SAPIEN 2.2.2 still
imports `pkg_resources`. Headless rendering uses the NVIDIA Vulkan ICD directly;
`vulkaninfo` is helpful for diagnostics but not required.

Setup is resumable and can also be split:

```bash
./scripts/remote_eval/setup.sh --models eval,pi05
./scripts/remote_eval/setup.sh --models vpp
./scripts/remote_eval/setup.sh --models cosmos3
```

## Prepare the policies

If π0.5 and VPP have already finished, rerunning this command reuses the
converted dataset, OpenPI normalization statistics, and final checkpoints. It
also adds the new Cosmos configuration and manifest without retraining finished
models:

```bash
TRAIN_GPUS=3,4 ./scripts/remote_eval/train.sh 2>&1 | tee train-all.log
```

The command:

1. reuses or converts one deterministic canonical Bridge train/validation
   dataset;
2. records which evaluation task families are present or absent in the public
   demonstrations;
3. resumes or trains the 30,000-step π0.5 LoRA run;
4. resumes or trains the predeclared 50,000-step VPP action-policy run;
5. configures `Cosmos3-Edge` with `bridge_orig_lerobot` without pretending that
   it received matched local training;
6. writes `.remote_eval/training/training_manifest.json`, updates
   `scripts/remote_eval/models.env`, and dry-validates all launch commands.

Data conversion is CPU/network/disk work; no GPU utilization is expected during
that stage. Model training begins after conversion. Completed conversion and
training outputs are reused on later runs. Do not change step counts in the
middle of a resumed primary run.

The manifest records declared sample exposure: π0.5 uses 30,000 steps with
global batch 32 (960,000 samples); VPP uses 50,000 updates with effective global
batch 16 (800,000 samples). These are predeclared architecture-specific budgets,
not a compute-matched claim. If you run a matched-sample ablation, use a fresh
VPP run with 60,000 steps and label it separately rather than changing the
already completed primary run's learning-rate schedule.

Useful variants:

```bash
# Run the complete preparation/training pipeline on one physical GPU.
TRAIN_GPUS=3 ./scripts/remote_eval/train.sh

# Prepare or resume only selected policies.
TRAIN_GPUS=3,4 ./scripts/remote_eval/train.sh --models pi05,vpp
./scripts/remote_eval/train.sh --models cosmos3

# Only build/validate the common dataset.
./scripts/remote_eval/train.sh --data-only

# Print commands without changing anything.
./scripts/remote_eval/train.sh --dry-run
```

For a short integration test, use separate directories so trial weights cannot
be mistaken for the primary run:

```bash
BRIDGE_DATA_HOME="$PWD/.remote_eval/data/bridge_trial" \
TRAINING_HOME="$PWD/.remote_eval/training_trial" \
BRIDGE_MAX_EPISODES_PER_TASK=25 \
PI05_TRAIN_STEPS=10 PI05_SAVE_INTERVAL=5 \
VPP_TRAIN_STEPS=10 VPP_SAVE_INTERVAL=5 \
TRAIN_GPUS=3,4 ./scripts/remote_eval/train.sh 2>&1 | tee train-trial.log
```

## Evaluate all three models

Two GPUs provide the most memory headroom and keep rendering isolated from
policy inference:

```bash
POLICY_GPU=3 EVAL_GPU=4 POLICY_PORT=43891 \
RUN_TAG=bridge-standard-v1 \
./scripts/remote_eval/run.sh --models pi05,vpp,cosmos3 --continue-on-error \
2>&1 | tee eval-all.log
```

To use one physical GPU, assign the same index to both processes:

```bash
POLICY_GPU=3 EVAL_GPU=3 POLICY_PORT=43891 \
RUN_TAG=bridge-standard-v1-one-gpu \
./scripts/remote_eval/run.sh --models pi05,vpp,cosmos3 --continue-on-error \
2>&1 | tee eval-all-one-gpu.log
```

Shared-GPU mode is automatic when the indices are equal. CUDA inference and
Vulkan rendering mostly alternate, but both processes retain their allocations,
so this mode can be slower and needs enough combined VRAM. On a host that
exposes only one GPU, `run.sh` automatically defaults both roles to GPU 0.

The first Cosmos run downloads `nvidia/Cosmos3-Edge`, so its server startup can
take much longer than later runs. `SERVER_START_TIMEOUT` defaults to 1800
seconds and can be increased. All models use the same uncommon local port but
run sequentially. `run.sh --dry-run` validates artifacts and prints exact
commands without loading a model.

Each model creates:

```text
results/remote_eval/<model>-<RUN_TAG>/
├── evaluation_config.json
├── server_metadata.json
├── training_manifest.json    # local-adaptation group only
├── episodes.jsonl
├── summary.json
└── videos/<task>/*.mp4
```

The launcher logs are in `results/remote_eval/_launcher/<RUN_TAG>/`.
`server_metadata.json` records the exact checkpoint, upstream revision,
transforms, action domain, and training-comparison group. The evaluator rejects
an incompatible policy profile before starting an episode.

## Inspect results

```bash
./scripts/remote_eval/summary.sh bridge-standard-v1 --strict

# Machine-readable report:
./scripts/remote_eval/summary.sh bridge-standard-v1 --json > report.json
```

With no tag, `summary.sh` selects the latest launcher run. `--models` restricts
or requires a comma-separated subset. `--strict` exits nonzero if any selected
model lacks an artifact or has fewer episodes than the config declares.

The headline values should be presented as two views:

1. π0.5 versus VPP: matched local Bridge adaptation group.
2. Cosmos3-Edge native Bridge score alongside them: same evaluation protocol,
   different training regime.

This distinction is necessary for a defensible comparison; a common controller
interface alone does not make pretraining and adaptation budgets identical.
