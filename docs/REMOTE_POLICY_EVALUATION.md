# Headless remote-policy evaluation

This pipeline evaluates π0.5, Video Prediction Policy (VPP), and Cosmos 3 Edge Policy under one SimplerEnv contract while keeping their mutually incompatible dependencies in separate processes. It is designed for the 8× RTX PRO 6000 Blackwell machine and uses only two GPUs per run.

## Compatibility decision

“7-DOF” is ambiguous. SimplerEnv's WidowX controller accepts seven **control values**, not seven robot joints:

```text
[Δx, Δy, Δz] metres
+ [axis_x·angle, axis_y·angle, axis_z·angle] radians
+ [gripper], where +1 means open and -1 means closed
```

This is the sole wire-level action representation, named `eef_delta_axis_angle_gripper_v1`. Model-specific normalization and representation changes happen inside the corresponding policy server. The evaluator applies the same 5 cm translation and 0.25 radian rotation safety limits to every model and records every clipped step.

The public checkpoints are not equally compatible:

- π0.5 can produce 7-D LIBERO controls, but those are not automatically physical WidowX deltas. Use a Bridge/WidowX-adapted checkpoint for the main comparison; `libero_normalized` is a separately labeled diagnostic conversion.
- Released VPP action weights are CALVIN-specific. Use a Bridge/WidowX-adapted action head for the main comparison; `calvin_normalized` is a separately labeled diagnostic conversion.
- The public `Cosmos3-Edge-Policy-DROID` checkpoint outputs absolute DROID joint positions (seven joints plus gripper). There is no valid model-independent mapping from DROID joints to WidowX Cartesian actions. The adapter therefore requires an explicitly confirmed checkpoint post-trained with Cosmos's `midtrain` Cartesian representation.

These choices follow the official interfaces in [OpenPI](https://github.com/Physical-Intelligence/openpi), [VPP](https://github.com/roboterax/video-prediction-policy), and [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The released Cosmos Edge checkpoint's model card is [here](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID).

## Architecture

```text
GPU 1 / simpler_env environment                 GPU 0 / one model environment

SAPIEN offscreen renderer
        │ RGB + proprioception
        ▼
RemotePolicy ── HTTP/JSON protocol ───────────► π0.5 OR VPP OR Cosmos adapter
        ▲                                      │ model-native preprocessing
        │ canonical action chunk               │ isolated framework/checkpoint
        └───────────────────────────────────────┘
        │ fixed execution horizon + safety + robot gripper mapping
        ▼
WidowX Cartesian controller ──► JSONL metrics, metadata, summary, video
```

The layers are deliberately separate:

- `simpler_protocol/`: dependency-light schema, image transport, HTTP client/server, and rotation conversions.
- `policy_servers/`: model-native loading and conversion only. These modules never create a simulator.
- `simpler_env/policies/remote_policy.py`: chunk/replanning policy shared by every model.
- `simpler_env/evaluation/remote_evaluator.py`: environments, seeds, task success, safety, videos, and metrics only.
- `configs/remote_eval/`: immutable comparison settings.

Endpoints are `GET /healthz`, `GET /v1/metadata`, `POST /v1/reset`, and `POST /v1/actions`. The model server serializes inference with a lock, so accidental concurrent HTTP calls cannot race a stateful sampler.

## Fair comparison protocol

Use the adapted track as the primary result:

1. Use the same Bridge/WidowX demonstrations, train/validation split, language labels, primary third-person camera, and proprioception availability for all models. The current profile is `simpler_widowx_cartesian_v1`.
2. Convert training labels once to physical `Δxyz + Δaxis-angle + gripper-open`; store normalization statistics from the training split only.
3. Freeze each large visual/world-model backbone and train its action head or a parameter-efficient adapter. This is feasible on two 96 GB GPUs and is a more defensible common budget than full end-to-end tuning, which has very different cost across these architectures.
4. Match optimizer-step or GPU-hour budgets, report trainable parameter counts and actual GPU hours, and select checkpoints only on the common validation set.
5. Evaluate the untouched test tasks with exactly this config. Use identical seeds, `execution_horizon`, safety bounds, maximum steps, and camera inputs.
6. Report task success rate plus round-trip/server latency, clipping rate, memory, training cost, and per-task results. Run zero-shot released checkpoints in a separate table; never average them into the adapted comparison.

The default `execution_horizon` is 1, so all policies replan at the same 5 Hz control boundary. For a throughput study, set it to 4 for **all three models** and report that as a separate condition. The server may predict 16 actions, but the evaluator discards the unused suffix after the common execution horizon.

The WidowX environments do not expose a wrist camera. The common benchmark sends only `3rd_view_camera`; π0.5 and VPP zero-fill the required wrist tensor. Adapted checkpoints must be trained with this same missing-view behavior. Supplying a real wrist view to only one model would be a different experiment.

## Install and verify the headless evaluator

### Automated two-GPU setup and launcher

The repository includes a uv-based setup and sequential launcher. It creates four isolated environments, pins the audited upstream revisions, runs one policy at a time on GPU 0, and runs headless SAPIEN on GPU 1:

```bash
cd /path/to/SimplerEnv
./scripts/remote_eval/setup.sh

# Add the three adapted checkpoint/config paths created by your training runs.
nano scripts/remote_eval/models.env

./scripts/remote_eval/smoke_test.sh
./scripts/remote_eval/run.sh
```

Setup can be performed in pieces and safely rerun:

```bash
./scripts/remote_eval/setup.sh --models eval,pi05
./scripts/remote_eval/setup.sh --models vpp
./scripts/remote_eval/setup.sh --models cosmos3
```

Use `./scripts/remote_eval/run.sh --dry-run` to validate paths and print all commands. `--models pi05` runs just one policy, while `--continue-on-error` attempts the remaining policies if one fails. See [`models.env.example`](../scripts/remote_eval/models.env.example) for required values. The setup script does not and cannot create the three adapted policy checkpoints.

Use the repository's normal SimplerEnv installation in one environment. No X server or desktop is required. The config passes `offscreen_only=true` and `device=cuda:1` directly to SAPIEN's Vulkan renderer.

```bash
conda activate simpler_env
cd /path/to/SimplerEnv
pip install -e .
python -m unittest discover -s tests -v
```

First smoke-test the transport with the safe stationary mock policy:

```bash
# Terminal 1
conda activate simpler_env
cd /path/to/SimplerEnv
python -m policy_servers.mock_server --host 127.0.0.1 --port 8000

# Terminal 2; leave all GPUs visible because the JSON explicitly selects cuda:1
conda activate simpler_env
cd /path/to/SimplerEnv
unset DISPLAY
python -m simpler_env.evaluation.remote_evaluator \
  --config configs/remote_eval/widowx_smoke.json \
  --server-url http://127.0.0.1:8000 \
  --run-name mock-smoke
```

Your NVIDIA ICD and `libvulkan.so`/`libEGL_nvidia.so` installation are the important headless prerequisites. `vulkaninfo` is useful but not required by the pipeline. If SAPIEN reports a device-index error, run the evaluator with `CUDA_VISIBLE_DEVICES=1` and change the renderer device in the JSON to `cuda:0`, because visibility remaps the selected GPU.

## Start each policy on GPU 0

Create a separate environment per upstream repository. In each one, install its official dependencies and then only this project's protocol/adapter:

```bash
pip install -e /path/to/SimplerEnv --no-deps
```

### π0.5

For the primary comparison, `CONFIG_NAME` and `CHECKPOINT` must be the OpenPI configuration/checkpoint adapted on the common canonical Bridge data:

```bash
cd /path/to/SimplerEnv
CUDA_VISIBLE_DEVICES=0 python -m policy_servers.pi05.server \
  --config-name "$CONFIG_NAME" \
  --checkpoint "$CHECKPOINT" \
  --output-mode canonical \
  --confirm-canonical-adapted \
  --adaptation-dataset bridge_widowx \
  --adaptation-method action_head_or_adapter \
  --host 127.0.0.1 --port 8000
```

For a non-comparable π0.5 LIBERO diagnostic, use `--output-mode libero_normalized`, supply explicit `--translation-scale-m x,y,z`, `--rotation-scale-rad x,y,z`, and `--legacy-gripper-convention`, and add `--allow-profile-mismatch` to the evaluator. The adapter deliberately has no scale or gripper defaults because those values must be verified for the exact checkpoint.

### VPP

The action checkpoint should be a VPP action head trained on canonical Bridge labels; the video backbone can remain frozen:

```bash
cd /path/to/SimplerEnv
CUDA_VISIBLE_DEVICES=0 python -m policy_servers.vpp.server \
  --vpp-root /path/to/video-prediction-policy \
  --config /path/to/adapted_vpp_config.yaml \
  --video-model-path /path/to/svd-robot \
  --text-encoder-path /path/to/clip-vit-base-patch32 \
  --action-checkpoint /path/to/bridge_action_head.pt \
  --output-mode canonical \
  --confirm-canonical-adapted \
  --device cuda:0 \
  --host 127.0.0.1 --port 8000
```

Use `--output-mode calvin_normalized --allow-profile-mismatch` only for the released CALVIN diagnostic. That mode applies CALVIN's conventional translation ÷50 and Euler rotation ÷20 conversion before producing physical axis-angle deltas.

### Cosmos 3 Edge

This command refuses to start without the explicit compatibility acknowledgement. Do not point it at the released DROID joint-space checkpoint:

```bash
cd /path/to/SimplerEnv
CUDA_VISIBLE_DEVICES=0 python -m policy_servers.cosmos3.server \
  --cosmos-root /path/to/cosmos-framework \
  --checkpoint /path/to/exported_bridge_midtrain_checkpoint \
  --domain-name "$TRAINING_DOMAIN" \
  --confirm-cartesian-adapted \
  --action-chunk-size 16 \
  --host 127.0.0.1 --port 8000
```

The adapter sends current end-effector position/quaternion to the official `RobolabPolicyService`, receives an absolute Cartesian pose chunk, and converts consecutive poses to the canonical deltas. The default assumes its gripper output is an open fraction in `[0,1]`; select `--gripper-output canonical` only if the adapted checkpoint was trained directly with `[-1,+1]` canonical labels.

## Run the full comparison on GPU 1

Stop the previous model server, start the next one on the same port, and give each run a unique name. Do not run two policies simultaneously when measuring latency or memory.

```bash
conda activate simpler_env
cd /path/to/SimplerEnv
unset DISPLAY
python -m simpler_env.evaluation.remote_evaluator \
  --config configs/remote_eval/widowx_bridge.json \
  --server-url http://127.0.0.1:8000 \
  --run-name pi05-adapted
```

Repeat with `vpp-adapted` and `cosmos3-edge-adapted`. Each run produces:

```text
results/remote_eval/<run>/
├── evaluation_config.json
├── server_metadata.json
├── episodes.jsonl
├── summary.json
└── videos/<task>/*.mp4
```

`server_metadata.json` is part of the result, not optional bookkeeping: it records the checkpoint, upstream Git revision when available, output conversion, policy profile, adaptation declaration, and observation requirements. The evaluator rejects a mismatched profile before creating an environment.
