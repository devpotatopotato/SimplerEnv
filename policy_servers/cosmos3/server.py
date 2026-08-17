"""Serve ``nvidia/Cosmos3-Edge`` through the canonical SimplerEnv protocol.

Cosmos3-Edge contains a native ``bridge_orig_lerobot`` 10-D action domain. Its
actions are quantile-normalized, backward-framewise flange-pose deltas in the
Bridge OpenCV tool frame. This adapter restores the official normalization and
view/FPS settings, converts between that tool frame and SimplerEnv's WidowX TCP
frame, and exposes only canonical 7-D controller deltas.

The DROID policy checkpoints remain intentionally unsupported: their absolute
Franka joint targets cannot be mapped to WidowX Cartesian controls without a
learned embodiment adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from policy_servers.common import decode_request, git_revision
from simpler_protocol import (
    CanonicalAction,
    PolicyBackend,
    absolute_pose_chunk_to_deltas,
    serve_backend,
)
from simpler_protocol.schema import make_metadata

_DEFAULT_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)
_BRIDGE_TO_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
_TCP_TO_FLANGE = np.eye(4, dtype=np.float64)
_TCP_TO_FLANGE[0, 3] = -0.093575


def _homogeneous_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


# Matches BridgeOrigLeRobotDataset exactly:
# R_fk = R_state @ DEFAULT_ROTATION; then TCP->flange; then OpenCV tool frame.
_SIMPLER_TCP_TO_COSMOS_TOOL = (
    _homogeneous_rotation(_DEFAULT_ROTATION)
    @ _TCP_TO_FLANGE
    @ _homogeneous_rotation(_BRIDGE_TO_OPENCV)
)
_COSMOS_TOOL_TO_SIMPLER_TCP = np.linalg.inv(_SIMPLER_TCP_TO_COSMOS_TOOL)


def _quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    left, _, right = np.linalg.svd(matrix)
    matrix = left @ right
    if np.linalg.det(matrix) < 0:
        left[:, -1] *= -1
        matrix = left @ right
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return -quaternion if quaternion[3] < 0 else quaternion


def _pose_matrix(position, quaternion_xyzw) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quaternion_xyzw_to_matrix(quaternion_xyzw)
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def simpler_pose_to_cosmos(position, quaternion_xyzw) -> tuple[np.ndarray, np.ndarray]:
    pose = _pose_matrix(position, quaternion_xyzw) @ _SIMPLER_TCP_TO_COSMOS_TOOL
    return pose[:3, 3], _matrix_to_quaternion_xyzw(pose[:3, :3])


def cosmos_absolute_chunk_to_canonical(actions, current_position, current_quaternion_xyzw) -> np.ndarray:
    """Convert absolute Bridge-tool poses from Cosmos into Simpler TCP deltas."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"Cosmos absolute actions must have shape [T, >=8], got {actions.shape}")
    raw_absolute = []
    for action in actions:
        cosmos_pose = _pose_matrix(action[:3], action[3:7])
        simpler_pose = cosmos_pose @ _COSMOS_TOOL_TO_SIMPLER_TCP
        # RobolabPolicyService flips the last channel for its DROID API. The
        # native Bridge dataset stores 1=open, so undo that flip here.
        bridge_open_fraction = float(np.clip(1.0 - action[7], 0.0, 1.0))
        raw_absolute.append(
            np.r_[
                simpler_pose[:3, 3],
                _matrix_to_quaternion_xyzw(simpler_pose[:3, :3]),
                bridge_open_fraction,
            ]
        )
    return absolute_pose_chunk_to_deltas(raw_absolute, current_position, current_quaternion_xyzw)


class Cosmos3Backend(PolicyBackend):
    def __init__(self, args):
        if not args.confirm_native_bridge_domain:
            raise ValueError(
                "Pass --confirm-native-bridge-domain only for Cosmos3-Edge with "
                "domain_name=bridge_orig_lerobot; DROID joint-space policies are unsupported."
            )
        if "DROID" in args.checkpoint.upper():
            raise ValueError("Cosmos DROID checkpoints emit Franka joint targets and are incompatible with WidowX")
        if args.domain_name != "bridge_orig_lerobot":
            raise ValueError("this adapter is valid only for Cosmos's bridge_orig_lerobot action domain")

        sys.path.insert(0, str(Path(args.cosmos_root).resolve()))
        from cosmos_framework.data.generator.action.action_processing import (
            resolve_action_normalization,
        )
        from cosmos_framework.data.generator.action.datasets.bridge_orig_lerobot_dataset import (
            BridgeOrigLeRobotDataset,
        )
        from cosmos_framework.data.generator.action.transforms import (
            ActionTransformPipeline,
        )
        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService,
            RobolabServerArgs,
        )

        service_args = RobolabServerArgs(
            checkpoint_path=args.checkpoint,
            domain_name=args.domain_name,
            action_space="midtrain",
            action_dim=10,
            action_chunk_size=args.action_chunk_size,
            conditioning_fps=5.0,
            use_state=False,
            history_length=0,
            seed=args.seed,
            deterministic_seed=args.deterministic_seed,
            guidance=args.guidance,
            num_steps=args.num_steps,
            shift=args.shift,
            resolution=args.resolution,
            image_height=args.image_height,
            image_width=args.image_width,
            decode_video=False,
        )
        self.service = RobolabPolicyService(service_args)

        stats = BridgeOrigLeRobotDataset.load_action_stats()
        normalizer = resolve_action_normalization("quantile", stats, apply_forward_clamp=False)
        max_action_dim = int(getattr(self.service.model.config, "max_action_dim", 64))
        transform = ActionTransformPipeline(max_action_dim=max_action_dim, cfg_dropout_rate=0.0)

        def bridge_transform(sample, resolution):
            # The official Bridge pretraining samples use one ego view, not the
            # DROID server's three-camera concat-view prompt.
            sample["viewpoint"] = "ego_view"
            sample.pop("additional_view_description", None)
            return transform(sample, resolution, action_normalizer=normalizer)

        self.service._transform = bridge_transform
        self._metadata = make_metadata(
            model_id="nvidia/Cosmos3-Edge",
            checkpoint_id=args.checkpoint,
            policy_profile=args.policy_profile,
            output_mode="cosmos_bridge_relative_pose_to_canonical_delta",
            wrist_image="unused",
            adaptation={"dataset": args.adaptation_dataset, "method": args.adaptation_method},
            extra={
                "comparison_group": args.comparison_group,
                "cosmos_action_space": "midtrain",
                "domain_name": args.domain_name,
                "action_normalization": "official_bridge_quantile",
                "pose_frame": "bridge_flange_opencv",
                "conditioning_fps": 5,
                "upstream_revision": git_revision(args.cosmos_root),
            },
        )

    @property
    def metadata(self):
        return self._metadata

    def reset(self, request):
        if not self.service.cfg.deterministic_seed:
            self.service._rng = np.random.default_rng(int(request.get("seed", self.service.cfg.seed)))

    def predict(self, request):
        instruction, images, state, horizon = decode_request(request)
        eef_position = np.asarray(state["eef_position"], dtype=np.float64).reshape(3)
        eef_quaternion = np.asarray(state["eef_quaternion_xyzw"], dtype=np.float64).reshape(4)
        model_position, model_quaternion = simpler_pose_to_cosmos(eef_position, eef_quaternion)
        closedness = float(np.asarray(state.get("gripper_closedness", 0.0)).reshape(()))
        output = self.service.infer(
            {
                "prompt": instruction,
                "observation/image": images["primary"],
                "observation/eef_pos": model_position.astype(np.float32),
                "observation/eef_quat": model_quaternion.astype(np.float32),
                "observation/gripper_position": float(np.clip(1.0 - closedness, 0.0, 1.0)),
            }
        )
        deltas = cosmos_absolute_chunk_to_canonical(output["action"], eef_position, eef_quaternion)
        return [
            CanonicalAction(delta[:3], delta[3:6], 2.0 * np.clip(delta[6], 0.0, 1.0) - 1.0)
            for delta in deltas[:horizon]
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-root", required=True, help="checkout of NVIDIA/cosmos-framework")
    parser.add_argument(
        "--checkpoint",
        default="Cosmos3-Edge",
        help="Cosmos registry name (Cosmos3-Edge) or a compatible local/exported checkpoint",
    )
    parser.add_argument("--domain-name", default="bridge_orig_lerobot")
    parser.add_argument("--confirm-native-bridge-domain", action="store_true")
    parser.add_argument("--policy-profile", default="simpler_widowx_cartesian_v1")
    parser.add_argument("--adaptation-dataset", default="upstream_bridge_original")
    parser.add_argument("--adaptation-method", default="native_bridge_action_head")
    parser.add_argument("--comparison-group", default="native_pretrained_bridge")
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_backend(Cosmos3Backend(args), args.host, args.port)


if __name__ == "__main__":
    main()
