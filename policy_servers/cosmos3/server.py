"""Serve a Cartesian-adapted Cosmos 3 Edge checkpoint to SimplerEnv.

The public Cosmos3-Edge-Policy-DROID checkpoint is intentionally unsupported:
it emits absolute DROID joint positions, which cannot be fairly applied to a
WidowX Cartesian controller.  Adapt/post-train a checkpoint with the upstream
``midtrain`` Cartesian action space before using this server.
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


class Cosmos3Backend(PolicyBackend):
    def __init__(self, args):
        if not args.confirm_cartesian_adapted:
            raise ValueError(
                "Refusing to serve a potentially incompatible joint-space checkpoint. "
                "Pass --confirm-cartesian-adapted only after training/exporting a Cosmos checkpoint with action_space=midtrain."
            )
        sys.path.insert(0, str(Path(args.cosmos_root).resolve()))
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
        self.gripper_output = args.gripper_output
        self._metadata = make_metadata(
            model_id="cosmos3-edge-policy",
            checkpoint_id=args.checkpoint,
            policy_profile=args.policy_profile,
            output_mode="cosmos_midtrain_absolute_pose_to_canonical_delta",
            wrist_image="unused",
            adaptation={"dataset": args.adaptation_dataset, "method": args.adaptation_method},
            extra={
                "cosmos_action_space": "midtrain",
                "domain_name": args.domain_name,
                "gripper_output": args.gripper_output,
                "upstream_revision": git_revision(args.cosmos_root),
            },
        )

    @property
    def metadata(self):
        return self._metadata

    def reset(self, request):
        # Upstream service advances an RNG across requests. Reset it per episode
        # so task seeds remain comparable across independent evaluation runs.
        if not self.service.cfg.deterministic_seed:
            self.service._rng = np.random.default_rng(int(request.get("seed", self.service.cfg.seed)))

    def predict(self, request):
        instruction, images, state, horizon = decode_request(request)
        eef_position = np.asarray(state["eef_position"], dtype=np.float32).reshape(3)
        eef_quaternion = np.asarray(state["eef_quaternion_xyzw"], dtype=np.float32).reshape(4)
        closedness = float(np.asarray(state.get("gripper_closedness", 0.0)).reshape(()))
        open_fraction = float(np.clip(1.0 - closedness, 0.0, 1.0))
        output = self.service.infer(
            {
                "prompt": instruction,
                "observation/image": images["primary"],
                "observation/eef_pos": eef_position,
                "observation/eef_quat": eef_quaternion,
                "observation/gripper_position": open_fraction,
            }
        )
        deltas = absolute_pose_chunk_to_deltas(output["action"], eef_position, eef_quaternion)
        result = []
        for delta in deltas[:horizon]:
            gripper = float(delta[6])
            if self.gripper_output == "open_fraction":
                gripper = 2.0 * np.clip(gripper, 0.0, 1.0) - 1.0
            result.append(CanonicalAction(delta[:3], delta[3:6], np.clip(gripper, -1.0, 1.0)))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-root", required=True, help="checkout of NVIDIA/cosmos-framework")
    parser.add_argument("--checkpoint", required=True, help="consolidated Cartesian-adapted checkpoint")
    parser.add_argument("--domain-name", required=True, help="domain ID used during checkpoint post-training")
    parser.add_argument("--confirm-cartesian-adapted", action="store_true")
    parser.add_argument("--policy-profile", default="simpler_widowx_cartesian_v1")
    parser.add_argument("--adaptation-dataset", default="bridge_widowx")
    parser.add_argument("--adaptation-method", default="frozen_backbone_action_adapter")
    parser.add_argument("--gripper-output", choices=("open_fraction", "canonical"), default="open_fraction")
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_backend(Cosmos3Backend(args), args.host, args.port)


if __name__ == "__main__":
    main()
