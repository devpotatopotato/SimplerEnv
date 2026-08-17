"""Serve an OpenPI π0.5 checkpoint through the canonical SimplerEnv protocol."""

from __future__ import annotations

import argparse

import numpy as np

from policy_servers.common import (
    canonical_actions,
    decode_request,
    git_revision,
    zero_wrist_like,
)
from simpler_protocol import (
    PolicyBackend,
    euler_xyz_to_rotvec,
    quaternion_xyzw_to_rotvec,
    serve_backend,
)
from simpler_protocol.schema import make_metadata


class Pi05Backend(PolicyBackend):
    def __init__(self, args):
        if args.output_mode == "canonical" and not args.confirm_canonical_adapted:
            raise ValueError(
                "Canonical mode requires a Bridge/WidowX-adapted checkpoint. "
                "Pass --confirm-canonical-adapted after verifying its output labels and transforms."
            )
        if args.output_mode == "libero_normalized" and (
            args.translation_scale_m is None or args.rotation_scale_rad is None or args.legacy_gripper_convention is None
        ):
            raise ValueError(
                "libero_normalized mode requires explicit --translation-scale-m, --rotation-scale-rad, "
                "and --legacy-gripper-convention values for the selected checkpoint"
            )
        # Imports stay inside this process so OpenPI/JAX never enter SimplerEnv.
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        if args.config_name == "pi05_simpler_bridge_lora":
            # The training config lives in this repository so the pinned
            # upstream checkout stays unmodified and auditable.
            from simpler_training.openpi_config import register_config

            register_config()

        config = openpi_config.get_config(args.config_name)
        checkpoint = download.maybe_download(args.checkpoint)
        self.policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device=args.device)
        self.output_mode = args.output_mode
        self.translation_scale = (
            None if args.translation_scale_m is None else np.asarray(args.translation_scale_m, dtype=np.float64)
        )
        self.rotation_scale = (
            None if args.rotation_scale_rad is None else np.asarray(args.rotation_scale_rad, dtype=np.float64)
        )
        self.legacy_gripper_plus_one_open = args.legacy_gripper_convention == "plus_one_open"
        profile = (
            args.policy_profile
            if args.policy_profile
            else "simpler_widowx_cartesian_v1"
            if args.output_mode == "canonical"
            else "cross_embodiment_unadapted"
        )
        self._metadata = make_metadata(
            model_id="pi0.5",
            checkpoint_id=args.checkpoint,
            policy_profile=profile,
            output_mode=args.output_mode,
            wrist_image="zero-filled when absent",
            adaptation=(
                {"dataset": args.adaptation_dataset, "method": args.adaptation_method}
                if args.output_mode == "canonical"
                else {"dataset": "libero", "method": "none; output conversion only"}
            ),
            extra={
                "comparison_group": args.comparison_group,
                "openpi_config": args.config_name,
                "upstream_revision": git_revision(openpi_config.__file__),
            },
        )

    @property
    def metadata(self):
        return self._metadata

    def reset(self, request):
        # OpenPI policies are stateless at this boundary, but keep compatibility
        # with policy implementations that expose an episode reset.
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        seed = int(request.get("seed", 0))
        # Current OpenPI uses a JAX key for JAX checkpoints and PyTorch's
        # process RNG for safetensors checkpoints. Reset the applicable one.
        if hasattr(self.policy, "_rng"):
            import jax

            self.policy._rng = jax.random.key(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def predict(self, request):
        instruction, images, state, horizon = decode_request(request)
        wrist = images.get("wrist", zero_wrist_like(images["primary"]))
        eef_position = np.asarray(state["eef_position"], dtype=np.float32).reshape(3)
        eef_quaternion = np.asarray(state["eef_quaternion_xyzw"], dtype=np.float64).reshape(4)
        gripper_qpos = np.asarray(state.get("gripper_qpos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if gripper_qpos.size < 2:
            gripper_qpos = np.pad(gripper_qpos, (0, 2 - gripper_qpos.size))
        libero_state = np.r_[eef_position, quaternion_xyzw_to_rotvec(eef_quaternion), gripper_qpos[:2]].astype(
            np.float32
        )
        output = self.policy.infer(
            {
                "observation/image": images["primary"],
                "observation/wrist_image": wrist,
                "observation/state": libero_state,
                "prompt": instruction,
            }
        )
        actions = np.asarray(output["actions"], dtype=np.float64)
        if self.output_mode == "libero_normalized":
            converted = np.zeros_like(actions[..., :7])
            converted[..., :3] = actions[..., :3] * self.translation_scale
            flat_input = actions.reshape(-1, actions.shape[-1])
            flat_output = converted.reshape(-1, 7)
            for row_index, action in enumerate(flat_input):
                flat_output[row_index, 3:6] = euler_xyz_to_rotvec(action[3:6] * self.rotation_scale)
                flat_output[row_index, 6] = action[6]
            actions = converted
        return canonical_actions(
            actions,
            horizon,
            gripper_plus_one_is_open=(True if self.output_mode == "canonical" else self.legacy_gripper_plus_one_open),
        )


def _triplet(value: str):
    values = [float(item) for item in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", required=True, help="OpenPI config used to train the checkpoint")
    parser.add_argument("--device", default="cuda:0", help="device for OpenPI PyTorch checkpoints")
    parser.add_argument("--output-mode", choices=("canonical", "libero_normalized"), default="canonical")
    parser.add_argument("--confirm-canonical-adapted", action="store_true")
    parser.add_argument("--translation-scale-m", type=_triplet)
    parser.add_argument("--rotation-scale-rad", type=_triplet)
    parser.add_argument(
        "--legacy-gripper-convention",
        choices=("plus_one_open", "plus_one_close"),
    )
    parser.add_argument("--policy-profile")
    parser.add_argument("--adaptation-dataset", default="bridge_widowx")
    parser.add_argument("--adaptation-method", default="action_head_or_adapter")
    parser.add_argument("--comparison-group", default="shared_bridge_adaptation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_backend(Pi05Backend(args), args.host, args.port)


if __name__ == "__main__":
    main()
