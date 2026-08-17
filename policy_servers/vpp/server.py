"""Serve a VPP checkpoint through the canonical SimplerEnv protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from policy_servers.common import (
    canonical_actions,
    decode_request,
    git_revision,
    zero_wrist_like,
)
from simpler_protocol import PolicyBackend, euler_xyz_to_rotvec, serve_backend
from simpler_protocol.schema import make_metadata


class VPPBackend(PolicyBackend):
    def __init__(self, args):
        if args.output_mode == "canonical" and not args.confirm_canonical_adapted:
            raise ValueError(
                "Canonical mode requires a Bridge/WidowX-adapted action head. "
                "Pass --confirm-canonical-adapted after verifying its output labels."
            )
        # VPP is an upstream research repository, not a stable wheel.  Import
        # the explicit checkout selected by the operator.
        sys.path.insert(0, str(Path(args.vpp_root).resolve()))
        import hydra
        import torch
        from omegaconf import OmegaConf
        from torch.nn import functional

        self.torch = torch
        self.functional = functional
        config = OmegaConf.load(args.config)
        config.seed = int(args.seed)
        config.model.pretrained_model_path = args.video_model_path
        if args.text_encoder_path:
            config.model.text_encoder_path = args.text_encoder_path
        target = str(config.model.get("_target_", ""))
        if target.startswith("models."):
            config.model._target_ = "policy_models." + target[len("models.") :]
        # The released evaluation YAML uses this older boolean name while the
        # current class constructor expects a string-valued ``use_Former``.
        if "use_3d_Former" in config.model and "use_Former" not in config.model:
            config.model.use_Former = "3d" if bool(config.model.use_3d_Former) else "2d"
            del config.model["use_3d_Former"]
        self.model = hydra.utils.instantiate(config.model)
        state = torch.load(args.action_checkpoint, map_location="cpu")
        state_dict = state.get("model", state.get("state_dict", state))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=args.strict_checkpoint)
        if not args.strict_checkpoint and (missing or unexpected):
            print(f"VPP checkpoint non-strict load: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        self.device = torch.device(args.device)
        self.model = self.model.to(self.device).eval()
        process_device = getattr(self.model, "process_device", None)
        if callable(process_device):
            process_device()
        self.output_mode = args.output_mode
        adapter = config.get("action_adapter", {})
        self.action_adapter_kind = str(adapter.get("kind", "identity"))
        self.action_output_scale = np.asarray(adapter.get("output_scale", [1.0] * 7), dtype=np.float64)
        self.clip_normalized = bool(adapter.get("clip_normalized", False))
        if self.action_adapter_kind not in ("identity", "normalized_symmetric"):
            raise ValueError(f"unsupported VPP action_adapter.kind: {self.action_adapter_kind}")
        if self.action_output_scale.shape != (7,) or np.any(self.action_output_scale <= 0):
            raise ValueError("VPP action_adapter.output_scale must contain seven positive values")
        if args.output_mode == "canonical" and self.action_adapter_kind == "normalized_symmetric":
            missing_policy = [
                name for name in missing if name.startswith(("Video_Former.", "model."))
            ]
            if missing_policy or unexpected:
                raise ValueError(
                    "adapted VPP checkpoint is incomplete or incompatible: "
                    f"missing_policy={missing_policy[:10]}, unexpected={unexpected[:10]}"
                )
        profile = (
            args.policy_profile
            if args.policy_profile
            else "simpler_widowx_cartesian_v1"
            if args.output_mode == "canonical"
            else "cross_embodiment_unadapted"
        )
        self._metadata = make_metadata(
            model_id="vpp",
            checkpoint_id=args.action_checkpoint,
            policy_profile=profile,
            output_mode=args.output_mode,
            wrist_image="zero-filled when absent",
            adaptation=(
                {"dataset": args.adaptation_dataset, "method": args.adaptation_method}
                if args.output_mode == "canonical"
                else {"dataset": "calvin", "method": "none; output conversion only"}
            ),
            extra={
                "comparison_group": args.comparison_group,
                "video_model_path": args.video_model_path,
                "image_size": args.image_size,
                "action_adapter": self.action_adapter_kind,
                "action_output_scale": self.action_output_scale.tolist(),
                "upstream_revision": git_revision(args.vpp_root),
            },
        )
        self.image_size = args.image_size

    @property
    def metadata(self):
        return self._metadata

    def reset(self, request):
        seed = int(request.get("seed", 0))
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _image_tensor(self, image):
        tensor = self.torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        tensor = self.functional.interpolate(
            tensor[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0]
        return ((tensor - 0.5) / 0.5)[None, None].to(self.device)

    def predict(self, request):
        instruction, images, _, horizon = decode_request(request)
        wrist = images.get("wrist", zero_wrist_like(images["primary"]))
        observation = {
            "rgb_obs": {
                "rgb_static": self._image_tensor(images["primary"]),
                "rgb_gripper": self._image_tensor(wrist),
            }
        }
        with self.torch.inference_mode():
            actions = self.model.eval_forward(observation, {"lang_text": instruction})
        actions = actions.detach().float().cpu().numpy()
        if self.output_mode == "canonical" and self.action_adapter_kind == "normalized_symmetric":
            actions = np.asarray(actions, dtype=np.float64)[..., :7]
            if self.clip_normalized:
                actions = np.clip(actions, -1.0, 1.0)
            actions = actions * self.action_output_scale
        elif self.output_mode == "calvin_normalized":
            actions = np.asarray(actions, dtype=np.float64)
            converted = np.zeros_like(actions[..., :7])
            flat_input = actions.reshape(-1, actions.shape[-1])
            flat_output = converted.reshape(-1, 7)
            for row_index, action in enumerate(flat_input):
                flat_output[row_index, :3] = action[:3] / 50.0
                flat_output[row_index, 3:6] = euler_xyz_to_rotvec(action[3:6] / 20.0)
                flat_output[row_index, 6] = action[6]
            actions = converted
        return canonical_actions(actions, horizon, gripper_plus_one_is_open=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpp-root", required=True, help="checkout of roboterax/video-prediction-policy")
    parser.add_argument("--config", required=True, help="VPP YAML containing the model section")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--text-encoder-path")
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--output-mode", choices=("canonical", "calvin_normalized"), default="canonical")
    parser.add_argument("--confirm-canonical-adapted", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy-profile")
    parser.add_argument("--adaptation-dataset", default="bridge_widowx")
    parser.add_argument("--adaptation-method", default="frozen_video_backbone_action_head")
    parser.add_argument("--comparison-group", default="shared_bridge_adaptation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_backend(VPPBackend(args), args.host, args.port)


if __name__ == "__main__":
    main()
