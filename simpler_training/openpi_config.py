"""OpenPI configuration for canonical Bridge/WidowX LoRA fine-tuning."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

CONFIG_NAME = "pi05_simpler_bridge_lora"
DEFAULT_REPO_ID = "local/simpler_bridge_train"


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected a three-dimensional image, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return image.astype(np.uint8, copy=False)


def build_config(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    exp_name: str = "bridge_lora",
    assets_base_dir: str = "./assets",
    checkpoint_base_dir: str = "./checkpoints",
    train_steps: int = 30_000,
    batch_size: int = 32,
    num_workers: int = 4,
    save_interval: int = 1_000,
    resume: bool = False,
):
    # These imports occur only in the isolated OpenPI process.
    from openpi.models import model as model_api
    from openpi.models import pi0_config
    from openpi.training import config as config_api
    from openpi.training import optimizer
    from openpi.training import weight_loaders
    import openpi.transforms as transforms

    @dataclasses.dataclass(frozen=True)
    class BridgeInputs(transforms.DataTransformFn):
        model_type: model_api.ModelType

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            base_image = _parse_image(data["observation/image"])
            missing_wrist = np.zeros_like(base_image)
            result: dict[str, Any] = {
                "state": np.asarray(data["observation/state"], dtype=np.float32),
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": missing_wrist,
                    "right_wrist_0_rgb": missing_wrist.copy(),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                },
            }
            if "actions" in data:
                result["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if "prompt" in data:
                result["prompt"] = data["prompt"]
            return result

    @dataclasses.dataclass(frozen=True)
    class BridgeOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"actions": np.asarray(data["actions"])[..., :7]}

    @dataclasses.dataclass(frozen=True)
    class LeRobotBridgeDataConfig(config_api.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = transforms.Group(
                inputs=[BridgeInputs(model_type=model_config.model_type)],
                outputs=[BridgeOutputs()],
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=config_api.ModelTransformFactory()(model_config),
            )

    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=16,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    config = config_api.TrainConfig(
        name=CONFIG_NAME,
        project_name="simpler-remote-eval",
        exp_name=exp_name,
        model=model,
        data=LeRobotBridgeDataConfig(
            repo_id=repo_id,
            base_config=config_api.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(10_000, max(100, train_steps // 3)),
            peak_lr=5e-5,
            decay_steps=max(train_steps, 10_000),
            decay_lr=5e-5,
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        num_train_steps=train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        save_interval=save_interval,
        keep_period=max(save_interval, 5_000),
        resume=resume,
        wandb_enabled=os.environ.get("WANDB_MODE", "disabled") not in ("disabled", "offline"),
        policy_metadata={
            "policy_profile": "simpler_widowx_cartesian_v1",
            "action_encoding": "eef_delta_axis_angle_gripper_v1",
            "missing_wrist": "zero-filled-and-masked",
        },
        # Explicitly retain the default non-FSDP layout. With two visible GPUs,
        # OpenPI data-parallelizes one replica per GPU.
        fsdp_devices=1,
    )
    return config


def register_config(config=None):
    """Register this external config in OpenPI's process-local registry."""

    from openpi.training import config as config_api

    config = config or build_config()
    config_api._CONFIGS_DICT[CONFIG_NAME] = config
    return config
