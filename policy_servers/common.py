"""Utilities shared by model-specific server processes."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from simpler_protocol import CanonicalAction, ProtocolError, decode_image


def decode_request(request: Mapping[str, Any]):
    instruction = request.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProtocolError("instruction must be a non-empty string")
    images_payload = request.get("images")
    state_payload = request.get("state")
    if not isinstance(images_payload, Mapping) or "primary" not in images_payload:
        raise ProtocolError("images.primary is required")
    if not isinstance(state_payload, Mapping):
        raise ProtocolError("state must be an object")
    images = {name: decode_image(value) for name, value in images_payload.items()}
    state = {name: np.asarray(value, dtype=np.float32) for name, value in state_payload.items()}
    horizon = int(request.get("requested_horizon", 1))
    if not 1 <= horizon <= 256:
        raise ProtocolError("requested_horizon must be in [1, 256]")
    return instruction, images, state, horizon


def canonical_actions(array, horizon: int, *, gripper_plus_one_is_open: bool = True):
    actions = np.asarray(array, dtype=np.float64)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ProtocolError(f"model output must have shape [T, >=7], got {actions.shape}")
    result = []
    for action in actions[:horizon]:
        gripper = float(action[6]) * (1.0 if gripper_plus_one_is_open else -1.0)
        result.append(CanonicalAction(action[:3], action[3:6], np.clip(gripper, -1.0, 1.0)))
    return result


def zero_wrist_like(primary: np.ndarray) -> np.ndarray:
    return np.zeros_like(primary)


def git_revision(path: str | Path) -> str | None:
    """Best-effort source revision for reproducibility metadata."""

    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                return None
    return None
