"""Schemas and validation for the remote-policy protocol.

The protocol intentionally exposes only one action representation.  Model and
dataset-specific normalization must be undone in the policy server.
"""

from __future__ import annotations

import base64
import dataclasses
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = "1.0"
ACTION_ENCODING = "eef_delta_axis_angle_gripper_v1"
IMAGE_ENCODING = "zlib+base64"


class ProtocolError(ValueError):
    """Raised when a request or response violates the shared contract."""


def json_safe(value: Any) -> Any:
    """Recursively convert simulator/model values into deterministic JSON data.

    SAPIEN reset metadata may contain ``Pose`` objects. Keeping this conversion
    dependency-free lets the policy protocol and evaluator share it without
    importing SAPIEN in model-server environments.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if hasattr(value, "p") and hasattr(value, "q"):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__name__}",
            "position": json_safe(np.asarray(value.p)),
            "quaternion_wxyz": json_safe(np.asarray(value.q)),
        }
    return {
        "__type__": f"{type(value).__module__}.{type(value).__name__}",
        "repr": repr(value),
    }


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ProtocolError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ProtocolError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class CanonicalAction:
    """One physical Cartesian controller action.

    ``translation`` is a world/controller-frame delta in metres, ``rotation``
    is an axis-angle rotation vector in radians, and ``gripper_open`` uses +1
    for open and -1 for closed.  SimplerEnv maps the last value to each robot's
    native convention at the final boundary.
    """

    translation: np.ndarray
    rotation: np.ndarray
    gripper_open: float
    terminate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _finite_vector(self.translation, 3, "translation"))
        object.__setattr__(self, "rotation", _finite_vector(self.rotation, 3, "rotation"))
        gripper = float(self.gripper_open)
        if not np.isfinite(gripper) or not -1.0 <= gripper <= 1.0:
            raise ProtocolError("gripper_open must be finite and within [-1, 1]")
        object.__setattr__(self, "gripper_open", gripper)
        object.__setattr__(self, "terminate", bool(self.terminate))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalAction:
        if not isinstance(value, Mapping):
            raise ProtocolError("each action must be a JSON object")
        return cls(
            translation=value.get("translation"),
            rotation=value.get("rotation"),
            gripper_open=value.get("gripper_open"),
            terminate=value.get("terminate", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "gripper_open": self.gripper_open,
            "terminate": self.terminate,
        }


def encode_image(image: np.ndarray) -> dict[str, Any]:
    """Encode an HWC uint8 image without adding an image-codec dependency."""

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ProtocolError(f"image must be HWC uint8 with 1, 3, or 4 channels, got {array.shape}/{array.dtype}")
    compressed = zlib.compress(array.tobytes(), level=1)
    return {
        "encoding": IMAGE_ENCODING,
        "dtype": "uint8",
        "shape": list(array.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def decode_image(payload: Mapping[str, Any], max_pixels: int = 32_000_000) -> np.ndarray:
    if payload.get("encoding") != IMAGE_ENCODING or payload.get("dtype") != "uint8":
        raise ProtocolError(f"unsupported image encoding: {payload.get('encoding')}/{payload.get('dtype')}")
    shape = payload.get("shape")
    if not isinstance(shape, Sequence) or len(shape) != 3:
        raise ProtocolError("encoded image shape must have three dimensions")
    try:
        height, width, channels = (int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("encoded image shape is invalid") from exc
    if height <= 0 or width <= 0 or channels not in (1, 3, 4) or height * width > max_pixels:
        raise ProtocolError(f"encoded image shape is unsafe or invalid: {shape}")
    expected = height * width * channels
    try:
        compressed = base64.b64decode(payload["data"], validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected + 1)
        if len(raw) > expected or decompressor.unconsumed_tail:
            raise ProtocolError("decoded image exceeds its declared shape")
        raw += decompressor.flush(expected - len(raw) + 1)
    except (KeyError, ValueError, zlib.error) as exc:
        raise ProtocolError("invalid image data") from exc
    if len(raw) != expected:
        raise ProtocolError(f"decoded image has {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels).copy()


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ProtocolError("policy metadata must be a JSON object")
    required = ("protocol_version", "model_id", "checkpoint_id", "action_encoding", "policy_profile")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ProtocolError(f"policy metadata is missing: {', '.join(missing)}")
    if metadata["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: client={PROTOCOL_VERSION}, server={metadata['protocol_version']}"
        )
    if metadata["action_encoding"] != ACTION_ENCODING:
        raise ProtocolError(f"unsupported action encoding: {metadata['action_encoding']}")
    return dict(metadata)


def make_metadata(
    *,
    model_id: str,
    checkpoint_id: str,
    policy_profile: str,
    output_mode: str,
    wrist_image: str = "optional",
    adaptation: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "action_encoding": ACTION_ENCODING,
        "policy_profile": policy_profile,
        "output_mode": output_mode,
        "action_units": {"translation": "metres", "rotation": "axis_angle_radians"},
        "gripper_convention": "+1_open_-1_close",
        "observation": {"primary_image": "required", "wrist_image": wrist_image},
        "adaptation": dict(adaptation or {}),
    }
    metadata.update(dict(extra or {}))
    return validate_metadata(metadata)
