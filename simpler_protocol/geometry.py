"""Small NumPy-only rotation helpers used at model boundaries."""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def _normalize_quaternion_xyzw(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite vector of shape (4,)")
    norm = np.linalg.norm(q)
    if norm < _EPS:
        raise ValueError("zero quaternion is invalid")
    q = q / norm
    return -q if q[3] < 0.0 else q


def quaternion_multiply_xyzw(left, right) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quaternion_xyzw(left)
    x2, y2, z2, w2 = _normalize_quaternion_xyzw(right)
    return _normalize_quaternion_xyzw(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quaternion_inverse_xyzw(quaternion) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quaternion)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quaternion_xyzw_to_rotvec(quaternion) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quaternion)
    vector = np.array([x, y, z], dtype=np.float64)
    sine_half = np.linalg.norm(vector)
    if sine_half < _EPS:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(sine_half, np.clip(w, -1.0, 1.0))
    return vector / sine_half * angle


def rotvec_to_quaternion_xyzw(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3,) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation vector must be finite with shape (3,)")
    angle = np.linalg.norm(rotation)
    if angle < _EPS:
        return _normalize_quaternion_xyzw([rotation[0] / 2, rotation[1] / 2, rotation[2] / 2, 1.0])
    axis = rotation / angle
    return _normalize_quaternion_xyzw(np.r_[axis * np.sin(angle / 2.0), np.cos(angle / 2.0)])


def euler_xyz_to_rotvec(euler) -> np.ndarray:
    """Convert static XYZ Euler angles to an axis-angle rotation vector."""

    x, y, z = np.asarray(euler, dtype=np.float64)
    qx = np.array([np.sin(x / 2), 0.0, 0.0, np.cos(x / 2)])
    qy = np.array([0.0, np.sin(y / 2), 0.0, np.cos(y / 2)])
    qz = np.array([0.0, 0.0, np.sin(z / 2), np.cos(z / 2)])
    # transforms3d's default ``sxyz`` convention (used by the existing
    # SimplerEnv adapters) applies extrinsic X, then Y, then Z rotations.
    return quaternion_xyzw_to_rotvec(quaternion_multiply_xyzw(qz, quaternion_multiply_xyzw(qy, qx)))


def absolute_pose_chunk_to_deltas(actions, current_position, current_quaternion_xyzw):
    """Convert ``xyz + quaternion_xyzw + gripper`` absolute poses to 7-D deltas."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"absolute pose actions must have shape [T, >=8], got {actions.shape}")
    position = np.asarray(current_position, dtype=np.float64)
    quaternion = _normalize_quaternion_xyzw(current_quaternion_xyzw)
    result = []
    for action in actions:
        target_position = action[:3]
        target_quaternion = _normalize_quaternion_xyzw(action[3:7])
        delta_quaternion = quaternion_multiply_xyzw(target_quaternion, quaternion_inverse_xyzw(quaternion))
        delta = np.r_[target_position - position, quaternion_xyzw_to_rotvec(delta_quaternion), action[7]]
        result.append(delta)
        position = target_position
        quaternion = target_quaternion
    return np.asarray(result)
