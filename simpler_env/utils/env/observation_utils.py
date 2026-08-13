import numpy as np


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _robot_uid(env):
    base = _base_env(env)
    return str(getattr(base, "robot_uid", getattr(env, "robot_uid", "")))


def default_camera_name(env):
    robot_uid = _robot_uid(env)
    if "google_robot" in robot_uid:
        return "overhead_camera"
    if "widowx" in robot_uid:
        return "3rd_view_camera"
    raise NotImplementedError(f"no default policy camera for robot {robot_uid!r}")


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        camera_name = default_camera_name(env)
    return obs["image"][camera_name]["rgb"]


def _uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"policy image must be HWC RGB, got {image.shape}")
    image = image[..., :3]
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        return np.ascontiguousarray(np.clip(image * scale, 0, 255).astype(np.uint8))
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def get_policy_observation(env, obs, primary_camera_name=None, wrist_camera_name=None):
    """Build the model-independent observation sent to remote policy servers.

    Quaternion state is serialized as XYZW even though SAPIEN exposes WXYZ.
    All vectors remain in SimplerEnv's native robot/world controller frames.
    """

    base = _base_env(env)
    primary_camera_name = primary_camera_name or default_camera_name(env)
    images = {"primary": _uint8_rgb(obs["image"][primary_camera_name]["rgb"])}
    if wrist_camera_name is not None:
        if wrist_camera_name not in obs["image"]:
            raise KeyError(f"wrist camera {wrist_camera_name!r} is absent from the observation")
        images["wrist"] = _uint8_rgb(obs["image"][wrist_camera_name]["rgb"])

    robot = base.agent.robot
    qpos = np.asarray(robot.get_qpos(), dtype=np.float64)
    qvel = np.asarray(robot.get_qvel(), dtype=np.float64)
    tcp_pose = base.tcp.pose
    eef_position = np.asarray(tcp_pose.p, dtype=np.float64)
    quaternion_wxyz = np.asarray(tcp_pose.q, dtype=np.float64)
    eef_quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    try:
        gripper_closedness = float(base.agent.get_gripper_closedness())
    except (AttributeError, NotImplementedError):
        gripper_closedness = float("nan")

    state = {
        "qpos": qpos,
        "qvel": qvel,
        "eef_position": eef_position,
        "eef_quaternion_xyzw": eef_quaternion_xyzw,
        "gripper_qpos": qpos[-2:] if qpos.size >= 2 else qpos,
    }
    if np.isfinite(gripper_closedness):
        state["gripper_closedness"] = gripper_closedness
    return {
        "images": images,
        "state": state,
        "robot_uid": _robot_uid(env),
        "camera_names": {"primary": primary_camera_name, "wrist": wrist_camera_name},
    }
