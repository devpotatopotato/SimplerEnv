"""Fair, reproducible evaluation of remote policies in prepackaged SimplerEnv tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import simpler_env
from simpler_env.policies.remote_policy import RemotePolicy
from simpler_env.utils.env.observation_utils import get_policy_observation
from simpler_protocol import CanonicalAction


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class SafetyConfig:
    translation_limit_m: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    rotation_limit_rad: float = 0.25
    mode: str = "clip"

    @classmethod
    def from_dict(cls, value):
        value = value or {}
        result = cls(**value)
        if len(result.translation_limit_m) != 3 or any(float(item) <= 0 for item in result.translation_limit_m):
            raise ValueError("safety.translation_limit_m must contain three positive values")
        if result.rotation_limit_rad <= 0 or result.mode not in ("clip", "reject"):
            raise ValueError("safety rotation limit must be positive and mode must be clip or reject")
        return result


@dataclass
class EvaluationConfig:
    tasks: list[str]
    episodes_per_task: int = 10
    base_seed: int = 0
    max_episode_steps: int = 120
    request_horizon: int = 16
    execution_horizon: int = 1
    server_timeout_seconds: float = 180.0
    required_policy_profile: str | None = "simpler_widowx_cartesian_v1"
    primary_camera: str | None = None
    wrist_camera: str | None = None
    save_video: bool = True
    video_fps: int = 5
    output_dir: str = "results/remote_eval"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @classmethod
    def load(cls, path: str | os.PathLike[str]):
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        value["safety"] = SafetyConfig.from_dict(value.get("safety"))
        result = cls(**value)
        if not result.tasks or result.episodes_per_task < 1 or result.max_episode_steps < 1:
            raise ValueError("config must contain tasks and positive episode limits")
        unknown = sorted(set(result.tasks) - set(simpler_env.ENVIRONMENTS))
        if unknown:
            raise ValueError(f"unsupported tasks: {unknown}")
        return result


def apply_safety(action: CanonicalAction, config: SafetyConfig):
    translation = np.asarray(action.translation, dtype=np.float64)
    rotation = np.asarray(action.rotation, dtype=np.float64)
    clipped_translation = np.clip(translation, -np.asarray(config.translation_limit_m), config.translation_limit_m)
    rotation_norm = float(np.linalg.norm(rotation))
    clipped_rotation = (
        rotation * config.rotation_limit_rad / rotation_norm
        if rotation_norm > config.rotation_limit_rad
        else rotation
    )
    changed = not (np.allclose(translation, clipped_translation) and np.allclose(rotation, clipped_rotation))
    if changed and config.mode == "reject":
        raise ValueError(
            f"policy action exceeds safety bounds: translation={translation}, rotation_norm={rotation_norm:.4f}"
        )
    return (
        CanonicalAction(clipped_translation, clipped_rotation, action.gripper_open, action.terminate),
        changed,
    )


def to_environment_action(action: CanonicalAction, robot_uid: str) -> np.ndarray:
    """Map the sole robot-specific convention at the simulator boundary."""

    gripper = action.gripper_open
    if "google_robot" in robot_uid:
        gripper = -gripper  # Google controller: +1 closed; canonical/WidowX: +1 open.
    elif "widowx" not in robot_uid:
        raise NotImplementedError(f"unknown gripper convention for robot {robot_uid!r}")
    return np.r_[action.translation, action.rotation, gripper].astype(np.float64)


def _episode_id(task: str, episode_index: int, seed: int) -> str:
    return f"{task}-episode-{episode_index:04d}-seed-{seed}"


def run_episode(policy: RemotePolicy, config: EvaluationConfig, task: str, episode_index: int, run_dir: Path):
    seed = config.base_seed + episode_index
    env_kwargs = dict(config.env_kwargs)
    env_kwargs.setdefault("max_episode_steps", config.max_episode_steps)
    env = simpler_env.make(task, **env_kwargs)
    episode_id = _episode_id(task, episode_index, seed)
    frames = []
    last_info: dict[str, Any] = {}
    safety_clip_count = 0
    termination_reason = "step_limit"
    success = False
    steps = 0
    try:
        obs, reset_info = env.reset(seed=seed)
        instruction = str(env.get_language_instruction())
        policy.reset(episode_id=episode_id, instruction=instruction, seed=seed, task=task)
        policy_obs = get_policy_observation(
            env,
            obs,
            primary_camera_name=config.primary_camera,
            wrist_camera_name=config.wrist_camera,
        )
        frames.append(policy_obs["images"]["primary"])

        for step_index in range(config.max_episode_steps):
            action, was_clipped = apply_safety(policy.act(policy_obs), config.safety)
            safety_clip_count += int(was_clipped)
            env_action = to_environment_action(action, policy_obs["robot_uid"])
            obs, _reward, terminated, truncated, last_info = env.step(env_action)
            steps = step_index + 1
            success = bool(terminated)

            policy_obs = get_policy_observation(
                env,
                obs,
                primary_camera_name=config.primary_camera,
                wrist_camera_name=config.wrist_camera,
            )
            frames.append(policy_obs["images"]["primary"])

            if success:
                termination_reason = "environment_success"
                break
            if truncated:
                termination_reason = "environment_truncated"
                break
            if action.terminate:
                if bool(env.is_final_subtask()):
                    termination_reason = "policy_terminate"
                    break
                env.advance_to_next_subtask()

            new_instruction = str(env.get_language_instruction())
            if new_instruction != instruction:
                instruction = new_instruction
                policy.set_instruction(instruction, seed=seed, task=task)
        else:
            termination_reason = "step_limit"

        result = {
            "episode_id": episode_id,
            "task": task,
            "episode_index": episode_index,
            "seed": seed,
            "success": success,
            "steps": steps,
            "termination_reason": termination_reason,
            "safety_clip_count": safety_clip_count,
            "timing": policy.timing_summary(),
            "episode_stats": last_info.get("episode_stats", {}),
            "reset_info": reset_info,
        }
        if config.save_video and frames:
            from simpler_env.utils.visualization import write_video

            video_path = run_dir / "videos" / task / f"{episode_id}_{'success' if success else 'failure'}.mp4"
            write_video(str(video_path), frames, fps=config.video_fps)
            result["video"] = str(video_path.relative_to(run_dir))
        return _jsonable(result)
    finally:
        env.close()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _weighted_latency(results, category: str):
    weighted_total = 0.0
    count = 0
    for result in results:
        timing = result["timing"]
        mean = timing[category]["mean"]
        calls = timing["calls"]
        if mean is not None:
            weighted_total += float(mean) * calls
            count += calls
    return None if count == 0 else weighted_total / count


def run_evaluation(config: EvaluationConfig, server_url: str, run_name: str | None = None):
    policy = RemotePolicy(
        server_url,
        request_horizon=config.request_horizon,
        execution_horizon=config.execution_horizon,
        timeout=config.server_timeout_seconds,
        required_policy_profile=config.required_policy_profile,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = run_name or f"{policy.metadata['model_id'].replace('/', '_')}-{timestamp}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "server_metadata.json", policy.metadata)
    _write_json(run_dir / "evaluation_config.json", config.__dict__ | {"safety": config.safety.__dict__})

    results = []
    results_path = run_dir / "episodes.jsonl"
    with open(results_path, "a", encoding="utf-8") as stream:
        for task in config.tasks:
            for episode_index in range(config.episodes_per_task):
                result = run_episode(policy, config, task, episode_index, run_dir)
                results.append(result)
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                stream.flush()
                print(
                    f"[{len(results)}/{len(config.tasks) * config.episodes_per_task}] "
                    f"{result['episode_id']}: {'SUCCESS' if result['success'] else 'failure'} "
                    f"({result['steps']} steps)",
                    flush=True,
                )

    per_task = {}
    for task in config.tasks:
        task_results = [result for result in results if result["task"] == task]
        per_task[task] = {
            "episodes": len(task_results),
            "successes": sum(result["success"] for result in task_results),
            "success_rate": float(np.mean([result["success"] for result in task_results])),
        }
    total_steps = sum(result["steps"] for result in results)
    total_clips = sum(result["safety_clip_count"] for result in results)
    summary = {
        "model_id": policy.metadata["model_id"],
        "checkpoint_id": policy.metadata["checkpoint_id"],
        "policy_profile": policy.metadata["policy_profile"],
        "episodes": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "steps": total_steps,
        "safety_clipped_steps": total_clips,
        "safety_clip_rate": total_clips / total_steps if total_steps else 0.0,
        "policy_server_calls": sum(result["timing"]["calls"] for result in results),
        "mean_round_trip_ms": _weighted_latency(results, "round_trip_ms"),
        "mean_server_inference_ms": _weighted_latency(results, "server_inference_ms"),
        "per_task": per_task,
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON evaluation config")
    parser.add_argument("--server-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--allow-profile-mismatch",
        action="store_true",
        help="Allow a released/zero-shot checkpoint; results are not part of the fair adapted comparison.",
    )
    args = parser.parse_args()
    config = EvaluationConfig.load(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.allow_profile_mismatch:
        config.required_policy_profile = None
    run_evaluation(config, args.server_url, args.run_name)


if __name__ == "__main__":
    main()
