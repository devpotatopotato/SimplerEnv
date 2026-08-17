"""Fair, reproducible evaluation of remote policies in prepackaged SimplerEnv tasks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import simpler_env
from simpler_env.policies.remote_policy import RemotePolicy
from simpler_env.utils.env.observation_utils import get_policy_observation
from simpler_protocol import CanonicalAction, json_safe


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
    policy_seeds: list[int] | None = None
    environment_seed: int = 0
    task_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    protocol_name: str = "legacy_seeded_episodes"
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
    training_manifest: str | None = None
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
        unknown_settings = sorted(set(result.task_settings) - set(result.tasks))
        if unknown_settings:
            raise ValueError(f"task_settings contains tasks not selected for evaluation: {unknown_settings}")
        if result.policy_seeds is not None:
            if not result.policy_seeds or any(not isinstance(seed, int) for seed in result.policy_seeds):
                raise ValueError("policy_seeds must contain at least one integer")
            if len(set(result.policy_seeds)) != len(result.policy_seeds):
                raise ValueError("policy_seeds must not contain duplicates")
        for task, settings in result.task_settings.items():
            object_ids = settings.get("object_episode_ids")
            if not isinstance(object_ids, list) or not object_ids or any(
                not isinstance(identifier, int) or identifier < 0 for identifier in object_ids
            ):
                raise ValueError(f"task_settings.{task}.object_episode_ids must contain non-negative integers")
            if len(set(object_ids)) != len(object_ids):
                raise ValueError(f"task_settings.{task}.object_episode_ids must not contain duplicates")
            max_steps = settings.get("max_episode_steps", result.max_episode_steps)
            if not isinstance(max_steps, int) or max_steps < 1:
                raise ValueError(f"task_settings.{task}.max_episode_steps must be positive")
        return result

    def episode_specs(self):
        """Yield the predeclared environment/policy seed matrix.

        Explicit task settings implement SimplerEnv's fixed object-variation
        protocol. Configs without them retain the legacy smoke-test behavior.
        """

        for task in self.tasks:
            settings = self.task_settings.get(task)
            if settings is None:
                for episode_index in range(self.episodes_per_task):
                    seed = self.base_seed + episode_index
                    yield {
                        "task": task,
                        "episode_index": episode_index,
                        "object_episode_id": None,
                        "environment_seed": seed,
                        "policy_seed": seed,
                        "max_episode_steps": self.max_episode_steps,
                    }
                continue

            policy_seeds = self.policy_seeds or [0]
            episode_index = 0
            for object_episode_id in settings["object_episode_ids"]:
                for policy_seed in policy_seeds:
                    yield {
                        "task": task,
                        "episode_index": episode_index,
                        "object_episode_id": int(object_episode_id),
                        "environment_seed": int(settings.get("environment_seed", self.environment_seed)),
                        "policy_seed": int(policy_seed),
                        "max_episode_steps": int(settings.get("max_episode_steps", self.max_episode_steps)),
                    }
                    episode_index += 1


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


def _episode_id(spec: dict[str, Any]) -> str:
    if spec["object_episode_id"] is None:
        return f"{spec['task']}-episode-{spec['episode_index']:04d}-seed-{spec['policy_seed']}"
    return (
        f"{spec['task']}-object-{spec['object_episode_id']:02d}"
        f"-env-{spec['environment_seed']}-policy-{spec['policy_seed']}"
    )


def run_episode(policy: RemotePolicy, config: EvaluationConfig, spec: dict[str, Any], run_dir: Path):
    task = spec["task"]
    episode_index = spec["episode_index"]
    environment_seed = spec["environment_seed"]
    policy_seed = spec["policy_seed"]
    max_episode_steps = spec["max_episode_steps"]
    env_kwargs = dict(config.env_kwargs)
    env_kwargs["max_episode_steps"] = max_episode_steps
    env = simpler_env.make(task, **env_kwargs)
    base_env = getattr(env, "unwrapped", env)
    episode_id = _episode_id(spec)
    frames = []
    last_info: dict[str, Any] = {}
    safety_clip_count = 0
    termination_reason = "step_limit"
    success = False
    steps = 0
    try:
        reset_options = None
        if spec["object_episode_id"] is not None:
            reset_options = {"obj_init_options": {"episode_id": spec["object_episode_id"]}}
        obs, reset_info = env.reset(seed=environment_seed, options=reset_options)
        instruction = str(base_env.get_language_instruction())
        policy.reset(episode_id=episode_id, instruction=instruction, seed=policy_seed, task=task)
        policy_obs = get_policy_observation(
            env,
            obs,
            primary_camera_name=config.primary_camera,
            wrist_camera_name=config.wrist_camera,
        )
        frames.append(policy_obs["images"]["primary"])

        for step_index in range(max_episode_steps):
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
                if bool(base_env.is_final_subtask()):
                    termination_reason = "policy_terminate"
                    break
                base_env.advance_to_next_subtask()

            new_instruction = str(base_env.get_language_instruction())
            if new_instruction != instruction:
                instruction = new_instruction
                policy.set_instruction(instruction, seed=policy_seed, task=task)
        else:
            termination_reason = "step_limit"

        result = {
            "episode_id": episode_id,
            "task": task,
            "episode_index": episode_index,
            "object_episode_id": spec["object_episode_id"],
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            # Retained for old result consumers; policy_seed is authoritative.
            "seed": policy_seed,
            "max_episode_steps": max_episode_steps,
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
        return json_safe(result)
    finally:
        env.close()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, indent=2, sort_keys=True)
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


def _wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054):
    if episodes == 0:
        return None
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


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
    training_manifest = None
    if config.training_manifest and policy.metadata.get("comparison_group") == "shared_bridge_adaptation":
        with open(config.training_manifest, encoding="utf-8") as stream:
            training_manifest = json.load(stream)
        _write_json(run_dir / "training_manifest.json", training_manifest)

    results = []
    episode_specs = list(config.episode_specs())
    results_path = run_dir / "episodes.jsonl"
    with open(results_path, "a", encoding="utf-8") as stream:
        for spec in episode_specs:
            result = run_episode(policy, config, spec, run_dir)
            results.append(result)
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            print(
                f"[{len(results)}/{len(episode_specs)}] "
                f"{result['episode_id']}: {'SUCCESS' if result['success'] else 'failure'} "
                f"({result['steps']} steps)",
                flush=True,
            )

    per_task = {}
    for task in config.tasks:
        task_results = [result for result in results if result["task"] == task]
        successes = sum(result["success"] for result in task_results)
        per_task[task] = {
            "episodes": len(task_results),
            "successes": successes,
            "success_rate": float(np.mean([result["success"] for result in task_results])),
            "confidence_interval_95": _wilson_interval(successes, len(task_results)),
        }
    total_steps = sum(result["steps"] for result in results)
    total_clips = sum(result["safety_clip_count"] for result in results)
    total_successes = sum(result["success"] for result in results)
    summary = {
        "model_id": policy.metadata["model_id"],
        "checkpoint_id": policy.metadata["checkpoint_id"],
        "policy_profile": policy.metadata["policy_profile"],
        "protocol_name": config.protocol_name,
        "episodes": len(results),
        "successes": total_successes,
        "success_rate": float(np.mean([result["success"] for result in results])),
        "macro_success_rate": float(np.mean([metrics["success_rate"] for metrics in per_task.values()])),
        "confidence_interval_95": _wilson_interval(total_successes, len(results)),
        "steps": total_steps,
        "safety_clipped_steps": total_clips,
        "safety_clip_rate": total_clips / total_steps if total_steps else 0.0,
        "policy_server_calls": sum(result["timing"]["calls"] for result in results),
        "mean_round_trip_ms": _weighted_latency(results, "round_trip_ms"),
        "mean_server_inference_ms": _weighted_latency(results, "server_inference_ms"),
        "per_task": per_task,
    }
    if training_manifest is not None:
        summary["training_data"] = {
            "manifest": "training_manifest.json",
            "task_coverage": training_manifest.get("dataset", {}).get("task_coverage"),
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
