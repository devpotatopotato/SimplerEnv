"""Convert the public Bridge RLDS data into the shared canonical LeRobot data.

The resulting train and validation repositories are the single source of truth
for both OpenPI and VPP. Actions use physical SimplerEnv units:

    delta xyz [metres] + delta axis-angle [radians] + gripper (+1 open)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from simpler_protocol.geometry import euler_xyz_to_rotvec

TRAIN_REPO_ID = "local/simpler_bridge_train"
VAL_REPO_ID = "local/simpler_bridge_val"


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _text(value: Any) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def load_task_filters(path: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task filter must contain a non-empty 'tasks' list")
    for task in tasks:
        groups = task.get("required_groups")
        if not isinstance(groups, list) or not groups or any(not group for group in groups):
            raise ValueError(f"invalid required_groups for task {task!r}")
    return tasks


def matching_task(instruction: str, filters: list[dict[str, Any]]) -> str | None:
    if not filters:
        return "all"
    normalized = " ".join(instruction.casefold().split())
    for task in filters:
        if all(any(str(term).casefold() in normalized for term in group) for group in task["required_groups"]):
            return str(task["name"])
    return None


def canonical_action(raw_action: dict[str, Any]) -> np.ndarray:
    translation = _numpy(raw_action["world_vector"]).astype(np.float64).reshape(3)
    euler_delta = _numpy(raw_action["rotation_delta"]).astype(np.float64).reshape(3)
    open_fraction = float(_numpy(raw_action["open_gripper"]).reshape(-1)[0])
    gripper = 1.0 if open_fraction >= 0.5 else -1.0
    action = np.r_[translation, euler_xyz_to_rotvec(euler_delta), gripper].astype(np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError("non-finite action")
    return action


def canonical_state(raw_state: Any) -> np.ndarray:
    state = _numpy(raw_state).astype(np.float64).reshape(-1)
    if state.size < 6:
        raise ValueError(f"Bridge state must have at least 6 values, got {state.size}")
    gripper = float(state[6]) if state.size > 6 else 0.0
    result = np.r_[state[:3], euler_xyz_to_rotvec(state[3:6]), gripper, gripper].astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite state")
    return result


def _episode_is_validation(identifier: str, validation_fraction: float) -> bool:
    bucket = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big") / float(2**64)
    return bucket < validation_fraction


def _dataset_features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    return {
        "image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def _make_dataset(repo_id: str, root: Path, fps: int, image_shape: tuple[int, int, int]):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="widowx_bridge",
        fps=fps,
        features=_dataset_features(image_shape),
        use_videos=False,
        image_writer_threads=8,
    )


def _source_dataset(source: str):
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(builder_dir=source)
    return builder.as_dataset(
        split="train",
        shuffle_files=False,
        read_config=tfds.ReadConfig(add_tfds_id=True),
    )


def convert(
    source: str,
    output_home: Path,
    task_filter: Path | None,
    validation_fraction: float,
    fps: int,
    max_episodes: int | None,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    filters = load_task_filters(task_filter)
    destinations = {
        "train": output_home / TRAIN_REPO_ID,
        "val": output_home / VAL_REPO_ID,
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError("dataset destination already exists: " + ", ".join(existing))

    build_parent = output_home / ".building"
    build_parent.mkdir(parents=True, exist_ok=True)
    build_roots = {
        split: build_parent / f"{repo_id.replace('/', '_')}-{os.getpid()}"
        for split, repo_id in (("train", TRAIN_REPO_ID), ("val", VAL_REPO_ID))
    }
    for root in build_roots.values():
        if root.exists():
            raise FileExistsError(f"temporary dataset path exists: {root}")

    datasets: dict[str, Any] = {}
    counts = {"scanned": 0, "matched": 0, "train_episodes": 0, "val_episodes": 0, "frames": 0}
    matched_by_task: dict[str, int] = {}
    try:
        for episode_index, episode in enumerate(_source_dataset(source)):
            counts["scanned"] += 1
            steps = list(episode["steps"])
            if not steps:
                continue
            observation = steps[0]["observation"]
            instruction = _text(observation["natural_language_instruction"])
            task_name = matching_task(instruction, filters)
            if task_name is None:
                continue
            if max_episodes is not None and filters:
                per_task_limit = (max_episodes + len(filters) - 1) // len(filters)
                if matched_by_task.get(task_name, 0) >= per_task_limit:
                    continue
            identifier = _text(episode.get("tfds_id", f"episode-{episode_index}"))
            split = "val" if _episode_is_validation(identifier, validation_fraction) else "train"

            if not datasets:
                first_image = _numpy(observation["image"])
                if first_image.ndim != 3 or first_image.shape[-1] != 3:
                    raise ValueError(f"Bridge image must be HWC RGB, got {first_image.shape}")
                shape = tuple(int(item) for item in first_image.shape)
                datasets = {
                    name: _make_dataset(repo_id, build_roots[name], fps, shape)
                    for name, repo_id in (("train", TRAIN_REPO_ID), ("val", VAL_REPO_ID))
                }

            written = 0
            for step in steps:
                try:
                    frame_observation = step["observation"]
                    image = _numpy(frame_observation["image"]).astype(np.uint8)
                    datasets[split].add_frame(
                        {
                            "image": image,
                            "state": canonical_state(frame_observation["state"]),
                            "actions": canonical_action(step["action"]),
                            "task": instruction,
                        }
                    )
                    written += 1
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid frame in source episode {identifier}: {exc}") from exc
            if written:
                datasets[split].save_episode()
                counts["matched"] += 1
                counts[f"{split}_episodes"] += 1
                counts["frames"] += written
                matched_by_task[task_name] = matched_by_task.get(task_name, 0) + 1
                print(
                    f"[{counts['matched']}] {split}: {instruction!r} ({written} frames)",
                    flush=True,
                )
            if max_episodes is not None and counts["matched"] >= max_episodes:
                break

        if not datasets or counts["train_episodes"] == 0 or counts["val_episodes"] == 0:
            raise RuntimeError(
                "conversion needs at least one matching train and validation episode; "
                f"observed {counts}"
            )
        expected_tasks = {str(task["name"]) for task in filters}
        missing_tasks = sorted(expected_tasks - matched_by_task.keys())
        if missing_tasks:
            raise RuntimeError(
                "the Bridge source contained no matching episodes for required task filters: "
                + ", ".join(missing_tasks)
            )
        for dataset in datasets.values():
            dataset.stop_image_writer()
        for split, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(build_roots[split], destination)

        manifest = {
            "schema_version": 1,
            "source": source,
            "action_encoding": "eef_delta_axis_angle_gripper_v1",
            "fps": fps,
            "validation_fraction": validation_fraction,
            "max_episodes": max_episodes,
            "task_filter": None if task_filter is None else str(task_filter.resolve()),
            "counts": counts,
            "matched_by_task": matched_by_task,
            "repositories": {"train": TRAIN_REPO_ID, "val": VAL_REPO_ID},
        }
        with open(output_home / "simpler_bridge_manifest.json", "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return manifest
    except BaseException:
        for dataset in datasets.values():
            try:
                dataset.stop_image_writer()
            except Exception:
                pass
        for root in build_roots.values():
            if root.exists():
                shutil.rmtree(root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="gs://gresearch/robotics/bridge/0.1.0")
    parser.add_argument("--output-home", type=Path, required=True)
    parser.add_argument("--task-filter", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, help="limit matching episodes for a small trial")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 2:
        parser.error("--max-episodes must be at least 2")
    args.output_home.mkdir(parents=True, exist_ok=True)
    result = convert(
        args.source,
        args.output_home.resolve(),
        args.task_filter,
        args.validation_fraction,
        args.fps,
        args.max_episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
