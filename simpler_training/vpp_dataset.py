"""Minimal reader for the canonical LeRobot v2 parquet repositories."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class CanonicalLeRobotVPPDataset:
    """Random-access action chunks without adding LeRobot to VPP's environment."""

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int = 16,
        image_size: int = 256,
        sample_stride: int = 1,
        action_scale: tuple[float, ...] = (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0),
        table_cache_size: int = 4,
    ) -> None:
        import torch

        self.torch = torch
        self.root = Path(root).resolve()
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.sample_stride = int(sample_stride)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.table_cache_size = int(table_cache_size)
        if self.action_horizon < 1 or self.image_size < 1 or self.sample_stride < 1:
            raise ValueError("action horizon, image size, and sample stride must be positive")
        if self.action_scale.shape != (7,) or np.any(self.action_scale <= 0):
            raise ValueError("action_scale must have seven positive values")
        with open(self.root / "meta/info.json", encoding="utf-8") as stream:
            self.info = json.load(stream)
        episodes = _read_jsonlines(self.root / "meta/episodes.jsonl")
        self.episodes = sorted(episodes, key=lambda value: int(value["episode_index"]))
        if not self.episodes:
            raise ValueError(f"dataset contains no episodes: {self.root}")
        self._sample_counts = [
            (int(episode["length"]) + self.sample_stride - 1) // self.sample_stride
            for episode in self.episodes
        ]
        self._cumulative = np.cumsum(self._sample_counts).tolist()
        self._table_cache: OrderedDict[int, Any] = OrderedDict()

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _table(self, episode_index: int):
        import pyarrow.parquet as parquet

        if episode_index in self._table_cache:
            table = self._table_cache.pop(episode_index)
            self._table_cache[episode_index] = table
            return table
        chunk = episode_index // int(self.info["chunks_size"])
        relative = self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
        table = parquet.read_table(self.root / relative, columns=["image", "actions"])
        self._table_cache[episode_index] = table
        while len(self._table_cache) > self.table_cache_size:
            self._table_cache.popitem(last=False)
        return table

    def _decode_image(self, value: Any):
        from PIL import Image
        import torch.nn.functional as functional

        value = value.as_py() if hasattr(value, "as_py") else value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                image = Image.open(BytesIO(value["bytes"])).convert("RGB")
            elif value.get("path"):
                path = Path(value["path"])
                image = Image.open(path if path.is_absolute() else self.root / path).convert("RGB")
            else:
                raise ValueError("image parquet value has neither bytes nor path")
        elif isinstance(value, (bytes, bytearray)):
            image = Image.open(BytesIO(value)).convert("RGB")
        else:
            raise TypeError(f"unsupported image parquet value: {type(value)}")
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = self.torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        tensor = functional.interpolate(
            tensor[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0]
        return (tensor - 0.5) / 0.5

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect_right(self._cumulative, index)
        previous = 0 if episode_position == 0 else self._cumulative[episode_position - 1]
        frame_index = (index - previous) * self.sample_stride
        episode = self.episodes[episode_position]
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        table = self._table(episode_index)
        image = self._decode_image(table["image"][frame_index])

        action_rows = []
        for offset in range(self.action_horizon):
            row = min(frame_index + offset, length - 1)
            action_rows.append(np.asarray(table["actions"][row].as_py(), dtype=np.float32))
        actions = np.stack(action_rows)
        if actions.shape != (self.action_horizon, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid action chunk in episode {episode_index}: {actions.shape}")
        normalized = np.clip(actions / self.action_scale[None], -1.0, 1.0)
        instruction = str(episode.get("tasks", [""])[0])
        return {
            "rgb_obs": {
                "rgb_static": image[None],
                # Match the evaluator, which has no WidowX wrist camera.
                "rgb_gripper": self.torch.full_like(image[None], -1.0),
            },
            "actions": self.torch.from_numpy(normalized).float(),
            "lang_text": instruction,
            "idx": int(index),
        }
