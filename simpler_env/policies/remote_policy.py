"""Simulator-side client for the model-independent remote-policy protocol."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from simpler_protocol import CanonicalAction, PolicyClient, ProtocolError, encode_image


def _json_vector(value, name: str):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ProtocolError(f"observation state {name!r} contains non-finite values")
    return array.tolist()


class RemotePolicy:
    """Fetch action chunks and execute a fixed, model-independent prefix."""

    def __init__(
        self,
        server_url: str,
        *,
        request_horizon: int = 16,
        execution_horizon: int = 1,
        timeout: float = 120.0,
        required_policy_profile: str | None = "simpler_widowx_cartesian_v1",
    ):
        if request_horizon < 1 or execution_horizon < 1:
            raise ValueError("request and execution horizons must be positive")
        if execution_horizon > request_horizon:
            raise ValueError("execution_horizon cannot exceed request_horizon")
        self.client = PolicyClient(server_url, timeout=timeout)
        self.request_horizon = request_horizon
        self.execution_horizon = execution_horizon
        self.metadata = self.client.metadata()
        if required_policy_profile and self.metadata["policy_profile"] != required_policy_profile:
            raise ProtocolError(
                "policy profile mismatch: evaluation requires "
                f"{required_policy_profile!r}, server reports {self.metadata['policy_profile']!r}. "
                "Use an adapted checkpoint or explicitly disable this check for a zero-shot diagnostic."
            )
        self._queue: deque[CanonicalAction] = deque()
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self.request_latencies_ms: list[float] = []
        self.server_latencies_ms: list[float] = []
        self.server_calls = 0

    def reset(self, *, episode_id: str, instruction: str, seed: int, task: str):
        self._queue.clear()
        self._episode_id = episode_id
        self._instruction = instruction
        self.request_latencies_ms.clear()
        self.server_latencies_ms.clear()
        self.server_calls = 0
        self.client.reset(
            {"episode_id": episode_id, "instruction": instruction, "seed": int(seed), "task": task}
        )

    def set_instruction(self, instruction: str, *, seed: int, task: str):
        if instruction == self._instruction:
            return
        self._queue.clear()
        self._instruction = instruction
        self.client.reset(
            {
                "episode_id": self._episode_id,
                "instruction": instruction,
                "seed": int(seed),
                "task": task,
                "subtask_reset": True,
            }
        )

    def _request_actions(self, observation: Mapping[str, Any]):
        state = {name: _json_vector(value, name) for name, value in observation["state"].items()}
        payload = {
            "episode_id": self._episode_id,
            "instruction": self._instruction,
            "requested_horizon": self.request_horizon,
            "images": {name: encode_image(image) for name, image in observation["images"].items()},
            "state": state,
            "robot_uid": observation.get("robot_uid"),
            "camera_names": observation.get("camera_names", {}),
        }
        started = time.perf_counter()
        response = self.client.actions(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        actions = [CanonicalAction.from_dict(value) for value in response.get("actions", [])]
        if not actions:
            raise ProtocolError("policy server returned no actions")
        self._queue.extend(actions[: self.execution_horizon])
        self.request_latencies_ms.append(elapsed_ms)
        if "server_inference_ms" in response:
            self.server_latencies_ms.append(float(response["server_inference_ms"]))
        self.server_calls += 1

    def act(self, observation: Mapping[str, Any]) -> CanonicalAction:
        if self._episode_id is None or self._instruction is None:
            raise RuntimeError("RemotePolicy.reset must be called before act")
        if not self._queue:
            self._request_actions(observation)
        return self._queue.popleft()

    def timing_summary(self):
        def statistics(values):
            if not values:
                return {"mean": None, "p50": None, "p95": None}
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(np.mean(array)),
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)),
            }

        return {
            "calls": self.server_calls,
            "round_trip_ms": statistics(self.request_latencies_ms),
            "server_inference_ms": statistics(self.server_latencies_ms),
        }
