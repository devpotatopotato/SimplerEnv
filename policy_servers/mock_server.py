"""Dependency-free server used to smoke-test the full evaluation connection."""

from __future__ import annotations

import argparse

import numpy as np

from policy_servers.common import decode_request
from simpler_protocol import CanonicalAction, PolicyBackend, serve_backend
from simpler_protocol.schema import make_metadata


class MockBackend(PolicyBackend):
    def __init__(self, checkpoint_id: str = "mock-zero-v1"):
        self._metadata = make_metadata(
            model_id="mock-policy",
            checkpoint_id=checkpoint_id,
            policy_profile="simpler_widowx_cartesian_v1",
            output_mode="canonical",
            wrist_image="unused",
            adaptation={"dataset": "test-only", "method": "none"},
        )

    @property
    def metadata(self):
        return self._metadata

    def predict(self, request):
        _, _, _, horizon = decode_request(request)
        # A stationary open-gripper policy is safe for end-to-end plumbing tests.
        return [CanonicalAction(np.zeros(3), np.zeros(3), 1.0) for _ in range(horizon)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_backend(MockBackend(), args.host, args.port)


if __name__ == "__main__":
    main()
