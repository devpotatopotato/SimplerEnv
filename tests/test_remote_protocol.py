import threading
import unittest

import numpy as np

from policy_servers.mock_server import MockBackend
from simpler_protocol import (
    CanonicalAction,
    PolicyClient,
    PolicyHTTPServer,
    ProtocolError,
    absolute_pose_chunk_to_deltas,
    decode_image,
    encode_image,
    quaternion_xyzw_to_rotvec,
    rotvec_to_quaternion_xyzw,
)


class SchemaTest(unittest.TestCase):
    def test_image_round_trip(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
        np.testing.assert_array_equal(decode_image(encode_image(image)), image)

    def test_image_rejects_false_small_shape(self):
        payload = encode_image(np.zeros((16, 16, 3), dtype=np.uint8))
        payload["shape"] = [1, 1, 3]
        with self.assertRaises(ProtocolError):
            decode_image(payload)

    def test_action_rejects_invalid_values(self):
        with self.assertRaises(ProtocolError):
            CanonicalAction(np.zeros(2), np.zeros(3), 1.0)
        with self.assertRaises(ProtocolError):
            CanonicalAction(np.zeros(3), np.zeros(3), 2.0)


class GeometryTest(unittest.TestCase):
    def test_rotation_vector_round_trip(self):
        rotation = np.array([0.1, -0.2, 0.3])
        reconstructed = quaternion_xyzw_to_rotvec(rotvec_to_quaternion_xyzw(rotation))
        np.testing.assert_allclose(reconstructed, rotation, atol=1e-9)

    def test_absolute_pose_chunk(self):
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        ninety_z = rotvec_to_quaternion_xyzw([0.0, 0.0, np.pi / 2])
        chunk = np.array(
            [
                [1.1, 2.0, 3.0, *identity, 1.0],
                [1.1, 2.2, 3.0, *ninety_z, 0.0],
            ]
        )
        deltas = absolute_pose_chunk_to_deltas(chunk, [1.0, 2.0, 3.0], identity)
        np.testing.assert_allclose(deltas[0, :6], [0.1, 0, 0, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(deltas[1, :6], [0, 0.2, 0, 0, 0, np.pi / 2], atol=1e-9)


class HTTPTest(unittest.TestCase):
    def setUp(self):
        self.server = PolicyHTTPServer(("127.0.0.1", 0), MockBackend())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = PolicyClient(f"http://127.0.0.1:{self.server.server_port}", timeout=2)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_metadata_reset_and_actions(self):
        self.assertEqual(self.client.metadata()["policy_profile"], "simpler_widowx_cartesian_v1")
        self.client.reset({"episode_id": "test", "instruction": "do nothing", "seed": 0, "task": "test"})
        response = self.client.actions(
            {
                "episode_id": "test",
                "instruction": "do nothing",
                "requested_horizon": 3,
                "images": {"primary": encode_image(np.zeros((8, 8, 3), dtype=np.uint8))},
                "state": {},
            }
        )
        self.assertEqual(len(response["actions"]), 3)
        self.assertEqual(response["actions"][0]["gripper_open"], 1.0)


if __name__ == "__main__":
    unittest.main()
