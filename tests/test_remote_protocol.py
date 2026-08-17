import threading
import unittest

import numpy as np

from policy_servers.cosmos3.server import (
    cosmos_absolute_chunk_to_canonical,
    simpler_pose_to_cosmos,
)
from policy_servers.mock_server import MockBackend
from simpler_protocol import (
    CanonicalAction,
    PolicyClient,
    PolicyHTTPServer,
    ProtocolError,
    absolute_pose_chunk_to_deltas,
    decode_image,
    encode_image,
    json_safe,
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
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

    def test_json_safe_serializes_pose_and_nonfinite_values(self):
        class FakePose:
            p = np.array([1.0, 2.0, 3.0])
            q = np.array([1.0, 0.0, 0.0, 0.0])

        result = json_safe({"pose": FakePose(), "invalid": float("nan")})
        self.assertEqual(result["pose"]["position"], [1.0, 2.0, 3.0])
        self.assertEqual(result["pose"]["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertIsNone(result["invalid"])


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

    def test_cosmos_bridge_tool_frame_round_trip(self):
        current_position = np.array([0.35, -0.1, 0.22])
        current_quaternion = rotvec_to_quaternion_xyzw([0.2, -0.1, 0.3])
        target_position = current_position + np.array([0.01, -0.02, 0.005])
        target_quaternion = rotvec_to_quaternion_xyzw([0.21, -0.08, 0.29])
        model_position, model_quaternion = simpler_pose_to_cosmos(target_position, target_quaternion)
        # Service output 0 means Bridge's pre-flip gripper label was 1=open.
        output = np.array([[*model_position, *model_quaternion, 0.0]])
        delta = cosmos_absolute_chunk_to_canonical(output, current_position, current_quaternion)[0]

        np.testing.assert_allclose(delta[:3], target_position - current_position, atol=1e-9)
        expected_rotation = quaternion_xyzw_to_rotvec(
            quaternion_multiply_xyzw(target_quaternion, quaternion_inverse_xyzw(current_quaternion))
        )
        np.testing.assert_allclose(delta[3:6], expected_rotation, atol=1e-8)
        self.assertEqual(delta[6], 1.0)


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
