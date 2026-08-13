import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simpler_training.bridge_data import canonical_action, canonical_state, load_task_filters, matching_task
from simpler_training.models_env import update
from simpler_training.vpp_train import _load_policy_state_dict, _policy_state_dict


class BridgeConversionTest(unittest.TestCase):
    def test_task_filter_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "name": "spoon",
                                "required_groups": [["spoon"], ["towel", "tablecloth"]],
                                "excluded_terms": ["take", " off "],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            filters = load_task_filters(path)
        self.assertEqual(matching_task("Put the spoon on the TABLECLOTH", filters), "spoon")
        self.assertIsNone(matching_task("take the spoon off the tablecloth", filters))
        self.assertIsNone(matching_task("put the carrot on a plate", filters))

    def test_repository_filter_rejects_related_but_wrong_tasks(self):
        filters = load_task_filters("configs/remote_training/bridge_tasks.json")
        self.assertEqual(
            matching_task("PICK UP THE SPOON ANDPUT ON THE TOWEL", filters),
            "spoon_on_towel",
        )
        self.assertIsNone(matching_task("Place the pot between the spoon and towel", filters))
        self.assertIsNone(matching_task("Move the spoon to the right side of the towel.", filters))
        self.assertIsNone(matching_task("take carrot off plate cardboardfence", filters))
        self.assertEqual(
            matching_task("put the green block on the yellow block", filters),
            "stack_cube",
        )

    def test_task_filter_rejects_empty_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(
                json.dumps({"tasks": [{"name": "bad", "required_groups": [[]]}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_task_filters(path)

    def test_canonical_action_units_and_gripper(self):
        action = canonical_action(
            {
                "world_vector": [0.01, -0.02, 0.03],
                "rotation_delta": [0.0, 0.0, 0.0],
                "open_gripper": 1.0,
            }
        )
        np.testing.assert_allclose(action, [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0])
        closed = canonical_action(
            {
                "world_vector": [0.0, 0.0, 0.0],
                "rotation_delta": [0.0, 0.0, 0.0],
                "open_gripper": 0.0,
            }
        )
        self.assertEqual(closed[-1], -1.0)

    def test_canonical_state_has_inference_shape(self):
        state = canonical_state([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04])
        self.assertEqual(state.shape, (8,))
        np.testing.assert_allclose(state[-2:], [0.04, 0.04])


class ModelsEnvironmentTest(unittest.TestCase):
    def test_update_preserves_unrelated_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.env"
            path.write_text(
                '# comment\nPI05_CHECKPOINT="old"\nCOSMOS_CHECKPOINT="keep"\n',
                encoding="utf-8",
            )
            update(path, {"PI05_CHECKPOINT": "/new/$checkpoint", "TRAINED_MODELS": "pi05"})
            text = path.read_text(encoding="utf-8")
        self.assertIn('PI05_CHECKPOINT="/new/\\$checkpoint"', text)
        self.assertIn('COSMOS_CHECKPOINT="keep"', text)
        self.assertIn('TRAINED_MODELS="pi05"', text)


class VPPCheckpointTest(unittest.TestCase):
    class FakeModel:
        def __init__(self, missing=None, unexpected=None):
            self.missing = missing or []
            self.unexpected = unexpected or []

        def state_dict(self):
            return {
                "Video_Former.weight": "former",
                "model.inner_model.weight": "action",
                "TVP_encoder.pipeline.weight": "frozen",
                "language_goal.weight": "frozen",
            }

        def load_state_dict(self, _state, strict):
            self.strict = strict
            return self.missing, self.unexpected

    def test_compact_checkpoint_keeps_action_policy(self):
        state = _policy_state_dict(self.FakeModel())
        self.assertEqual(set(state), {"Video_Former.weight", "model.inner_model.weight"})

    def test_checkpoint_loader_allows_only_frozen_missing_tensors(self):
        model = self.FakeModel(missing=["TVP_encoder.pipeline.weight"])
        _load_policy_state_dict(model, {"model.inner_model.weight": "action"})
        self.assertFalse(model.strict)
        with self.assertRaises(RuntimeError):
            _load_policy_state_dict(
                self.FakeModel(missing=["model.inner_model.weight"]),
                {"Video_Former.weight": "former"},
            )


if __name__ == "__main__":
    unittest.main()
