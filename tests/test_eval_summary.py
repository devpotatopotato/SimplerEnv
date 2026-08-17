import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.remote_eval.summarize_results import (
    _expected_episode_count,
    build_report,
    discover_models,
    format_report,
    latest_run_tag,
)


class EvaluationSummaryTest(unittest.TestCase):
    def _write_run(self, root: Path, model: str, tag: str, *, recorded_episodes: int = 2):
        run_dir = root / f"{model}-{tag}"
        run_dir.mkdir(parents=True)
        (run_dir / "evaluation_config.json").write_text(
            json.dumps({"tasks": ["task_a"], "episodes_per_task": 2}), encoding="utf-8"
        )
        (run_dir / "server_metadata.json").write_text(json.dumps({"model_id": model}), encoding="utf-8")
        with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode in range(recorded_episodes):
                stream.write(
                    json.dumps(
                        {
                            "task": "task_a",
                            "episode_index": episode,
                            "seed": episode,
                            "success": episode == 0,
                        }
                    )
                    + "\n"
                )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "episodes": 2,
                    "successes": 1,
                    "success_rate": 0.5,
                    "safety_clip_rate": 0.25,
                    "mean_server_inference_ms": 10.0,
                    "mean_round_trip_ms": 12.0,
                    "per_task": {
                        "task_a": {"episodes": 2, "successes": 1, "success_rate": 0.5},
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_complete_report_and_text_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "vpp", "comparison")
            self._write_run(root, "pi05", "comparison")
            models = discover_models(root, "comparison")
            report = build_report(root, "comparison", models)
            output = format_report(report)

        self.assertEqual(models, ["pi05", "vpp"])
        self.assertTrue(report["complete"])
        self.assertEqual(report["paired_comparisons"][0]["paired_episodes"], 2)
        self.assertEqual(report["paired_comparisons"][0]["success_rate_delta_a_minus_b"], 0.0)
        self.assertIn("2/2 models complete", output)
        self.assertIn("1/2 (50.0%; -)", output)
        self.assertIn("inference_ms", output)

    def test_incomplete_episode_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "pi05", "partial", recorded_episodes=1)
            report = build_report(root, "partial", ["pi05"])

        self.assertFalse(report["complete"])
        self.assertIn("episodes.jsonl has 1", " ".join(report["models"][0]["errors"]))

    def test_discovery_includes_attempted_model_without_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "pi05", "attempted")
            launcher = root / "_launcher" / "attempted"
            launcher.mkdir(parents=True)
            (launcher / "vpp_server.log").touch()
            models = discover_models(root, "attempted")
            report = build_report(root, "attempted", models)

        self.assertEqual(models, ["pi05", "vpp"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"][1]["status"], "incomplete")

    def test_latest_tag_uses_launcher_modification_time(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "_launcher"
            old = launcher / "old"
            new = launcher / "new"
            old.mkdir(parents=True)
            new.mkdir()
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(latest_run_tag(Path(directory)), "new")

    def test_standard_protocol_episode_count(self):
        config = {
            "tasks": ["a", "b"],
            "policy_seeds": [0, 2],
            "task_settings": {
                "a": {"object_episode_ids": [0, 1, 2]},
                "b": {"object_episode_ids": [0, 1]},
            },
        }
        self.assertEqual(_expected_episode_count(config), 10)

        with Path("configs/remote_eval/widowx_bridge.json").open(encoding="utf-8") as stream:
            standard = json.load(stream)
        self.assertEqual(_expected_episode_count(standard), 96)
        self.assertEqual(standard["task_settings"]["widowx_spoon_on_towel"]["max_episode_steps"], 60)
        self.assertEqual(
            standard["task_settings"]["widowx_put_eggplant_in_basket"]["max_episode_steps"], 120
        )


if __name__ == "__main__":
    unittest.main()
