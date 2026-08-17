import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_executable(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class RemoteEvalLauncherTest(unittest.TestCase):
    def test_dry_run_accepts_one_shared_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote_home = root / "remote"
            eval_python = remote_home / "envs/simpler/bin/python"
            pi_python = remote_home / "sources/openpi/.venv/bin/python"
            eval_python.parent.mkdir(parents=True)
            pi_python.parent.mkdir(parents=True)
            eval_python.symlink_to(sys.executable)
            pi_python.symlink_to(sys.executable)

            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            env_file = root / "models.env"
            env_file.write_text(
                "\n".join(
                    [
                        'TRAINED_MODELS="pi05"',
                        'PI05_CONFIG_NAME="pi05_simpler_bridge_lora"',
                        f'PI05_CHECKPOINT="{checkpoint}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            fake_bin = root / "bin"
            _make_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nprintf '0\\n'\n")
            _make_executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "REMOTE_EVAL_HOME": str(remote_home),
                    "RESULTS_DIR": str(root / "results"),
                    "POLICY_GPU": "0",
                }
            )
            environment.pop("EVAL_GPU", None)
            result = subprocess.run(
                [
                    str(REPO_ROOT / "scripts/remote_eval/run.sh"),
                    "--env-file",
                    str(env_file),
                    "--models",
                    "pi05",
                    "--run-tag",
                    "shared-test",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("shared-GPU mode", result.stdout)
        self.assertIn("physical GPU 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
