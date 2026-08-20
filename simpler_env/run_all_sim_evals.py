"""Run the published SimplerEnv simulation-evaluation matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
from typing import NamedTuple


class EvalScript(NamedTuple):
    model: str
    suite: str
    setup: str
    path: str


EVAL_SCRIPTS = (
    EvalScript("rt1", "google", "visual-matching", "scripts/rt1_pick_coke_can_visual_matching.sh"),
    EvalScript("rt1", "google", "visual-matching", "scripts/rt1_move_near_visual_matching.sh"),
    EvalScript("rt1", "google", "visual-matching", "scripts/rt1_drawer_visual_matching.sh"),
    EvalScript("rt1", "google", "visual-matching", "scripts/rt1_put_in_drawer_visual_matching.sh"),
    EvalScript("octo", "google", "visual-matching", "scripts/octo_pick_coke_can_visual_matching.sh"),
    EvalScript("octo", "google", "visual-matching", "scripts/octo_move_near_visual_matching.sh"),
    EvalScript("octo", "google", "visual-matching", "scripts/octo_drawer_visual_matching.sh"),
    EvalScript("octo", "google", "visual-matching", "scripts/octo_put_in_drawer_visual_matching.sh"),
    EvalScript("rt1", "google", "variant-aggregation", "scripts/rt1_pick_coke_can_variant_agg.sh"),
    EvalScript("rt1", "google", "variant-aggregation", "scripts/rt1_move_near_variant_agg.sh"),
    EvalScript("rt1", "google", "variant-aggregation", "scripts/rt1_drawer_variant_agg.sh"),
    EvalScript("rt1", "google", "variant-aggregation", "scripts/rt1_put_in_drawer_variant_agg.sh"),
    EvalScript("octo", "google", "variant-aggregation", "scripts/octo_pick_coke_can_variant_agg.sh"),
    EvalScript("octo", "google", "variant-aggregation", "scripts/octo_move_near_variant_agg.sh"),
    EvalScript("octo", "google", "variant-aggregation", "scripts/octo_drawer_variant_agg.sh"),
    EvalScript("octo", "google", "variant-aggregation", "scripts/octo_put_in_drawer_variant_agg.sh"),
    EvalScript("rt1", "bridge", "visual-matching", "scripts/rt1x_bridge.sh"),
    EvalScript("octo", "bridge", "visual-matching", "scripts/octo_bridge.sh"),
)

RT1_CHECKPOINTS = (
    "checkpoints/rt_1_tf_trained_for_000400120",
    "checkpoints/rt_1_tf_trained_for_000058240",
    "checkpoints/rt_1_x_tf_trained_for_002272480_step",
    "checkpoints/rt_1_tf_trained_for_000001120",
)

MODEL_MODULES = {
    "rt1": ("tensorflow", "tensorflow_hub", "tf_agents"),
    "octo": ("tensorflow", "jax", "octo", "transformers"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all supported RT-1 and Octo simulation benchmark scripts sequentially."
    )
    parser.add_argument("--models", nargs="+", choices=("rt1", "octo"), default=("rt1", "octo"))
    parser.add_argument("--suite", choices=("all", "google", "bridge"), default="all")
    parser.add_argument(
        "--setup",
        choices=("both", "visual-matching", "variant-aggregation"),
        default="both",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", type=int, help="Use one CUDA device")
    gpu_group.add_argument("--gpus", type=int, nargs="+", help="Run one evaluation script per CUDA device")
    parser.add_argument("--dry-run", action="store_true", help="Print scripts without running them")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    args.gpus = tuple(args.gpus if args.gpus is not None else (args.gpu if args.gpu is not None else 0,))
    if any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU device indices must be non-negative")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU device indices must be unique")
    return args


def select_scripts(args: argparse.Namespace) -> list[EvalScript]:
    return [
        script
        for script in EVAL_SCRIPTS
        if script.model in args.models
        and (args.suite == "all" or script.suite == args.suite)
        and (args.setup == "both" or script.setup == args.setup)
    ]


def preflight(repo_root: Path, models: set[str]) -> list[str]:
    errors = []
    for model in sorted(models):
        missing = [module for module in MODEL_MODULES[model] if importlib.util.find_spec(module) is None]
        if missing:
            errors.append(f"{model}: missing Python modules: {', '.join(missing)}")

    if "rt1" in models:
        missing_checkpoints = [path for path in RT1_CHECKPOINTS if not (repo_root / path).is_dir()]
        if missing_checkpoints:
            errors.append("rt1: missing checkpoints: " + ", ".join(missing_checkpoints))
    return errors


def run_script(
    repo_root: Path,
    script: EvalScript,
    index: int,
    total: int,
    gpu: int,
) -> tuple[int, EvalScript, int]:
    child_env = os.environ.copy()
    child_env["SIMPLER_GPU_ID"] = str(gpu)
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = ["bash", script.path]
    print(f"\n[{index}/{total}][GPU {gpu}] {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=repo_root, env=child_env)
    return index, script, result.returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scripts = select_scripts(args)

    if not scripts:
        print("No evaluation scripts match the requested filters.")
        return 2

    print(f"Selected {len(scripts)} evaluation scripts:")
    for script in scripts:
        print(f"  {script.model:4} {script.suite:6} {script.setup:19} {script.path}")

    print(f"GPU workers: {', '.join(map(str, args.gpus))}")

    if args.dry_run:
        print("\nInitial GPU assignment:")
        for index, script in enumerate(scripts[: len(args.gpus)], start=1):
            gpu = args.gpus[index - 1]
            print(f"  [GPU {gpu}] bash {script.path}")
        return 0

    if not args.skip_preflight:
        errors = preflight(repo_root, {script.model for script in scripts})
        if errors:
            print("\nPreflight failed:")
            for error in errors:
                print(f"  - {error}")
            print("\nInstall the model runtimes and checkpoints described in README.md, then rerun this command.")
            return 2

    failures = []
    next_script_index = 0
    running: dict[Future[tuple[int, EvalScript, int]], int] = {}
    stop_scheduling = False

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        for gpu in args.gpus:
            if next_script_index >= len(scripts):
                break
            script_index = next_script_index
            future = executor.submit(
                run_script,
                repo_root,
                scripts[script_index],
                script_index + 1,
                len(scripts),
                gpu,
            )
            running[future] = gpu
            next_script_index += 1

        while running:
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                gpu = running.pop(future)
                index, script, returncode = future.result()
                if returncode:
                    failures.append((index, script.path, returncode))
                    if not args.continue_on_error:
                        stop_scheduling = True

                if not stop_scheduling and next_script_index < len(scripts):
                    script_index = next_script_index
                    next_future = executor.submit(
                        run_script,
                        repo_root,
                        scripts[script_index],
                        script_index + 1,
                        len(scripts),
                        gpu,
                    )
                    running[next_future] = gpu
                    next_script_index += 1

    if failures:
        print("\nFailed evaluations:")
        for _, path, returncode in sorted(failures):
            print(f"  - {path} (exit {returncode})")
        return 1

    print("\nAll selected simulation evaluations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
