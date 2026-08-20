"""Run the published SimplerEnv simulation-evaluation matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import os
from pathlib import Path
import shlex
import subprocess
import sys
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
    parser.add_argument(
        "--rt1-python",
        help="Python executable used by RT-1 scripts (default: the runner's Python)",
    )
    parser.add_argument(
        "--octo-python",
        help="Python executable used by Octo scripts (default: the runner's Python)",
    )
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


def resolve_model_pythons(repo_root: Path, args: argparse.Namespace) -> dict[str, Path]:
    model_pythons = {}
    for model in ("rt1", "octo"):
        configured = getattr(args, f"{model}_python")
        python = Path(configured).expanduser() if configured else Path(sys.executable)
        if not python.is_absolute():
            python = repo_root / python
        model_pythons[model] = python.absolute()
    return model_pythons


def find_missing_modules(python: Path, modules: tuple[str, ...]) -> list[str]:
    check_code = (
        "import importlib.util, sys; "
        "print('\\n'.join(name for name in sys.argv[1:] if importlib.util.find_spec(name) is None))"
    )
    result = subprocess.run(
        [str(python), "-c", check_code, *modules],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(detail)
    return [line for line in result.stdout.splitlines() if line]


def preflight(repo_root: Path, models: set[str], model_pythons: dict[str, Path]) -> list[str]:
    errors = []
    for model in sorted(models):
        python = model_pythons[model]
        if not python.is_file() or not os.access(python, os.X_OK):
            errors.append(f"{model}: Python executable not found or not executable: {python}")
            continue
        try:
            missing = find_missing_modules(python, MODEL_MODULES[model])
        except RuntimeError as error:
            errors.append(f"{model}: could not inspect {python}: {error}")
            continue
        if missing:
            errors.append(f"{model}: missing Python modules in {python}: {', '.join(missing)}")

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
    python: Path,
) -> tuple[int, EvalScript, int]:
    child_env = os.environ.copy()
    child_env["SIMPLER_GPU_ID"] = str(gpu)
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if script.model == "octo":
        child_env.setdefault("SIMPLER_DISABLE_TF_GPU", "1")
    child_env["PATH"] = str(python.parent) + os.pathsep + child_env.get("PATH", "")
    child_env["PYTHONPATH"] = str(repo_root) + os.pathsep + child_env.get("PYTHONPATH", "")
    command = ["bash", script.path]
    print(f"\n[{index}/{total}][GPU {gpu}][Python {python}] {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=repo_root, env=child_env)
    return index, script, result.returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_pythons = resolve_model_pythons(repo_root, args)
    scripts = select_scripts(args)

    if not scripts:
        print("No evaluation scripts match the requested filters.")
        return 2

    print(f"Selected {len(scripts)} evaluation scripts:")
    for script in scripts:
        print(f"  {script.model:4} {script.suite:6} {script.setup:19} {script.path}")

    print(f"GPU workers: {', '.join(map(str, args.gpus))}")
    for model in sorted({script.model for script in scripts}):
        print(f"{model.upper()} Python: {model_pythons[model]}")

    if args.dry_run:
        print("\nInitial GPU assignment:")
        for index, script in enumerate(scripts[: len(args.gpus)], start=1):
            gpu = args.gpus[index - 1]
            print(f"  [GPU {gpu}][Python {model_pythons[script.model]}] bash {script.path}")
        return 0

    if not args.skip_preflight:
        errors = preflight(repo_root, {script.model for script in scripts}, model_pythons)
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
                model_pythons[scripts[script_index].model],
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
                        model_pythons[scripts[script_index].model],
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
