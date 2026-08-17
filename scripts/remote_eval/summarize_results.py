#!/usr/bin/env python3
"""Print a comparable summary for one remote-evaluation run tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


PREFERRED_MODEL_ORDER = {"pi05": 0, "vpp": 1, "cosmos3": 2}
REQUIRED_ARTIFACTS = ("evaluation_config.json", "server_metadata.json", "episodes.jsonl", "summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _episode_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _ordered_models(models: Iterable[str]) -> list[str]:
    return sorted(set(models), key=lambda item: (PREFERRED_MODEL_ORDER.get(item, 100), item))


def discover_models(results_dir: Path, run_tag: str) -> list[str]:
    suffix = f"-{run_tag}"
    models = {
        path.name[: -len(suffix)]
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix) and path.name != suffix
    }
    launcher_dir = results_dir / "_launcher" / run_tag
    if launcher_dir.is_dir():
        for path in launcher_dir.glob("*_server.log"):
            models.add(path.name.removesuffix("_server.log"))
        for path in launcher_dir.glob("*_evaluator.log"):
            models.add(path.name.removesuffix("_evaluator.log"))
    return _ordered_models(models)


def latest_run_tag(results_dir: Path) -> str:
    launcher_dir = results_dir / "_launcher"
    candidates = [path for path in launcher_dir.iterdir() if path.is_dir()] if launcher_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no launcher runs found under {launcher_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def load_model_result(results_dir: Path, run_tag: str, model: str) -> dict[str, Any]:
    run_dir = results_dir / f"{model}-{run_tag}"
    errors: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (run_dir / artifact).is_file():
            errors.append(f"missing {artifact}")

    summary: dict[str, Any] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = _read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid summary.json: {error}")

    expected_episodes: int | None = None
    config_path = run_dir / "evaluation_config.json"
    if config_path.is_file():
        try:
            config = _read_json(config_path)
            tasks = config.get("tasks")
            episodes_per_task = config.get("episodes_per_task")
            if not isinstance(tasks, list) or not isinstance(episodes_per_task, int):
                raise ValueError("tasks and episodes_per_task are required")
            expected_episodes = len(tasks) * episodes_per_task
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid evaluation_config.json: {error}")

    recorded_episodes: int | None = None
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.is_file():
        try:
            recorded_episodes = _episode_count(episodes_path)
        except OSError as error:
            errors.append(f"cannot read episodes.jsonl: {error}")

    summary_episodes = summary.get("episodes")
    if summary and not isinstance(summary_episodes, int):
        errors.append("summary.json has no integer episodes field")
    if isinstance(summary_episodes, int) and recorded_episodes is not None and summary_episodes != recorded_episodes:
        errors.append(f"summary reports {summary_episodes} episodes but episodes.jsonl has {recorded_episodes}")
    if isinstance(summary_episodes, int) and expected_episodes is not None and summary_episodes != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes but summary reports {summary_episodes}")

    return {
        "model": model,
        "run_dir": str(run_dir),
        "complete": not errors,
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "expected_episodes": expected_episodes,
        "recorded_episodes": recorded_episodes,
        "summary": summary,
    }


def build_report(results_dir: Path, run_tag: str, models: Iterable[str]) -> dict[str, Any]:
    model_results = [load_model_result(results_dir, run_tag, model) for model in _ordered_models(models)]
    return {
        "run_tag": run_tag,
        "results_dir": str(results_dir.resolve()),
        "complete": bool(model_results) and all(item["complete"] for item in model_results),
        "models": model_results,
    }


def _percent(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{100.0 * value:.1f}%"


def _milliseconds(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    return "\n".join([render(headers), render(["-" * width for width in widths]), *(render(row) for row in rows)])


def format_report(report: dict[str, Any]) -> str:
    model_results = report["models"]
    lines = [
        f"Evaluation: {report['run_tag']}",
        f"Results: {report['results_dir']}",
        f"Status: {sum(item['complete'] for item in model_results)}/{len(model_results)} models complete",
        "",
        "Overall",
    ]
    overall_rows = []
    for item in model_results:
        summary = item["summary"]
        successes = summary.get("successes")
        episodes = summary.get("episodes")
        success_count = f"{successes}/{episodes}" if isinstance(successes, int) and isinstance(episodes, int) else "-"
        overall_rows.append(
            [
                item["model"],
                item["status"],
                success_count,
                _percent(summary.get("success_rate")),
                _percent(summary.get("safety_clip_rate")),
                _milliseconds(summary.get("mean_server_inference_ms")),
                _milliseconds(summary.get("mean_round_trip_ms")),
            ]
        )
    lines.append(
        _table(
            ["model", "status", "success", "rate", "clipped", "inference_ms", "round_trip_ms"],
            overall_rows,
        )
    )

    task_names = set()
    for item in model_results:
        per_task = item["summary"].get("per_task")
        if isinstance(per_task, dict):
            task_names.update(per_task)
    tasks = sorted(task_names)
    if tasks:
        lines.extend(["", "Per-task success"])
        task_rows = []
        for task in tasks:
            row = [task]
            for item in model_results:
                metrics = item["summary"].get("per_task", {}).get(task, {})
                successes = metrics.get("successes")
                episodes = metrics.get("episodes")
                if isinstance(successes, int) and isinstance(episodes, int):
                    row.append(f"{successes}/{episodes} ({_percent(metrics.get('success_rate'))})")
                else:
                    row.append("-")
            task_rows.append(row)
        lines.append(_table(["task", *(item["model"] for item in model_results)], task_rows))

    problems = [(item["model"], error) for item in model_results for error in item["errors"]]
    if problems:
        lines.extend(["", "Problems"])
        lines.extend(f"- {model}: {error}" for model, error in problems)
    return "\n".join(lines)


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("models must contain at least one name")
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag", nargs="?", help="RUN_TAG used by run.sh; defaults to the latest launcher run")
    parser.add_argument("--results-dir", default="results/remote_eval", help="remote evaluation result directory")
    parser.add_argument("--models", type=parse_models, help="comma-separated models; defaults to discovered runs")
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any selected run is incomplete")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    try:
        run_tag = args.run_tag or latest_run_tag(results_dir)
        models = args.models or discover_models(results_dir, run_tag)
    except OSError as error:
        parser.error(str(error))
    if not models:
        parser.error(f"no model runs ending in -{run_tag} found under {results_dir}")

    report = build_report(results_dir, run_tag, models)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_report(report))
    return int(args.strict and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
