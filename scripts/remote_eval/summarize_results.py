#!/usr/bin/env python3
"""Print a comparable summary for one remote-evaluation run tag."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

PREFERRED_MODEL_ORDER = {"pi05": 0, "vpp": 1, "cosmos3": 2}
REQUIRED_ARTIFACTS = ("evaluation_config.json", "server_metadata.json", "episodes.jsonl", "summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _episode_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"episode line {line_number} is not a JSON object")
            episodes.append(value)
    return episodes


def _expected_episode_count(config: dict[str, Any]) -> int:
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError("tasks is required")
    task_settings = config.get("task_settings")
    if isinstance(task_settings, dict) and task_settings:
        policy_seeds = config.get("policy_seeds") or [0]
        if not isinstance(policy_seeds, list) or not policy_seeds:
            raise ValueError("policy_seeds must be a non-empty list")
        total = 0
        for task in tasks:
            settings = task_settings.get(task)
            if not isinstance(settings, dict) or not isinstance(settings.get("object_episode_ids"), list):
                raise TypeError(f"task_settings is missing object_episode_ids for {task}")
            total += len(settings["object_episode_ids"]) * len(policy_seeds)
        return total
    episodes_per_task = config.get("episodes_per_task")
    if not isinstance(episodes_per_task, int):
        raise TypeError("episodes_per_task is required")
    return len(tasks) * episodes_per_task


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
    metadata: dict[str, Any] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = _read_json(summary_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid summary.json: {error}")

    metadata_path = run_dir / "server_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = _read_json(metadata_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid server_metadata.json: {error}")

    expected_episodes: int | None = None
    config_path = run_dir / "evaluation_config.json"
    if config_path.is_file():
        try:
            config = _read_json(config_path)
            expected_episodes = _expected_episode_count(config)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid evaluation_config.json: {error}")

    recorded_episodes: int | None = None
    episodes: list[dict[str, Any]] = []
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.is_file():
        try:
            episodes = _read_episodes(episodes_path)
            recorded_episodes = len(episodes)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
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
        "metadata": metadata,
        "episodes_data": episodes,
        "summary": summary,
    }


def _pair_key(episode: dict[str, Any]):
    task = episode.get("task")
    if task is None:
        return None
    object_episode_id = episode.get("object_episode_id")
    if object_episode_id is not None:
        return (task, "object", object_episode_id, episode.get("environment_seed"), episode.get("policy_seed"))
    return (task, "legacy", episode.get("episode_index"), episode.get("seed"))


def _paired_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if not left["complete"] or not right["complete"]:
        return None
    left_outcomes = {
        key: int(bool(episode.get("success")))
        for episode in left["episodes_data"]
        if (key := _pair_key(episode)) is not None
    }
    right_outcomes = {
        key: int(bool(episode.get("success")))
        for episode in right["episodes_data"]
        if (key := _pair_key(episode)) is not None
    }
    keys = sorted(set(left_outcomes) & set(right_outcomes))
    if not keys:
        return None
    differences = np.asarray([left_outcomes[key] - right_outcomes[key] for key in keys], dtype=np.float64)
    delta = float(np.mean(differences))
    if len(differences) == 1:
        interval = [delta, delta]
    else:
        rng = np.random.default_rng(0)
        indexes = rng.integers(0, len(differences), size=(10_000, len(differences)))
        bootstrap = differences[indexes].mean(axis=1)
        interval = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    return {
        "model_a": left["model"],
        "model_b": right["model"],
        "paired_episodes": len(keys),
        "success_rate_delta_a_minus_b": delta,
        "confidence_interval_95": interval,
        "a_success_b_failure": int(np.sum(differences == 1)),
        "a_failure_b_success": int(np.sum(differences == -1)),
        "same_comparison_group": bool(left["metadata"].get("comparison_group")) and (
            left["metadata"].get("comparison_group") == right["metadata"].get("comparison_group")
        ),
    }


def build_report(results_dir: Path, run_tag: str, models: Iterable[str]) -> dict[str, Any]:
    model_results = [load_model_result(results_dir, run_tag, model) for model in _ordered_models(models)]
    paired = [
        comparison
        for left, right in itertools.combinations(model_results, 2)
        if (comparison := _paired_comparison(left, right)) is not None
    ]
    for result in model_results:
        result.pop("episodes_data", None)
    return {
        "run_tag": run_tag,
        "results_dir": str(results_dir.resolve()),
        "complete": bool(model_results) and all(item["complete"] for item in model_results),
        "models": model_results,
        "paired_comparisons": paired,
    }


def _percent(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{100.0 * value:.1f}%"


def _milliseconds(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:.1f}"


def _interval(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
        return "-"
    return f"[{100.0 * value[0]:.1f}, {100.0 * value[1]:.1f}]%"


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
                str(item["metadata"].get("comparison_group", "-")),
                success_count,
                _percent(summary.get("success_rate")),
                _interval(summary.get("confidence_interval_95")),
                _percent(summary.get("safety_clip_rate")),
                _milliseconds(summary.get("mean_server_inference_ms")),
                _milliseconds(summary.get("mean_round_trip_ms")),
            ]
        )
    lines.append(
        _table(
            ["model", "status", "training_group", "success", "rate", "95% CI", "clipped", "inference_ms", "round_trip_ms"],
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
                    row.append(
                        f"{successes}/{episodes} ({_percent(metrics.get('success_rate'))}; "
                        f"{_interval(metrics.get('confidence_interval_95'))})"
                    )
                else:
                    row.append("-")
            task_rows.append(row)
        lines.append(_table(["task", *(item["model"] for item in model_results)], task_rows))

    comparisons = report.get("paired_comparisons", [])
    if comparisons:
        lines.extend(["", "Paired success-rate differences"])
        comparison_rows = []
        for comparison in comparisons:
            comparison_rows.append(
                [
                    f"{comparison['model_a']} - {comparison['model_b']}",
                    str(comparison["paired_episodes"]),
                    _percent(comparison["success_rate_delta_a_minus_b"]),
                    _interval(comparison["confidence_interval_95"]),
                    "matched" if comparison["same_comparison_group"] else "different training regimes",
                ]
            )
        lines.append(_table(["models", "pairs", "delta", "bootstrap 95% CI", "interpretation"], comparison_rows))

    groups = {
        str(item["metadata"].get("comparison_group"))
        for item in model_results
        if item["metadata"].get("comparison_group")
    }
    if len(groups) > 1:
        lines.extend(
            [
                "",
                "Comparability note",
                (
                    "π0.5/VPP share the local Bridge adaptation. Cosmos3-Edge uses its released native Bridge "
                    "pretraining; it shares the evaluation protocol, not the adaptation-data budget."
                ),
            ]
        )

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
