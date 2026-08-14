"""Run normalization and π0.5 LoRA training through pinned OpenPI code."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from simpler_training.openpi_config import CONFIG_NAME, build_config, register_config


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.exists():
        return []
    return sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/simpler_bridge_train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp-name", default="bridge_lora")
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--norm-max-frames", type=int)
    args = parser.parse_args()
    if args.train_steps < 2 or args.batch_size < 1 or args.save_interval < 1:
        parser.error("training steps, batch size, and save interval must be positive")

    root = args.openpi_root.resolve()
    if not (root / "scripts/train.py").is_file():
        parser.error(f"not an OpenPI checkout: {root}")
    output_root = args.output_root.resolve()
    assets_root = output_root / "assets"
    checkpoints_root = output_root / "checkpoints"
    checkpoint_dir = checkpoints_root / CONFIG_NAME / args.exp_name
    final_step = args.train_steps - 1
    final_checkpoint = checkpoint_dir / str(final_step)

    existing_steps = _checkpoint_steps(checkpoint_dir)
    if final_checkpoint.is_dir():
        print(f"π0.5 training already complete: {final_checkpoint}")
    else:
        # OpenPI's checkpoint manager also handles a directory created before
        # the first checkpoint was reached, so request resume whenever the run
        # directory exists (not only when it already has numeric steps).
        resume = checkpoint_dir.exists()
        config = build_config(
            repo_id=args.repo_id,
            exp_name=args.exp_name,
            assets_base_dir=str(assets_root),
            checkpoint_base_dir=str(checkpoints_root),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            # The pinned upstream compute_norm_stats.py is loaded dynamically
            # and defines its RemoveStrings transform in that script module.
            # Spawned DataLoader workers cannot import that dynamic module, so
            # normalization must run in the main process. The training config
            # is rebuilt below with the requested worker count.
            num_workers=0,
            save_interval=args.save_interval,
            resume=resume,
        )
        register_config(config)
        norm_path = assets_root / CONFIG_NAME / args.repo_id / "norm_stats.json"
        if not norm_path.is_file():
            print(f"Computing OpenPI normalization statistics for {args.repo_id}", flush=True)
            norm_script = _load_script(root / "scripts/compute_norm_stats.py", "simpler_openpi_norm_stats")
            norm_script.main(CONFIG_NAME, max_frames=args.norm_max_frames)
        else:
            print(f"Using existing normalization statistics: {norm_path}", flush=True)

        # Rebuild after norm-stat creation so DataConfigFactory sees the assets.
        config = build_config(
            repo_id=args.repo_id,
            exp_name=args.exp_name,
            assets_base_dir=str(assets_root),
            checkpoint_base_dir=str(checkpoints_root),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_interval=args.save_interval,
            resume=resume,
        )
        register_config(config)
        print(f"{'Resuming' if resume else 'Starting'} π0.5 LoRA training", flush=True)
        train_script = _load_script(root / "scripts/train.py", "simpler_openpi_train")
        train_script.main(config)

    if not final_checkpoint.is_dir():
        steps = _checkpoint_steps(checkpoint_dir)
        raise RuntimeError(f"π0.5 did not create expected checkpoint {final_checkpoint}; available={steps}")
    artifact = {
        "model": "pi0.5",
        "config_name": CONFIG_NAME,
        "checkpoint": str(final_checkpoint),
        "repo_id": args.repo_id,
        "train_steps": args.train_steps,
        "adaptation_method": "lora",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "artifacts.json", "w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
