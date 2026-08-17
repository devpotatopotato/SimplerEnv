"""Train VPP's action model on the canonical Bridge LeRobot repositories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic


def _atomic_torch_save(torch, value, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def _policy_state_dict(model) -> dict:
    """Keep the learned VPP policy while omitting reloadable frozen encoders."""

    prefixes = ("Video_Former.", "model.")
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }
    if not state:
        raise RuntimeError("VPP policy checkpoint filter selected no tensors")
    return state


def _load_policy_state_dict(model, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        name
        for name in missing
        if name.startswith(("Video_Former.", "model."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "incompatible VPP policy checkpoint: "
            f"missing_policy={invalid_missing[:10]}, unexpected={unexpected[:10]}"
        )


def _mean_validation_loss(accelerator, model, loader, max_batches: int) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            torch.manual_seed(10_000 + batch_index)
            loss = model(batch)
            gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
            losses.append(float(gathered.mean().cpu()))
    model.train()
    return float(sum(losses) / len(losses)) if losses else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpp-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--val-dataset", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--video-model-path", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2, help="per-process batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (
        args.vpp_root,
        args.train_dataset,
        args.val_dataset,
        args.video_model_path,
        args.text_encoder_path,
    ):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    if min(args.max_steps, args.batch_size, args.gradient_accumulation, args.save_interval) < 1:
        parser.error("step, batch, accumulation, and save values must be positive")

    sys.path.insert(0, str(args.vpp_root.resolve()))
    import hydra
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from simpler_training.vpp_dataset import CanonicalLeRobotVPPDataset

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation, mixed_precision="bf16")
    torch.manual_seed(args.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "config.yaml"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"

    config = OmegaConf.load(args.template_config)
    config.seed = args.seed
    config.model.pretrained_model_path = str(args.video_model_path.resolve())
    config.model.text_encoder_path = str(args.text_encoder_path.resolve())
    config.model.lr_scheduler.lr_scheduler.total_steps = args.max_steps
    scale = tuple(float(value) for value in config.action_adapter.output_scale)
    if accelerator.is_main_process:
        OmegaConf.save(config, resolved_config_path)
    accelerator.wait_for_everyone()

    train_dataset = CanonicalLeRobotVPPDataset(
        args.train_dataset,
        action_horizon=int(config.model.action_seq_len),
        action_scale=scale,
        sample_stride=args.sample_stride,
    )
    val_dataset = CanonicalLeRobotVPPDataset(
        args.val_dataset,
        action_horizon=int(config.model.action_seq_len),
        action_scale=scale,
        sample_stride=max(args.sample_stride, 2),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise RuntimeError("VPP training dataset is smaller than one batch")

    model = hydra.utils.instantiate(config.model)
    model = model.to(accelerator.device)
    process_device = getattr(model, "process_device", None)
    if callable(process_device):
        process_device()
    optimizer_result = model.configure_optimizers()
    optimizer = optimizer_result["optimizer"]
    scheduler = optimizer_result["lr_scheduler"]["scheduler"]
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    completed_steps = 0
    best_loss = float("inf")
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu")
        _load_policy_state_dict(accelerator.unwrap_model(model), checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        completed_steps = int(checkpoint["completed_steps"])
        best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        accelerator.print(f"Resuming VPP from step {completed_steps}")
        if completed_steps >= args.max_steps and not final_path.is_file():
            if accelerator.is_main_process:
                recovered_final = {
                    "model": checkpoint["model"],
                    "completed_steps": completed_steps,
                    "validation_loss": checkpoint.get("validation_loss"),
                    "config": checkpoint.get(
                        "config", OmegaConf.to_container(config, resolve=True)
                    ),
                }
                _atomic_torch_save(torch, recovered_final, final_path)
            accelerator.wait_for_everyone()
    if completed_steps >= args.max_steps and final_path.is_file():
        accelerator.print(f"VPP training already complete: {final_path}")
    else:
        model.train()
        start = monotonic()
        running_loss = 0.0
        running_updates = 0
        while completed_steps < args.max_steps:
            for batch in train_loader:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss = model(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu())
                if not accelerator.sync_gradients:
                    continue
                completed_steps += 1
                running_updates += 1
                if completed_steps % 50 == 0 and accelerator.is_main_process:
                    elapsed = max(monotonic() - start, 1e-6)
                    print(
                        f"VPP step {completed_steps}/{args.max_steps}: "
                        f"loss={running_loss / max(running_updates, 1):.6f}, "
                        f"updates/s={running_updates / elapsed:.3f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_updates = 0
                    start = monotonic()
                should_save = completed_steps % args.save_interval == 0 or completed_steps == args.max_steps
                if should_save:
                    val_loss = _mean_validation_loss(
                        accelerator, model, val_loader, args.max_val_batches
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        state = {
                            # The SVD and text encoders are frozen and already
                            # identified by the resolved config. Saving only the
                            # action policy avoids duplicating those large
                            # public backbones in every checkpoint.
                            "model": _policy_state_dict(accelerator.unwrap_model(model)),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "completed_steps": completed_steps,
                            "best_validation_loss": min(best_loss, val_loss),
                            "validation_loss": val_loss,
                            "config": OmegaConf.to_container(config, resolve=True),
                        }
                        _atomic_torch_save(torch, state, last_path)
                        if val_loss <= best_loss:
                            best_loss = val_loss
                            deployment_state = {
                                "model": state["model"],
                                "completed_steps": completed_steps,
                                "validation_loss": val_loss,
                                "config": state["config"],
                            }
                            _atomic_torch_save(torch, deployment_state, best_path)
                        if completed_steps == args.max_steps:
                            # Use the predeclared final step for the primary
                            # comparison, matching π0.5. best.pt remains a
                            # separately labeled validation diagnostic.
                            final_state = {
                                "model": state["model"],
                                "completed_steps": completed_steps,
                                "validation_loss": val_loss,
                                "config": state["config"],
                            }
                            _atomic_torch_save(torch, final_state, final_path)
                        print(f"VPP checkpoint step={completed_steps}, val_loss={val_loss:.6f}", flush=True)
                    accelerator.wait_for_everyone()
                if completed_steps >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not final_path.is_file():
            raise RuntimeError(f"VPP did not create {final_path}")
        artifact = {
            "model": "vpp",
            "config": str(resolved_config_path),
            "checkpoint": str(final_path),
            "best_validation_checkpoint": str(best_path),
            "video_model_path": str(args.video_model_path.resolve()),
            "text_encoder_path": str(args.text_encoder_path.resolve()),
            "train_steps": args.max_steps,
            "effective_global_batch_size": (
                args.batch_size * accelerator.num_processes * args.gradient_accumulation
            ),
            "declared_training_samples": (
                args.max_steps * args.batch_size * accelerator.num_processes * args.gradient_accumulation
            ),
            "seed": args.seed,
            "adaptation_method": "frozen_video_backbone_action_head",
        }
        with open(output_dir / "artifacts.json", "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
