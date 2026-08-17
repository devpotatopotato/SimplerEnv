"""Create an auditable manifest for the declared three-model comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--native", action="append", default=[], metavar="MODEL=CHECKPOINT")
    args = parser.parse_args()

    dataset_manifest = _read_json(args.dataset_manifest)
    models = {}
    for value in args.artifact:
        try:
            model, path_value = value.split("=", 1)
        except ValueError:
            parser.error(f"invalid --artifact {value!r}; expected MODEL=PATH")
        path = Path(path_value)
        models[model] = {
            "training_regime": "shared_local_bridge_adaptation",
            "artifact": _read_json(path),
            "artifact_manifest": str(path.resolve()),
            "artifact_manifest_sha256": _sha256(path),
            "seed": args.seed,
        }
    for value in args.native:
        try:
            model, checkpoint = value.split("=", 1)
        except ValueError:
            parser.error(f"invalid --native {value!r}; expected MODEL=CHECKPOINT")
        models[model] = {
            "training_regime": "released_native_pretraining",
            "checkpoint": checkpoint,
            "local_training": False,
            "seed": None,
        }

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "predeclared_final_checkpoint",
        "dataset": {
            **dataset_manifest,
            "manifest": str(args.dataset_manifest.resolve()),
            "manifest_sha256": _sha256(args.dataset_manifest),
        },
        "models": models,
        "comparability": {
            "shared_bridge_adaptation": sorted(
                model for model, value in models.items()
                if value["training_regime"] == "shared_local_bridge_adaptation"
            ),
            "native_pretrained_bridge": sorted(
                model for model, value in models.items()
                if value["training_regime"] == "released_native_pretraining"
            ),
            "warning": (
                "Models in different groups share the evaluation protocol but not the adaptation-data budget; "
                "do not present their score difference as a matched-training comparison."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
