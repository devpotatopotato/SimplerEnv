"""Download a resumable Hugging Face snapshot and mark it complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    completion_marker = destination / ".snapshot_complete.json"
    if completion_marker.is_file():
        print(f"Using existing snapshot: {destination}")
        return
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {destination}", flush=True)
    snapshot_download(repo_id=args.repo_id, local_dir=destination)
    temporary_marker = completion_marker.with_suffix(".json.tmp")
    temporary_marker.write_text(
        json.dumps({"repo_id": args.repo_id}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(completion_marker)
    print(f"Snapshot complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
