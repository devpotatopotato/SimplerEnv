"""Update selected shell assignments while preserving comments and other models."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`") + '"'


def update(path: Path, assignments: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    output = []
    for line in lines:
        match = _ASSIGNMENT.match(line)
        key = match.group(1) if match else None
        if key in assignments:
            output.append(f"{key}={_quote(assignments[key])}")
            seen.add(key)
        else:
            output.append(line)
    missing = [key for key in assignments if key not in seen]
    if missing and output and output[-1] != "":
        output.append("")
    output.extend(f"{key}={_quote(assignments[key])}" for key in missing)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("assignments", nargs="+")
    args = parser.parse_args()
    values: dict[str, str] = {}
    for assignment in args.assignments:
        if "=" not in assignment:
            parser.error(f"expected KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if not _ASSIGNMENT.match(key + "="):
            parser.error(f"invalid key {key!r}")
        values[key] = value
    update(args.path, values)


if __name__ == "__main__":
    main()
