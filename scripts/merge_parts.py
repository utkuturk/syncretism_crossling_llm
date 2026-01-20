#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path


PART_RE = re.compile(r"\.part(\d+)$")


def part_key(path: Path) -> int:
    match = PART_RE.search(path.name)
    if not match:
        return -1
    return int(match.group(1))


def merge_parts(base_path: Path, out_path: Path, parts_dir: Path) -> int:
    prefix = f"{base_path.name}.part"
    parts = sorted(parts_dir.glob(f"{prefix}*"), key=part_key)
    parts = [p for p in parts if PART_RE.search(p.name)]
    if not parts:
        raise SystemExit(f"No parts found for {base_path.name} in {parts_dir}")

    with out_path.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)
    return len(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge file parts back into a file.")
    parser.add_argument(
        "base_file",
        help="Original filename used for parts (e.g., data/file.csv).",
    )
    parser.add_argument(
        "--parts-dir",
        default=None,
        help="Directory containing parts (default: same dir as base_file).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for merged file (default: base_file).",
    )
    args = parser.parse_args()

    base_path = Path(args.base_file)
    parts_dir = Path(args.parts_dir) if args.parts_dir else base_path.parent
    out_path = Path(args.output) if args.output else base_path

    count = merge_parts(base_path, out_path, parts_dir)
    print(f"Merged {count} parts into {out_path}")


if __name__ == "__main__":
    main()
