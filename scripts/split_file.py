#!/usr/bin/env python3
import argparse
from pathlib import Path


DEFAULT_CHUNK_MB = 50


def split_file(path: Path, chunk_bytes: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    part_idx = 0
    with path.open("rb") as src:
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            part_path = out_dir / f"{path.name}.part{part_idx:03d}"
            with part_path.open("wb") as out:
                out.write(chunk)
            part_idx += 1
    return part_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a file into fixed-size parts.")
    parser.add_argument("file", help="Path to the file to split.")
    parser.add_argument(
        "--size-mb",
        type=int,
        default=DEFAULT_CHUNK_MB,
        help=f"Chunk size in MB (default: {DEFAULT_CHUNK_MB}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for parts (default: same dir as input).",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Input file not found: {file_path}")

    out_dir = Path(args.out_dir) if args.out_dir else file_path.parent
    chunk_bytes = args.size_mb * 1024 * 1024

    parts = split_file(file_path, chunk_bytes, out_dir)
    print(f"Wrote {parts} parts to {out_dir}")


if __name__ == "__main__":
    main()
