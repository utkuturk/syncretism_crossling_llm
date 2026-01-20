#!/usr/bin/env bash
set -euo pipefail

size_mb="${1:-50}"
out_dir="${2:-}"

shopt -s nullglob
files=(data/nsubj_root_sentences_*.csv)
if (( ${#files[@]} == 0 )); then
  echo "No nsubj_root_sentences_*.csv files found in data/"
  exit 1
fi

for f in "${files[@]}"; do
  if [[ -n "$out_dir" ]]; then
    python3 scripts/split_file.py "$f" --size-mb "$size_mb" --out-dir "$out_dir"
  else
    python3 scripts/split_file.py "$f" --size-mb "$size_mb"
  fi
done
