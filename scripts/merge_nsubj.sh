#!/usr/bin/env bash
set -euo pipefail

parts_dir="${1:-data}"
output_dir="${2:-data}"

mkdir -p "$output_dir"

shopt -s nullglob
parts=( "$parts_dir"/nsubj_root_sentences_*.csv.part* )
if (( ${#parts[@]} == 0 )); then
  echo "No parts found in ${parts_dir}"
  exit 1
fi

declare -A bases=()
for part in "${parts[@]}"; do
  base="${part%%.part*}"
  bases["$base"]=1
done

for base in "${!bases[@]}"; do
  out="${output_dir}/$(basename "$base")"
  python3 scripts/merge_parts.py "$base" --parts-dir "$parts_dir" --output "$out"
done
