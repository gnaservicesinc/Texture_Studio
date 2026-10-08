#!/bin/bash
# Standardize source folders, create native paired crops, verify source slices.
set -euo pipefail
material_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
material_repo_dir="$(cd "$material_script_dir/.." && pwd)"
material_dataset_dir="${1:-$material_repo_dir/../material-dataset}"
material_crop_size="${2:-1024}"
case "$material_crop_size" in
  1024) material_output_dir="$material_dataset_dir" ;;
  2048) material_output_dir="$material_dataset_dir/prepared-2048" ;;
  *) printf '%s\n' 'Usage: prepare-materials.sh DATASET_DIRECTORY [1024|2048]' >&2; exit 2 ;;
esac
material_python="${IPDE_MATERIAL_PYTHON:-$material_repo_dir/.venv/bin/python}"
material_report_dir="$material_output_dir/reports"
mkdir -p "$material_report_dir"
material_provenance_args=()
if [[ -f "$material_dataset_dir/reports/published-source-audit.json" ]]; then
  material_provenance_args=(--published-source-audit "$material_dataset_dir/reports/published-source-audit.json")
fi
"$material_python" "$material_script_dir/material_dataset.py" normalize-names \
  --sources "$material_dataset_dir/sources" --crop-size "$material_crop_size" --apply \
  --archive-identical-duplicates "$material_dataset_dir/archived-duplicates" \
  --report "$material_report_dir/latest-name-normalization.json"
"$material_python" "$material_script_dir/material_dataset.py" prepare-region-splits \
  --sources "$material_dataset_dir/sources" --dataset "$material_output_dir" --crop-size "$material_crop_size" \
  --migrate-split-policy \
  --report "$material_report_dir/latest-preparation.json" "${material_provenance_args[@]}"
"$material_python" "$material_script_dir/material_dataset.py" verify \
  --dataset "$material_output_dir" --check-sources \
  --report "$material_report_dir/latest-crop-verification.json"
