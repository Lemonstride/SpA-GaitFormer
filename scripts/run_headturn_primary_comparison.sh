#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
GAIT=${SPA_GAITFORMER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
ROOT=${SPA_USER_ROOT:-/vepfs/group02/USER/fanzu}
PY=${PYTHON_BIN:-$ROOT/env/spa-gaitformer/bin/python}
SOURCE_RUN=${SOURCE_RUN_DIR:-$GAIT/runs/from_scratch_26cohort_complete25_walk_v1}
SOURCE_MANIFEST=${SOURCE_MANIFEST_PATH:-$SOURCE_RUN/manifests/all_walk_w10_s5.csv}
WORKBOOK=${CLINICAL_WORKBOOK:-$ROOT/datasets/SpA-MMD/total.xlsx}
RUN=${RUN_DIR:-$GAIT/runs/from_scratch_walk_headturn_primary20_v1}
BASE_CONFIG=${BASE_CONFIG_PATH:-$GAIT/configs/from_scratch_26_walk.yaml}
HEADTURN_CONFIG=${HEADTURN_CONFIG_PATH:-$GAIT/configs/from_scratch_26_walk_headturn.yaml}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DEVICE=${DEVICE:-cuda}

case "$(readlink -m "$RUN")" in "$ROOT"/*) ;; *) echo "Out-of-bound run path: $RUN" >&2; exit 2 ;; esac
mkdir -p "$RUN/manifests/binary" "$RUN/manifests/severity" "$RUN/logs"
echo PREPARING > "$RUN/status.txt"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "FAILED exit=$code" > "$RUN/status.txt"; fi' EXIT
cd "$GAIT"

MANIFEST="$RUN/manifests/all_walk_headturn_primary.csv"
"$PY" scripts/attach_headturn_features.py \
  --manifest "$SOURCE_MANIFEST" \
  --workbook "$WORKBOOK" \
  --mode primary \
  --output "$MANIFEST" \
  --audit-output "$RUN/headturn_manifest_audit.json"

for task in binary severity; do
  "$PY" -m spa_gaitformer.splits \
    --manifest "$MANIFEST" \
    --output-dir "$RUN/manifests/$task" \
    --task "$task" \
    --repeats 5 \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --seed 2026
done

{
  echo "created_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "cuda_device=$CUDA_DEVICE"
  echo "source_manifest=$SOURCE_MANIFEST"
  sha256sum "$SOURCE_MANIFEST" "$WORKBOOK" "$BASE_CONFIG" "$HEADTURN_CONFIG"
  "$PY" --version
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} > "$RUN/provenance.txt"

echo RUNNING > "$RUN/status.txt"
for task in binary severity; do
  for split in 0 1 2 3 4; do
    for variant in walk_only walk_headturn; do
      if [ "$variant" = walk_only ]; then config="$BASE_CONFIG"; else config="$HEADTURN_CONFIG"; fi
      output="$RUN/outputs/$variant/${task}_split_$split"
      metrics="$output/test_metrics.json"
      if [ -f "$metrics" ]; then
        echo "SKIP variant=$variant task=$task split=$split reason=complete"
        continue
      fi
      mkdir -p "$output"
      echo "TRAIN variant=$variant task=$task split=$split time=$(date -Is)"
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" -m spa_gaitformer.train \
        --config "$config" \
        --train-manifest "$RUN/manifests/$task/split_${split}_train.csv" \
        --val-manifest "$RUN/manifests/$task/split_${split}_val.csv" \
        --task "$task" \
        --output-dir "$output" \
        --device "$DEVICE"
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" -m spa_gaitformer.evaluate \
        --config "$config" \
        --manifest "$RUN/manifests/$task/split_${split}_test.csv" \
        --task "$task" \
        --checkpoint "$output/best.pt" \
        --device "$DEVICE" \
        --output "$metrics"
      echo "DONE variant=$variant task=$task split=$split time=$(date -Is)"
    done
  done
done

"$PY" -m spa_gaitformer.summarize_headturn_comparison \
  --run-dir "$RUN" \
  --repeats 5 \
  --output "$RUN/subject_headturn_comparison_summary.json"
echo DONE > "$RUN/status.txt"
date -Is > "$RUN/completed_at.txt"

