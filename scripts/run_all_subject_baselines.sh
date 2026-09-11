#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GAIT=${SPA_GAITFORMER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PY=${PYTHON_BIN:-python}
RUN=${RUN_DIR:-$GAIT/runs/from_scratch_26cohort_complete25_walk_v1}
CONFIG=${CONFIG_PATH:-$GAIT/configs/from_scratch_26_walk.yaml}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DEVICE=${DEVICE:-cuda}

cd "$GAIT"
mkdir -p "$RUN"
echo RUNNING > "$RUN/all_subject_baselines_status.txt"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "FAILED exit=$code" > "$RUN/all_subject_baselines_status.txt"; fi' EXIT

for task in binary severity; do
  for split in 0 1 2 3 4; do
    output="$RUN/outputs/${task}_split_${split}"
    metrics="$output/test_metrics.json"
    if [ -f "$metrics" ]; then
      echo "BASELINE_SKIP task=$task split=$split reason=complete"
      continue
    fi
    mkdir -p "$output"
    echo "BASELINE_TRAIN task=$task split=$split"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" -m spa_gaitformer.train \
      --config "$CONFIG" \
      --train-manifest "$RUN/manifests/$task/split_${split}_train.csv" \
      --val-manifest "$RUN/manifests/$task/split_${split}_val.csv" \
      --task "$task" \
      --output-dir "$output" \
      --device "$DEVICE"
    echo "BASELINE_TEST task=$task split=$split"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" -m spa_gaitformer.evaluate \
      --config "$CONFIG" \
      --manifest "$RUN/manifests/$task/split_${split}_test.csv" \
      --task "$task" \
      --checkpoint "$output/best.pt" \
      --device "$DEVICE" \
      --output "$metrics"
    echo "BASELINE_DONE task=$task split=$split"
  done
done

"$PY" -m spa_gaitformer.summarize_subject_runs \
  --run-dir "$RUN" \
  --repeats 5 \
  --output "$RUN/subject_baseline_summary.json"
echo DONE > "$RUN/all_subject_baselines_status.txt"
