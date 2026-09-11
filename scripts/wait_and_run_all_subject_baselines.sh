#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GAIT=${SPA_GAITFORMER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
RUN=${RUN_DIR:-$GAIT/runs/from_scratch_26cohort_complete25_walk_v1}
MAIN_PID_FILE=${MAIN_PID_FILE:-$RUN/pid.txt}

echo WAITING > "$RUN/continuation_status.txt"
while true; do
  main_pid=$(cat "$MAIN_PID_FILE" 2>/dev/null || true)
  if [ -n "$main_pid" ] && kill -0 "$main_pid" 2>/dev/null; then
    sleep 60
    continue
  fi
  break
done

main_status=$(cat "$RUN/status.txt" 2>/dev/null || echo UNKNOWN)
if [ "$main_status" != DONE ]; then
  echo "STOPPED main_status=$main_status" > "$RUN/continuation_status.txt"
  exit 1
fi

echo RUNNING > "$RUN/continuation_status.txt"
bash "$GAIT/scripts/run_all_subject_baselines.sh"
echo DONE > "$RUN/continuation_status.txt"
