#!/usr/bin/env bash
# Launch one stage script under the repository's runner instrumentation:
# program stdout/stderr go to the log verbatim, and a single trailing
# instrumentation block records runtime, device, GPU use, and exit code.
#
# Stage 50 runs the depth comparison on the host's RTX 3090 Ti, which is off the
# lineage device, so the canonical device guard is opted out of deliberately:
# --allow-foreign-device makes the stage claim an internally matched pair instead
# of byte-identical control reproduction, and score the stored C256 L=6 champion
# as the panel's device-shift row. To run the canonical variant instead, replace
# the device arguments with `--device cuda`.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE_LOG="$REPO_ROOT/checkpoints/logs/50_50_depth_revisit.log"
STAGE_DEVICE_ARGS=(--device cuda:0 --allow-foreign-device)
cd "$REPO_ROOT"

sampler_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
    ( max_u=0; max_m=0
      while :; do
        # nvidia-smi emits comma-separated fields, so read with IFS=, and strip
        # the leading space its "nounits" format leaves on the memory field.
        line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | sort -t, -k1 -n | tail -1)
        if [ -n "$line" ]; then
            IFS=, read -r u m <<< "$line"
            u=${u// /}; m=${m// /}
            if [ -n "$u" ] && [ "$u" -gt "$max_u" ] 2>/dev/null; then max_u=$u; fi
            if [ -n "$m" ] && [ "$m" -gt "$max_m" ] 2>/dev/null; then max_m=$m; fi
            printf '%s %s\n' "$max_u" "$max_m" > /tmp/stage50_gpu_peaks.txt
        fi
        sleep 2
      done ) &
    sampler_pid=$!
fi

started=$(date +%s)
.venv/bin/python -u 50_depth_revisit.py "${STAGE_DEVICE_ARGS[@]}" > "$STAGE_LOG" 2>&1
exit_code=$?
elapsed=$(( $(date +%s) - started ))

[ -n "$sampler_pid" ] && kill "$sampler_pid" 2>/dev/null

{
  echo "__RUN_TIME_SECONDS__ $elapsed"
  echo "__DEVICE__ cuda:0 (NVIDIA GeForce RTX 3090 Ti, foreign)"
  if [ -s /tmp/stage50_gpu_peaks.txt ]; then
    read -r max_u max_m < /tmp/stage50_gpu_peaks.txt
    echo "__GPU_MAX_UTIL__ ${max_u}%"
    echo "__GPU_MAX_MEM__ ${max_m} MiB"
  fi
  echo "__EXIT_CODE__ $exit_code"
  [ "$exit_code" -eq 0 ] && echo "__RUN_OK__"
} >> "$STAGE_LOG"

echo "stage 50 finished with exit code $exit_code after ${elapsed}s"
exit "$exit_code"
