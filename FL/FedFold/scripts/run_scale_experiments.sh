#!/usr/bin/env bash
set -euo pipefail

# Quick experiment driver for FedFold train.py
# - Creates a small set of runs (quick) and a few replicate-scale tests
# - Writes logs to FedFold/run_logs/
# Usage:
#   cd FedFold
#   ./scripts/run_scale_experiments.sh
# Environment: activate your Python environment with torch installed before running (e.g. `conda activate fl_env`)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python3}
TRAIN_SCRIPT="$ROOT/train.py"
LOGDIR="$ROOT/run_logs"
mkdir -p "$LOGDIR"

# Optional: list of GPUs to use (by PCI id shown in `nvidia-smi`, e.g. 0,1). If empty, the system default is used.
# Example: export GPUS=(0 1) or edit below. When multiple GPUs are provided the script will round-robin assign them.
GPUS=()

# RUN_MODE: "sequential" (default) or "parallel". In parallel mode jobs are started with nohup and their PIDs
# are written to $LOGDIR/manifest.csv so you can track them.
RUN_MODE="sequential"
MANIFEST_PATH="$LOGDIR/manifest.csv"
if [ -f "$MANIFEST_PATH" ]; then
  rm -f "$MANIFEST_PATH"
fi
echo "case,log,pid,gpu,start_time" > "$MANIFEST_PATH"

# Define experiment cases as: name;n_device;sim_mode;replicate_splits;selection_mode;selected_device;participation_rate;global_epochs;local_epochs;n_split;quantize;quant_bits
# Keep epochs small for quick turnaround; for longer runs adjust the last fields.
cases=(
  # quick_small: a slightly longer smoke test you can check tomorrow morning
  # "quick_small;100;none;100;fixed;10;0.1;50;5;2;true;-1"
  # replicate with proportional selection; enable quantization (random bits)
  "replicate_1k_prop;1000;replicate;100;proportional;0;0.01;200;5;2;true;-1"
  # larger replicate run (shorter epochs to keep time reasonable); fixed 8-bit quant
  "replicate_10k_prop;10000;replicate;100;proportional;0;0.01;100;5;2;true;-1"
  # replicate with fixed selection, no quantize
  "replicate_1k_fixed;1000;replicate;100;fixed;10;0.0;200;5;2;true;-1"
)

for c in "${cases[@]}"; do
  IFS=';' read -r name n_device sim_mode rep_splits selection_mode selected_device participation_rate global_epochs local_epochs n_split quantize_flag quant_bits <<< "$c"
  logf="$LOGDIR/${name}.log"
  echo "==========" | tee -a "$logf"
  echo "RUN: $name" | tee -a "$logf"
  echo "Params: n_device=$n_device sim_mode=$sim_mode selection_mode=$selection_mode selected_device=$selected_device participation_rate=$participation_rate global_epochs=$global_epochs local_epochs=$local_epochs n_split=$n_split" | tee -a "$logf"
  echo "Start: $(date)" | tee -a "$logf"

  cmd=("$PYTHON" "$TRAIN_SCRIPT"
       --n_device "$n_device"
       --sim_mode "$sim_mode"
       --replicate_splits "$rep_splits"
       --selection_mode "$selection_mode"
       --selected_device "$selected_device"
       --participation_rate "$participation_rate"
       --global_epochs "$global_epochs"
       --local_epochs "$local_epochs"
       --n_split "$n_split"
       --dataset CIFAR10)

  # optionally enable quantization for this run
  if [ "${quantize_flag,,}" = "true" ] || [ "${quantize_flag,,}" = "1" ]; then
    cmd+=(--quantize --quant_bits "$quant_bits")
  fi

  # Determine GPU for this case (round-robin if GPUS provided)
  gpu_assign=""
  if [ ${#GPUS[@]} -gt 0 ]; then
    idx=$((case_index % ${#GPUS[@]}))
    gpu_assign=${GPUS[$idx]}
  fi

  # Print the command we'll run
  echo "Command: ${cmd[*]}" | tee -a "$logf"

  if [ "$RUN_MODE" = "parallel" ]; then
    # Run with nohup in background and capture PID; set CUDA_VISIBLE_DEVICES if gpu_assign set
    start_time="$(date --rfc-3339=seconds)"
    if [ -n "$gpu_assign" ]; then
      CUDA_VISIBLE_DEVICES="$gpu_assign" nohup "${cmd[@]}" > "$logf" 2>&1 &
    else
      nohup "${cmd[@]}" > "$logf" 2>&1 &
    fi
    pid=$!
    echo "$name,$logf,$pid,$gpu_assign,$start_time" >> "$MANIFEST_PATH"
    echo "Launched in background (pid=$pid), see $MANIFEST_PATH" | tee -a "$logf"
  else
    # Sequential run (default). Use CUDA_VISIBLE_DEVICES for GPU pinning if requested.
    if [ -n "$gpu_assign" ]; then
      echo "Running sequentially pinned to GPU $gpu_assign" | tee -a "$logf"
      CUDA_VISIBLE_DEVICES="$gpu_assign" "${cmd[@]}" 2>&1 | tee -a "$logf"
    else
      "${cmd[@]}" 2>&1 | tee -a "$logf"
    fi
  fi

  echo "End: $(date)" | tee -a "$logf"
  echo "==========" | tee -a "$logf"
  echo "" # blank line between runs
done

echo "All experiments finished. Logs are in: $LOGDIR"

# End of script
