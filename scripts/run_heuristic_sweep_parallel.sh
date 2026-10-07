#!/bin/bash
# Run the hand-written S3 solvers in heuristics/<task>/solver_s3.py against variant presets.
# The "LLM" is a stub HTTP server (capx/serving/static_solver_server.py) that returns each
# task's solver file as if a model had written it; the rest of the pipeline (shared
# vision/planning services, per-variant yaml patching, run_batch) is scripts/sweep_lib.sh,
# shared with run_claude_code_sweep_parallel.sh.
#
# Usage: bash scripts/run_heuristic_sweep_parallel.sh [SHARED_SERVICES_GPU] [TOTAL_TRIALS] [STUB_PORT] [TASKS] [OUTPUT_PREFIX] [VARIANT_SET]
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT and the render workers (default: 0)
#   TOTAL_TRIALS         Trials per (task, variant) (default: 10). May also be a path to a
#                         TSV of "<task> <variant> <n>" lines: then exactly the listed cells
#                         run, with their own counts (VARIANT_SET is ignored, and the variant
#                         may be any preset in the task's VARIATIONS table, e.g. harder_spill).
#   STUB_PORT            Port for the static-solver stub (default: 8118)
#   TASKS                Comma-separated task short names, or "all" (default). Each needs a
#                         heuristics/<task>/solver_s3.py.
#   OUTPUT_PREFIX         outputs/<prefix>_<variant>/... (default: heuristic). Use a fresh
#                         prefix per run: run_batch numbers trials from trial_01 every time.
#   VARIANT_SET           singles (default), pairwise, combined, or hard_vs_base
#                         (baseline + each task's difficulty preset).
set -e
cd "$(git rev-parse --show-toplevel)"
source scripts/sweep_lib.sh

SHARED_SERVICES_GPU=${1:-0}
TOTAL_TRIALS=${2:-10}
STUB_PORT=${3:-8118}
TASKS_FILTER=${4:-all}
OUTPUT_PREFIX=${5:-heuristic}
VARIANT_SET=${6:-singles}

SWEEP_LOG_PREFIX="heur_sweep"
SWEEP_SERVER_URL="http://127.0.0.1:${STUB_PORT}/chat/completions"
# Solvers are written against each task's reduced low-level API.
SWEEP_APIS_OVERRIDE="S3"

sweep_resolve_variant_set "$VARIANT_SET"
sweep_resolve_tasks "$TASKS_FILTER"
for t in "${TASK_NAMES[@]}"; do
    if [ ! -f "heuristics/${t}/solver_s3.py" ]; then
        echo "ERROR: heuristics/${t}/solver_s3.py not found -- can't run task '$t'"
        exit 1
    fi
done
sweep_print_plan
echo "=== Backend: static-solver stub on port ${STUB_PORT} ==="

sweep_start_shared_services

STUB_LOG="/tmp/${SWEEP_LOG_PREFIX}_stub.log"
setsid .venv/bin/python -m capx.serving.static_solver_server \
    --solver-root heuristics --host 127.0.0.1 --port "${STUB_PORT}" > "$STUB_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
wait_for_port "static-solver stub" "${STUB_PORT}" "$!" "$STUB_LOG" "/health"
echo "=== static-solver stub ready (port ${STUB_PORT}). ==="

sweep_launch_workers per-task
