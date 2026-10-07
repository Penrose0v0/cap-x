#!/bin/bash
# Run variant presets with the claude-code CLI proxy (capx/serving/claude_code_server.py) as
# the LLM backend. One worker per task, all against a single shared proxy; the shared
# vision/planning services, per-variant yaml patching and run_batch loop are
# scripts/sweep_lib.sh, shared with run_heuristic_sweep_parallel.sh.
#
# Usage: bash scripts/run_claude_code_sweep_parallel.sh [MODEL_ALIAS] [SHARED_SERVICES_GPU] [TOTAL_TRIALS] [CLAUDE_PORT] [TASKS] [OUTPUT_PREFIX] [VARIANT_SET] [APIS_OVERRIDE]
#   MODEL_ALIAS          claude -p --model alias, e.g. opus / sonnet / haiku (default: sonnet)
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT and the render workers (default: 0)
#   TOTAL_TRIALS         Trials per (task, variant) (default: 3). May also be a path to a TSV
#                         of "<task> <variant> <n>" lines: then exactly the listed cells run,
#                         with their own counts (VARIANT_SET is ignored).
#   CLAUDE_PORT          Port for the shared claude-code-cli proxy (default: 8112)
#   TASKS                Comma-separated task short names, or "all" (default).
#   OUTPUT_PREFIX         outputs/<prefix>_<variant>/... (default: claude_code). Use a fresh
#                         prefix per run: run_batch numbers trials from trial_01 every time.
#   VARIANT_SET           singles (default), pairwise, combined, or hard_vs_base
#                         (baseline + each task's difficulty preset).
#   APIS_OVERRIDE         "" (the yaml's own high-level API, default), "S3" for each task's
#                         reduced low-level API (the tier the hard-coded solvers use), or an
#                         explicit API class name.
set -e
cd "$(git rev-parse --show-toplevel)"
source scripts/sweep_lib.sh

MODEL_ALIAS=${1:-sonnet}
SHARED_SERVICES_GPU=${2:-0}
TOTAL_TRIALS=${3:-3}
CLAUDE_PORT=${4:-8112}
TASKS_FILTER=${5:-all}
OUTPUT_PREFIX=${6:-claude_code}
VARIANT_SET=${7:-singles}
SWEEP_APIS_OVERRIDE=${8:-}

MODEL="claude-code/${MODEL_ALIAS}"
SWEEP_LOG_PREFIX="claude_sweep"
SWEEP_SERVER_URL="http://127.0.0.1:${CLAUDE_PORT}/chat/completions"

sweep_resolve_variant_set "$VARIANT_SET"
sweep_resolve_tasks "$TASKS_FILTER"
sweep_print_plan
echo "=== Model: ${MODEL} ==="

sweep_start_shared_services

CLAUDE_LOG="/tmp/${SWEEP_LOG_PREFIX}_proxy.log"
setsid uv run --frozen python -m capx.serving.claude_code_server --port "${CLAUDE_PORT}" > "$CLAUDE_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
wait_for_port "claude-code proxy" "${CLAUDE_PORT}" "$!" "$CLAUDE_LOG" "/health"
echo "=== claude-code proxy ready (model=${MODEL}). ==="

sweep_launch_workers "$MODEL"
