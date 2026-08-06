#!/bin/bash
# Run all 11 variation presets across all 7 robosuite tasks using the claude-code-cli proxy
# (capx/serving/claude_code_server.py) as the LLM backend -- the Claude-backed counterpart to
# run_variant_sweep_parallel.sh (which drives per-GPU vLLM workers instead).
#
# Unlike the vLLM sweep, there's no GPU needed for the LLM itself -- `claude -p` is a network
# call to Anthropic. Parallelism here is one worker PER TASK (not per GPU): each worker loops
# through all 11 variants x TOTAL_TRIALS for its one assigned task, sequentially, all 7 workers
# running at once against a single shared claude-code-cli proxy instance.
#
# The vision/planning microservices (SAM3/SAM2/GraspNet/PyRoki/OWL-ViT) still need a GPU and
# still have the same hardcoded-port limitation as the vLLM sweep (see that script's header
# comment for the full explanation) -- so this reuses the same fix: ONE shared set of those
# servers, with `api_servers` stripped from every per-variant yaml copy.
#
# Usage: bash scripts/run_claude_code_sweep_parallel.sh [MODEL_ALIAS] [SHARED_SERVICES_GPU] [TOTAL_TRIALS] [CLAUDE_PORT]
#   MODEL_ALIAS          claude -p --model alias, e.g. opus / sonnet / haiku (default: sonnet)
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT (default: 0)
#   TOTAL_TRIALS         Trials per (task, variant) (default: 3)
#   CLAUDE_PORT          Port for the shared claude-code-cli proxy (default: 8112)
set -e

cd "$(git rev-parse --show-toplevel)"

MODEL_ALIAS=${1:-sonnet}
SHARED_SERVICES_GPU=${2:-0}
TOTAL_TRIALS=${3:-3}
CLAUDE_PORT=${4:-8112}
MODEL="claude-code/${MODEL_ALIAS}"
CLAUDE_SERVER_URL="http://127.0.0.1:${CLAUDE_PORT}/chat/completions"

ALL_VARIANTS=(
    baseline
    color_shift_mild
    color_shift_strong
    object_position_shift
    robot_position_shift
    lighting_bright
    lighting_dim
    distractor_object
    target_object_swap
    language_paraphrase
    compound_hard
)

TASK_CONFIGS=(
    env_configs/cube_stack/franka_robosuite_cube_stack_variants.yaml
    env_configs/cube_restack/franka_robosuite_cube_restack_variants.yaml
    env_configs/cube_lifting/franka_robosuite_cube_lifting_variants.yaml
    env_configs/nut_assembly/franka_robosuite_nut_assembly_variants.yaml
    env_configs/spill_wipe/franka_robosuite_spill_wipe_variants.yaml
    env_configs/two_arm_handover/two_arm_handover_variants.yaml
    env_configs/two_arm_lift/franka_robosuite_two_arm_lift_variants.yaml
)

# --- Python helper: sets low_level.variation and empties api_servers (see note above). ---
PATCH_SCRIPT="/tmp/patch_capx_variant_yaml.py"
cat > "$PATCH_SCRIPT" << 'PYEOF'
import sys
import yaml

src, dst, variation = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    data = yaml.safe_load(f)

data["env"]["cfg"]["low_level"]["variation"] = variation
data["api_servers"] = []

with open(dst, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PYEOF

echo "=== ${#TASK_CONFIGS[@]} tasks x ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials = $(( ${#TASK_CONFIGS[@]} * ${#ALL_VARIANTS[@]} * TOTAL_TRIALS )) total trials ==="
echo "=== Model: ${MODEL}; shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

ALL_BG_PIDS=()
cleanup() {
    echo ""
    echo "=== Shutting down all background servers: ${ALL_BG_PIDS[*]} ==="
    # Each of these was started via `setsid`, making its PID also its process group ID --
    # signaling -$pid hits the whole group, including any subprocess/multiprocessing children
    # it spawned internally. A plain `kill $pid` only hits the top process and leaves
    # descendants orphaned (reparented to PID 1, still holding GPU memory and ports).
    for pid in "${ALL_BG_PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    sleep 2
    for pid in "${ALL_BG_PIDS[@]}"; do
        kill -KILL -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${ALL_BG_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

wait_for_port() {
    local name="$1" port="$2" pid="$3" logfile="$4" path="${5:-/openapi.json}"
    for i in $(seq 1 180); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "${name} died during startup -- last 40 lines of ${logfile}:"
            tail -n 40 "$logfile"
            return 1
        fi
        if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}${path}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "${name} timed out waiting to become ready -- last 40 lines of ${logfile}:"
    tail -n 40 "$logfile"
    return 1
}

# --- Start the ONE shared set of vision/planning services. ---
echo "=== Starting shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

SAM3_LOG="/tmp/claude_sweep_shared_sam3.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_sam3_server \
    --device cuda --port 8114 --host 127.0.0.1 > "$SAM3_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
SAM3_PID=$!

GRASPNET_LOG="/tmp/claude_sweep_shared_graspnet.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_contact_graspnet_server \
    --port 8115 --host 127.0.0.1 > "$GRASPNET_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
GRASPNET_PID=$!

PYROKI_LOG="/tmp/claude_sweep_shared_pyroki.log"
# launch_pyroki_server.py's __main__ calls main() directly (no tyro.cli), so CLI flags here
# would be silently ignored -- its defaults (panda_description/panda_hand/port 8116) already
# match what we want, so no flags are passed.
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_pyroki_server \
    > "$PYROKI_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
PYROKI_PID=$!

SAM2_LOG="/tmp/claude_sweep_shared_sam2.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_sam2_server \
    --port 8113 --host 127.0.0.1 --model-name facebook/sam2.1-hiera-large > "$SAM2_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
SAM2_PID=$!

OWLVIT_LOG="/tmp/claude_sweep_shared_owlvit.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_owlvit_server \
    --port 8117 --host 127.0.0.1 --model-name google/owlv2-large-patch14-ensemble > "$OWLVIT_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
OWLVIT_PID=$!

wait_for_port "SAM3"     8114 "$SAM3_PID"     "$SAM3_LOG"
wait_for_port "GraspNet" 8115 "$GRASPNET_PID" "$GRASPNET_LOG"
wait_for_port "PyRoki"   8116 "$PYROKI_PID"   "$PYROKI_LOG"
wait_for_port "SAM2"     8113 "$SAM2_PID"     "$SAM2_LOG"
wait_for_port "OWL-ViT"  8117 "$OWLVIT_PID"   "$OWLVIT_LOG"
echo "=== Shared vision/planning services ready. ==="

# --- Start the ONE shared claude-code-cli proxy. ---
CLAUDE_LOG="/tmp/claude_sweep_proxy.log"
setsid uv run --frozen python -m capx.serving.claude_code_server --port "${CLAUDE_PORT}" > "$CLAUDE_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
CLAUDE_PID=$!
wait_for_port "claude-code proxy" "${CLAUDE_PORT}" "$CLAUDE_PID" "$CLAUDE_LOG" "/health"
echo "=== claude-code proxy ready (model=${MODEL}). ==="

run_task_worker() {
    local task_cfg="$1"
    local task_name
    task_name=$(basename "$task_cfg" .yaml)
    local worker_log="/tmp/claude_sweep_${task_name}.log"
    : > "$worker_log"

    echo "[${task_name}] Starting -- ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials" | tee -a "$worker_log"

    for variant in "${ALL_VARIANTS[@]}"; do
        echo "[${task_name}] === variant: ${variant} ===" | tee -a "$worker_log"
        local tmp_cfg="/tmp/claude_sweep_configs_${task_name}_${variant}.yaml"
        .venv/bin/python "$PATCH_SCRIPT" "$task_cfg" "$tmp_cfg" "$variant"

        uv run --frozen python -m capx.envs.scripts.run_batch \
            --args.config-paths "$tmp_cfg" \
            --args.models "${MODEL}" \
            --args.server-url "${CLAUDE_SERVER_URL}" \
            --args.total-trials "${TOTAL_TRIALS}" \
            --args.output-dir "outputs/claude_code_${variant}" \
            >> "$worker_log" 2>&1

        echo "[${task_name}] === finished variant: ${variant} ===" | tee -a "$worker_log"
    done

    echo "[${task_name}] All variants done." | tee -a "$worker_log"
}

WORKER_PIDS=()
for task_cfg in "${TASK_CONFIGS[@]}"; do
    run_task_worker "$task_cfg" &
    WORKER_PIDS+=("$!")
done

echo "=== All ${#TASK_CONFIGS[@]} task workers launched. Tail their logs with: ==="
for task_cfg in "${TASK_CONFIGS[@]}"; do
    task_name=$(basename "$task_cfg" .yaml)
    echo "    tail -f /tmp/claude_sweep_${task_name}.log"
done

# Wait on just the 7 task workers, NOT a bare `wait` -- the shared vision/planning services
# and the claude-code proxy are also background jobs of this shell, but they're long-running
# servers that never exit on their own, so a bare `wait` would hang forever even after every
# task worker finishes.
wait "${WORKER_PIDS[@]}"
echo "=== All task workers finished. ==="
