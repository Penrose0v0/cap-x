#!/bin/bash
# Run the remaining (untested) variation presets across all 7 robosuite tasks, using as many
# GPUs as you specify in parallel for the LLM -- one vLLM server + worker per GPU. The variant
# queue is split evenly (round-robin) across however many GPUs you pass.
#
# IMPORTANT ARCHITECTURE NOTE: capx's vision/planning service clients
# (capx/integrations/vision/sam3.py, sam2.py, owlvit.py) hardcode their server URL as a
# bare module-level constant (127.0.0.1:8114/8113/8117) with NO env-var or config override
# -- unlike GraspNet (honors $GRASPNET_SERVICE_URL) and PyRoki (accepts a server_url kwarg
# that capx's own task setup code does thread through from config). So per-worker port
# offsets for these three CANNOT work: the trial code inside every worker always talks to
# the same hardcoded ports regardless of what port their own yaml's api_servers section
# requests. The correct fix is the opposite of per-worker isolation: start ONE shared set of
# SAM3/SAM2/GraspNet/PyRoki/OWL-ViT servers ONCE (on a dedicated GPU, on their default
# ports), and strip `api_servers` out of every worker's yaml entirely so capx's runner
# doesn't also try to start/stop redundant copies of them on every task transition (which is
# what caused the original port-collision hangs).
#
# Usage: bash scripts/run_variant_sweep_parallel.sh [GPUS] [SHARED_SERVICES_GPU] [MODEL] [TOTAL_TRIALS] [LLM_GPU_MEM_UTIL]
#   GPUS               Comma-separated CUDA device indices, one LLM worker per GPU
#                       (default: 1,3,4,5)
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT (must not be in
#                       GPUS -- these run alongside a 35B-class LLM and don't share well).
#                       (default: 0)
#   MODEL             HF model id to serve via vLLM (default: Qwen/Qwen3.6-35B-A3B)
#   TOTAL_TRIALS      Trials per (task, variant) (default: 3)
#   LLM_GPU_MEM_UTIL  Fraction of each LLM GPU's TOTAL memory vLLM may reserve (default: 0.85)
#
# Example: bash scripts/run_variant_sweep_parallel.sh 0,1,2,4,5,6 3 Qwen/Qwen3.6-35B-A3B 3 0.85
#
# DONE_VARIANTS is hardcoded below -- edit if your completed-variant list changes.
set -e

cd "$(git rev-parse --show-toplevel)"

IFS=',' read -ra GPUS <<< "${1:-1,3,4,5}"
SHARED_SERVICES_GPU=${2:-0}
MODEL=${3:-Qwen/Qwen3.6-35B-A3B}
TOTAL_TRIALS=${4:-3}
GPU_MEM_UTIL=${5:-0.85}

BASE_LLM_PORT=8110

for g in "${GPUS[@]}"; do
    if [ "$g" = "$SHARED_SERVICES_GPU" ]; then
        echo "ERROR: SHARED_SERVICES_GPU (${SHARED_SERVICES_GPU}) is also in GPUS (${GPUS[*]})." \
             "A 35B-class LLM and the 5 shared vision/planning models don't fit together on one" \
             "GPU -- pick a SHARED_SERVICES_GPU outside your GPUS list."
        exit 1
    fi
done

# Presets already tested previously -- skipped here. Edit this if that changes.
DONE_VARIANTS=(color_shift_mild baseline robot_position_shift color_shift_strong object_position_shift lighting_bright lighting_dim distractor_object target_object_swap language_paraphrase compound_hard)

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

BASE_CONFIG_PATHS=(
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

# Compute REMAINING = ALL_VARIANTS - DONE_VARIANTS, preserving ALL_VARIANTS order.
REMAINING=()
for v in "${ALL_VARIANTS[@]}"; do
    skip=false
    for d in "${DONE_VARIANTS[@]}"; do
        [ "$v" = "$d" ] && skip=true && break
    done
    [ "$skip" = false ] && REMAINING+=("$v")
done

N_WORKERS=${#GPUS[@]}
echo "=== ${#REMAINING[@]} variants left to test: ${REMAINING[*]:-<none>} ==="
echo "=== Using ${N_WORKERS} GPUs for LLM workers: ${GPUS[*]}; shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

if [ "${#REMAINING[@]}" -eq 0 ]; then
    echo "Nothing left to run -- all variants are in DONE_VARIANTS."
    exit 0
fi

# Round-robin assignment: worker i gets REMAINING[i], REMAINING[i+N_WORKERS], ... Written to
# one file per worker index (rather than N hardcoded bash arrays) so this scales to any GPU
# count passed via $1, not just a fixed number of workers.
ASSIGN_DIR=$(mktemp -d)
for idx in "${!REMAINING[@]}"; do
    worker_idx=$(( idx % N_WORKERS ))
    echo "${REMAINING[$idx]}" >> "${ASSIGN_DIR}/worker_${worker_idx}.txt"
done

ALL_BG_PIDS=()
cleanup() {
    echo ""
    echo "=== Shutting down all background servers: ${ALL_BG_PIDS[*]} ==="
    # Each of these was started via `setsid`, making its PID also its process group ID --
    # signaling -$pid hits the whole group, including any subprocess/multiprocessing children
    # it spawned internally. A plain `kill $pid` only hits the top process and leaves
    # descendants orphaned (reparented to PID 1, still holding GPU memory and ports) -- this
    # is exactly what happened to a previous run of this script, discovered days later.
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
    rm -rf "$ASSIGN_DIR"
}
trap cleanup EXIT

wait_for_port() {
    local name="$1" port="$2" pid="$3" logfile="$4"
    for i in $(seq 1 180); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "${name} died during startup -- last 40 lines of ${logfile}:"
            tail -n 40 "$logfile"
            return 1
        fi
        if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/openapi.json" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "${name} timed out waiting to become ready -- last 40 lines of ${logfile}:"
    tail -n 40 "$logfile"
    return 1
}

# --- Start the ONE shared set of vision/planning services (see architecture note above). ---
echo "=== Starting shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

SAM3_LOG="/tmp/shared_sam3.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_sam3_server \
    --device cuda --port 8114 --host 127.0.0.1 > "$SAM3_LOG" 2>&1 &
SAM3_PID=$!
ALL_BG_PIDS+=("$SAM3_PID")

GRASPNET_LOG="/tmp/shared_graspnet.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_contact_graspnet_server \
    --port 8115 --host 127.0.0.1 > "$GRASPNET_LOG" 2>&1 &
GRASPNET_PID=$!
ALL_BG_PIDS+=("$GRASPNET_PID")

PYROKI_LOG="/tmp/shared_pyroki.log"
# launch_pyroki_server.py's __main__ calls main() directly (no tyro.cli), so CLI flags here
# would be silently ignored -- its defaults (panda_description/panda_hand/port 8116) already
# match what we want, so no flags are passed.
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_pyroki_server \
    > "$PYROKI_LOG" 2>&1 &
PYROKI_PID=$!
ALL_BG_PIDS+=("$PYROKI_PID")

SAM2_LOG="/tmp/shared_sam2.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_sam2_server \
    --port 8113 --host 127.0.0.1 --model-name facebook/sam2.1-hiera-large > "$SAM2_LOG" 2>&1 &
SAM2_PID=$!
ALL_BG_PIDS+=("$SAM2_PID")

OWLVIT_LOG="/tmp/shared_owlvit.log"
CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m capx.serving.launch_owlvit_server \
    --port 8117 --host 127.0.0.1 --model-name google/owlv2-large-patch14-ensemble > "$OWLVIT_LOG" 2>&1 &
OWLVIT_PID=$!
ALL_BG_PIDS+=("$OWLVIT_PID")

wait_for_port "SAM3"     8114 "$SAM3_PID"     "$SAM3_LOG"
wait_for_port "GraspNet" 8115 "$GRASPNET_PID" "$GRASPNET_LOG"
wait_for_port "PyRoki"   8116 "$PYROKI_PID"   "$PYROKI_LOG"
wait_for_port "SAM2"     8113 "$SAM2_PID"     "$SAM2_LOG"
wait_for_port "OWL-ViT"  8117 "$OWLVIT_PID"   "$OWLVIT_LOG"
echo "=== Shared services ready. ==="

make_variant_configs() {
    local variant="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    for cfg in "${BASE_CONFIG_PATHS[@]}"; do
        .venv/bin/python "$PATCH_SCRIPT" "$cfg" "${out_dir}/$(basename "$cfg")" "$variant"
    done
}

run_worker() {
    local gpu="$1"
    local port="$2"
    shift 2
    local variants=("$@")
    local worker_log="/tmp/variant_sweep_gpu${gpu}.log"
    : > "$worker_log"

    if [ "${#variants[@]}" -eq 0 ]; then
        echo "[GPU ${gpu}] No variants assigned, nothing to do." >> "$worker_log"
        return
    fi

    echo "[GPU ${gpu}] Assigned variants: ${variants[*]} (llm_port=${port})" | tee -a "$worker_log"

    local vllm_log="/tmp/vllm_server_gpu${gpu}.log"
    PATH="$(pwd)/.venv-vllm/bin:${PATH}" CUDA_VISIBLE_DEVICES=${gpu} setsid .venv-vllm/bin/python capx/serving/vllm_server.py \
        --model "${MODEL}" \
        --port "${port}" \
        --enforce-eager \
        --moe-backend triton \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        > "${vllm_log}" 2>&1 &
    local vllm_pid=$!
    ALL_BG_PIDS+=("$vllm_pid")

    echo "[GPU ${gpu}] vLLM server starting (pid ${vllm_pid}, log ${vllm_log})..." | tee -a "$worker_log"
    local ready=false
    for i in $(seq 1 180); do
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "[GPU ${gpu}] vLLM server died during startup -- see ${vllm_log}" | tee -a "$worker_log"
            return 1
        fi
        if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 5
    done
    if [ "$ready" != "true" ]; then
        echo "[GPU ${gpu}] Timed out waiting for vLLM -- see ${vllm_log}" | tee -a "$worker_log"
        return 1
    fi
    echo "[GPU ${gpu}] vLLM ready." | tee -a "$worker_log"

    for variant in "${variants[@]}"; do
        echo "[GPU ${gpu}] === Starting variant: ${variant} ===" | tee -a "$worker_log"
        local cfg_dir="/tmp/variant_configs_${variant}"
        make_variant_configs "${variant}" "${cfg_dir}"
        local variant_paths=()
        for cfg in "${BASE_CONFIG_PATHS[@]}"; do
            variant_paths+=("${cfg_dir}/$(basename "$cfg")")
        done

        .venv/bin/python -m capx.envs.scripts.run_batch \
            --args.config-paths "${variant_paths[@]}" \
            --args.models "${MODEL}" \
            --args.server-url "http://127.0.0.1:${port}/v1/chat/completions" \
            --args.total-trials "${TOTAL_TRIALS}" \
            --args.output-dir "outputs/${variant}" \
            >> "$worker_log" 2>&1

        echo "[GPU ${gpu}] === Finished variant: ${variant} ===" | tee -a "$worker_log"
    done

    echo "[GPU ${gpu}] All assigned variants done. Shutting down vLLM (pid ${vllm_pid})." | tee -a "$worker_log"
    kill -TERM -- "-${vllm_pid}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${vllm_pid}" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
}

for i in "${!GPUS[@]}"; do
    variants_file="${ASSIGN_DIR}/worker_${i}.txt"
    assigned=()
    if [ -f "$variants_file" ]; then
        mapfile -t assigned < "$variants_file"
    fi
    run_worker "${GPUS[$i]}" "$(( BASE_LLM_PORT + i * 100 ))" "${assigned[@]}" &
done

echo "=== All ${N_WORKERS} LLM workers launched. Tail their logs with: ==="
for gpu in "${GPUS[@]}"; do
    echo "    tail -f /tmp/variant_sweep_gpu${gpu}.log"
done

wait
echo "=== All workers finished. ==="
