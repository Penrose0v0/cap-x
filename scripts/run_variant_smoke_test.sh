#!/bin/bash
# Start a local vLLM server for the given model, then run capx's robustness-eval
# variant-task batch against it -- all from one terminal, no need to juggle two.
#
# Usage: bash scripts/run_variant_smoke_test.sh [MODEL] [VLLM_GPU] [TOTAL_TRIALS] [GPU_MEM_UTIL]
#   MODEL         HF model id to serve via vLLM (default: Qwen/Qwen3.6-35B-A3B)
#   VLLM_GPU      CUDA device index to run the vLLM server on (default: 1)
#   TOTAL_TRIALS  Trials per config, overrides each yaml's own `trials:` (default: 3)
#   GPU_MEM_UTIL  Fraction of that GPU's TOTAL memory vLLM is allowed to reserve
#                 (default: 0.9). This is a shared machine -- check `nvidia-smi`
#                 first and pick a GPU + fraction that actually fits in its current
#                 free memory, or vLLM will fail immediately at startup.
#
# Set KEEP_SERVER_ALIVE=1 to leave the vLLM server running after the batch finishes
# (useful if you want to run more batches against it afterwards) instead of the
# default behavior of shutting it down when this script exits.
set -e

cd "$(git rev-parse --show-toplevel)"

MODEL=${1:-Qwen/Qwen3.6-35B-A3B}
VLLM_GPU=${2:-7}
TOTAL_TRIALS=${3:-3}
GPU_MEM_UTIL=${4:-0.9}
VLLM_PORT=8110
SERVER_URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"
VLLM_LOG="/tmp/vllm_server_$$.log"

CONFIG_PATHS=(
    env_configs/cube_stack/franka_robosuite_cube_stack_variants.yaml
    env_configs/cube_restack/franka_robosuite_cube_restack_variants.yaml
    env_configs/cube_lifting/franka_robosuite_cube_lifting_variants.yaml
    env_configs/nut_assembly/franka_robosuite_nut_assembly_variants.yaml
    env_configs/spill_wipe/franka_robosuite_spill_wipe_variants.yaml
    env_configs/two_arm_handover/two_arm_handover_variants.yaml
    env_configs/two_arm_lift/franka_robosuite_two_arm_lift_variants.yaml
)

# Read each config's `low_level.variation` and use it to label the output directory
# (outputs/<variation>/<model>/<config_stem>/trial_XX...) instead of dumping every run
# into the same untagged path, so separate variant experiments don't collide/mix.
echo "=== Reading 'variation' from each config ==="
VARIATIONS=()
for cfg in "${CONFIG_PATHS[@]}"; do
    v=$(uv run python -c "
from capx.envs.configs.loader import DictLoader
cfg = DictLoader.load('${cfg}')
print(cfg['env']['cfg']['low_level'].get('variation', 'UNKNOWN'))
" 2>/dev/null)
    echo "    ${cfg} -> ${v}"
    VARIATIONS+=("${v}")
done

VARIATION="${VARIATIONS[0]}"
for v in "${VARIATIONS[@]}"; do
    if [ "${v}" != "${VARIATION}" ]; then
        echo "WARNING: configs don't all use the same variation (saw '${VARIATION}' and '${v}')."
        echo "         Using 'mixed' as the output-dir label instead."
        VARIATION="mixed"
        break
    fi
done
echo "=== Output will be grouped under outputs/${VARIATION}/ ==="

echo "=== Starting vLLM server for ${MODEL} on GPU ${VLLM_GPU} (port ${VLLM_PORT}, gpu-memory-utilization=${GPU_MEM_UTIL}) ==="
echo "    Log: ${VLLM_LOG}"
# PATH must include .venv-vllm/bin (not just its python binary) -- flashinfer JIT-compiles
# some kernels on first use by shelling out to `ninja`, which is installed as a venv-local
# script; without this it fails with FileNotFoundError: 'ninja' even though the package is
# installed, since plain `.venv-vllm/bin/python ...` never puts that dir on PATH the way
# `source .venv-vllm/bin/activate` would.
PATH="$(pwd)/.venv-vllm/bin:${PATH}" CUDA_VISIBLE_DEVICES=${VLLM_GPU} setsid .venv-vllm/bin/python capx/serving/vllm_server.py \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --enforce-eager \
    --moe-backend triton \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    if [ "${KEEP_SERVER_ALIVE:-0}" = "1" ]; then
        echo ""
        echo "=== KEEP_SERVER_ALIVE=1: leaving vLLM server running (pid ${VLLM_PID}) ==="
        return
    fi
    echo ""
    echo "=== Shutting down vLLM server (pid ${VLLM_PID}) ==="
    # setsid above makes VLLM_PID its own process group ID -- vLLM spawns its own
    # EngineCore/worker subprocesses, which a plain `kill $VLLM_PID` leaves orphaned
    # (reparented to PID 1, still holding GPU memory). Signaling -VLLM_PID hits the whole group.
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for vLLM server to become healthy (up to 15 minutes for model load) ==="
ready=false
for i in $(seq 1 180); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM server process died during startup -- last 60 lines of ${VLLM_LOG}:"
        tail -n 60 "${VLLM_LOG}"
        exit 1
    fi
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 5
done

if [ "${ready}" != "true" ]; then
    echo "Timed out waiting for vLLM server -- last 60 lines of ${VLLM_LOG}:"
    tail -n 60 "${VLLM_LOG}"
    exit 1
fi
echo "vLLM server is ready."

echo "=== Running batch smoke test (model=${MODEL}, total_trials=${TOTAL_TRIALS}, variation=${VARIATION}) ==="
uv run python -m capx.envs.scripts.run_batch \
    --args.config-paths "${CONFIG_PATHS[@]}" \
    --args.models "${MODEL}" \
    --args.server-url "${SERVER_URL}" \
    --args.total-trials "${TOTAL_TRIALS}" \
    --args.output-dir "outputs/${VARIATION}"
