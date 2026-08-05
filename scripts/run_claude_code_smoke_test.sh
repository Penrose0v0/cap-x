#!/bin/bash
# Start the claude-code-cli proxy server (wraps `claude -p`, auth via
# CLAUDE_CODE_OAUTH_TOKEN), then run capx's robustness-eval variant-task batch
# against it -- the Claude-backed counterpart to run_variant_smoke_test.sh
# (which drives a local vLLM server instead).
#
# Usage: bash scripts/run_claude_code_smoke_test.sh [MODEL_ALIAS] [TOTAL_TRIALS] [PORT]
#   MODEL_ALIAS   claude -p --model alias, e.g. opus / sonnet / haiku (default: sonnet)
#   TOTAL_TRIALS  Trials per config, overrides each yaml's own `trials:` (default: 1)
#   PORT          Port for the local proxy server (default: 8112)
#
# Requires .claudeoauthtoken in the repo root (see capx/serving/claude_code_server.py
# for how it's used). Unlike the vLLM smoke test, there's no model-loading wait --
# the proxy is a thin wrapper, so it's healthy almost immediately.
#
# Set KEEP_SERVER_ALIVE=1 to leave the proxy server running after the batch finishes
# instead of the default behavior of shutting it down when this script exits.
set -e

cd "$(git rev-parse --show-toplevel)"

MODEL_ALIAS=${1:-sonnet}
TOTAL_TRIALS=${2:-1}
PORT=${3:-8112}
MODEL="claude-code/${MODEL_ALIAS}"
SERVER_URL="http://127.0.0.1:${PORT}/chat/completions"
SERVER_LOG="/tmp/claude_code_server_$$.log"

CONFIG_PATHS=(
    env_configs/cube_stack/franka_robosuite_cube_stack_variants.yaml
    env_configs/cube_restack/franka_robosuite_cube_restack_variants.yaml
    env_configs/cube_lifting/franka_robosuite_cube_lifting_variants.yaml
    env_configs/nut_assembly/franka_robosuite_nut_assembly_variants.yaml
    env_configs/spill_wipe/franka_robosuite_spill_wipe_variants.yaml
    env_configs/two_arm_handover/two_arm_handover_variants.yaml
    env_configs/two_arm_lift/franka_robosuite_two_arm_lift_variants.yaml
)

echo "=== Reading 'variation' from each config ==="
VARIATIONS=()
for cfg in "${CONFIG_PATHS[@]}"; do
    # Some capx imports (R1Pro/LIBERO availability checks) print noise straight to
    # stdout rather than stderr, so `2>/dev/null` alone doesn't clean this up --
    # only the actual variation value is guaranteed to be the last line.
    v=$(uv run --frozen python -c "
from capx.envs.configs.loader import DictLoader
cfg = DictLoader.load('${cfg}')
print(cfg['env']['cfg']['low_level'].get('variation', 'UNKNOWN'))
" 2>/dev/null | tail -n 1)
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
echo "=== Output will be grouped under outputs/claude_code_${VARIATION}/ ==="

echo "=== Starting claude-code-cli proxy on port ${PORT} (model=${MODEL}) ==="
echo "    Log: ${SERVER_LOG}"
uv run --frozen python -m capx.serving.claude_code_server \
    --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if [ "${KEEP_SERVER_ALIVE:-0}" = "1" ]; then
        echo ""
        echo "=== KEEP_SERVER_ALIVE=1: leaving proxy server running (pid ${SERVER_PID}) ==="
        return
    fi
    echo ""
    echo "=== Shutting down proxy server (pid ${SERVER_PID}) ==="
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for proxy server to become healthy ==="
ready=false
for i in $(seq 1 30); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Proxy server process died during startup -- log:"
        cat "${SERVER_LOG}"
        exit 1
    fi
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [ "${ready}" != "true" ]; then
    echo "Timed out waiting for proxy server -- log:"
    cat "${SERVER_LOG}"
    exit 1
fi
echo "Proxy server is ready."

echo "=== Running batch smoke test (model=${MODEL}, total_trials=${TOTAL_TRIALS}, variation=${VARIATION}) ==="
uv run --frozen python -m capx.envs.scripts.run_batch \
    --args.config-paths "${CONFIG_PATHS[@]}" \
    --args.models "${MODEL}" \
    --args.server-url "${SERVER_URL}" \
    --args.total-trials "${TOTAL_TRIALS}" \
    --args.output-dir "outputs/claude_code_${VARIATION}"
