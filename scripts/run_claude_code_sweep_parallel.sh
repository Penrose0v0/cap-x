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
# Usage: bash scripts/run_claude_code_sweep_parallel.sh [MODEL_ALIAS] [SHARED_SERVICES_GPU] [TOTAL_TRIALS] [CLAUDE_PORT] [TASKS] [OUTPUT_PREFIX] [VARIANT_SET]
#   MODEL_ALIAS          claude -p --model alias, e.g. opus / sonnet / haiku (default: sonnet)
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT (default: 0)
#   TOTAL_TRIALS         Trials per (task, variant) (default: 3)
#   CLAUDE_PORT          Port for the shared claude-code-cli proxy (default: 8112)
#   TASKS                Comma-separated task short names to run (default: all). One worker
#                         per task, so this also controls how many workers launch. Choices:
#                         cube_stack,cube_restack,cube_lifting,nut_assembly,spill_wipe,
#                         two_arm_handover,two_arm_lift
#   OUTPUT_PREFIX         outputs/<prefix>_<variant>/... (default: claude_code). Use a
#                         different prefix for a rerun so it doesn't mix trial numbering with
#                         an earlier run's folders in the same outputs/claude_code_<variant>/
#                         directory (run_batch always starts counting from trial_01).
#   VARIANT_SET           "singles" (the 11 named presets, default), "pairwise" (33 two-axis
#                         combinations, e.g. "color_shift_mild+target_object_swap" -- every way
#                         to turn on exactly 2 of the 6 independent perturbation axes at once;
#                         presets on the SAME axis, like color_shift_mild/_strong, aren't
#                         combined with each other since one just overwrites the other's config
#                         key rather than compounding), or "combined" (singles + pairwise, 44
#                         variants per task -- use when a single sweep needs both the reference
#                         single-axis rates AND the pairwise interaction terms at matched
#                         trial count, without paying the shared-services startup cost twice).
#                         A "+"-joined variant is passed through PATCH_SCRIPT as a list, using
#                         the same combination support resolve_variation_config()/resolve_prompt()
#                         already have.
set -e

cd "$(git rev-parse --show-toplevel)"

MODEL_ALIAS=${1:-sonnet}
SHARED_SERVICES_GPU=${2:-0}
TOTAL_TRIALS=${3:-3}
CLAUDE_PORT=${4:-8112}
TASKS_FILTER=${5:-all}
OUTPUT_PREFIX=${6:-claude_code}
VARIANT_SET=${7:-singles}
APIS_OVERRIDE=${8:-}
MODEL="claude-code/${MODEL_ALIAS}"
CLAUDE_SERVER_URL="http://127.0.0.1:${CLAUDE_PORT}/chat/completions"

SINGLE_VARIANTS=(
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

# Every way to have exactly 2 of the 6 independent perturbation axes (color, position,
# lighting, distractor, swap, language) active at once -- see the VARIANT_SET usage note above.
PAIRWISE_VARIANTS=(
    target_object_swap+language_paraphrase
    distractor_object+language_paraphrase
    distractor_object+target_object_swap
    lighting_bright+language_paraphrase
    lighting_bright+target_object_swap
    lighting_bright+distractor_object
    lighting_dim+language_paraphrase
    lighting_dim+target_object_swap
    lighting_dim+distractor_object
    object_position_shift+language_paraphrase
    object_position_shift+target_object_swap
    object_position_shift+distractor_object
    object_position_shift+lighting_bright
    object_position_shift+lighting_dim
    robot_position_shift+language_paraphrase
    robot_position_shift+target_object_swap
    robot_position_shift+distractor_object
    robot_position_shift+lighting_bright
    robot_position_shift+lighting_dim
    color_shift_mild+language_paraphrase
    color_shift_mild+target_object_swap
    color_shift_mild+distractor_object
    color_shift_mild+lighting_bright
    color_shift_mild+lighting_dim
    color_shift_mild+object_position_shift
    color_shift_mild+robot_position_shift
    color_shift_strong+language_paraphrase
    color_shift_strong+target_object_swap
    color_shift_strong+distractor_object
    color_shift_strong+lighting_bright
    color_shift_strong+lighting_dim
    color_shift_strong+object_position_shift
    color_shift_strong+robot_position_shift
)

# Per-task difficulty-knob variant (not a shared perturbation): baseline + task's own
# harder preset. Named differently per task because each targets its own metric --
# see the VARIATIONS dict in each robosuite_*_variants.py.
declare -A TASK_HARDER_MAP=(
    [cube_stack]="stack_3_cubes"
    [cube_restack]="stack_3_cubes"
    [cube_lifting]="lift_higher"
    [nut_assembly]="both_nuts"
    [spill_wipe]="harder_spill"
    [two_arm_handover]="harder_handover"
    [two_arm_lift]="harder_lift"
)

case "$VARIANT_SET" in
    singles)      ALL_VARIANTS=("${SINGLE_VARIANTS[@]}") ;;
    pairwise)     ALL_VARIANTS=("${PAIRWISE_VARIANTS[@]}") ;;
    combined)     ALL_VARIANTS=("${SINGLE_VARIANTS[@]}" "${PAIRWISE_VARIANTS[@]}") ;;
    hard_vs_base) ALL_VARIANTS=(baseline __task_harder__) ;;
    *)
        echo "ERROR: unknown VARIANT_SET '$VARIANT_SET'. Choices: singles, pairwise, combined, hard_vs_base"
        exit 1
        ;;
esac

declare -A TASK_CONFIG_MAP=(
    [cube_stack]="env_configs/cube_stack/franka_robosuite_cube_stack_variants.yaml"
    [cube_restack]="env_configs/cube_restack/franka_robosuite_cube_restack_variants.yaml"
    [cube_lifting]="env_configs/cube_lifting/franka_robosuite_cube_lifting_variants.yaml"
    [nut_assembly]="env_configs/nut_assembly/franka_robosuite_nut_assembly_variants.yaml"
    [spill_wipe]="env_configs/spill_wipe/franka_robosuite_spill_wipe_variants.yaml"
    [two_arm_handover]="env_configs/two_arm_handover/two_arm_handover_variants.yaml"
    [two_arm_lift]="env_configs/two_arm_lift/franka_robosuite_two_arm_lift_variants.yaml"
)
TASK_ORDER=(cube_stack cube_restack cube_lifting nut_assembly spill_wipe two_arm_handover two_arm_lift)

TASK_CONFIGS=()
TASK_NAMES=()
if [ "$TASKS_FILTER" = "all" ]; then
    for t in "${TASK_ORDER[@]}"; do
        TASK_CONFIGS+=("${TASK_CONFIG_MAP[$t]}")
        TASK_NAMES+=("$t")
    done
else
    IFS=',' read -ra REQUESTED_TASKS <<< "$TASKS_FILTER"
    for t in "${REQUESTED_TASKS[@]}"; do
        if [ -z "${TASK_CONFIG_MAP[$t]:-}" ]; then
            echo "ERROR: unknown task '$t'. Choices: ${TASK_ORDER[*]}"
            exit 1
        fi
        TASK_CONFIGS+=("${TASK_CONFIG_MAP[$t]}")
        TASK_NAMES+=("$t")
    done
fi

# --- Python helper: sets low_level.variation, empties api_servers (see note above), and sets
# env.cfg.prompt to the corrected text for any preset with a known prompt/scene conflict (see
# capx/envs/simulators/variant_prompts.py) -- e.g. target_object_swap turning cubeA into a
# cylinder while the class-default prompt still says "cube". Leaves cfg.prompt unset (falls
# through to the class default) for presets with no known conflict.
PATCH_SCRIPT="/tmp/patch_capx_variant_yaml.py"
cat > "$PATCH_SCRIPT" << 'PYEOF'
import sys
import yaml

sys.path.insert(0, ".")
from capx.envs.simulators.variant_prompts import resolve_prompt

src, dst, variation_arg, task_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# Optional 5th arg: override the API class (e.g. FrankaControlApiReduced for S3 tier).
apis_override = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
# A "+"-joined variant string (e.g. "color_shift_mild+target_object_swap") is a preset
# combination -- split it into a list, matching resolve_variation_config()'s list support.
variation = variation_arg.split("+") if "+" in variation_arg else variation_arg
with open(src) as f:
    data = yaml.safe_load(f)

data["env"]["cfg"]["low_level"]["variation"] = variation
data["api_servers"] = []
if apis_override:
    data["env"]["cfg"]["apis"] = [apis_override]

corrected_prompt = resolve_prompt(task_name, variation)
if corrected_prompt is not None:
    data["env"]["cfg"]["prompt"] = corrected_prompt

with open(dst, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PYEOF

echo "=== ${#TASK_CONFIGS[@]} tasks x ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials = $(( ${#TASK_CONFIGS[@]} * ${#ALL_VARIANTS[@]} * TOTAL_TRIALS )) total trials ==="
if [ -n "$APIS_OVERRIDE" ]; then
    echo "=== APIs override: $APIS_OVERRIDE (overrides whatever the yaml's env.cfg.apis says) ==="
fi
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
    local task_short_name="$2"
    local task_name
    task_name=$(basename "$task_cfg" .yaml)
    local worker_log="/tmp/claude_sweep_${task_name}.log"
    : > "$worker_log"

    echo "[${task_name}] Starting -- ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials" | tee -a "$worker_log"

    for variant in "${ALL_VARIANTS[@]}"; do
        # hard_vs_base mode: substitute the per-task harder preset from TASK_HARDER_MAP.
        # Skip the task entirely if no harder preset is registered for it.
        if [ "$variant" = "__task_harder__" ]; then
            variant="${TASK_HARDER_MAP[$task_short_name]:-}"
            if [ -z "$variant" ]; then
                echo "[${task_name}] no harder preset for ${task_short_name}, skipping" | tee -a "$worker_log"
                continue
            fi
        fi
        echo "[${task_name}] === variant: ${variant} ===" | tee -a "$worker_log"
        local tmp_cfg="/tmp/claude_sweep_configs_${task_name}_${variant}.yaml"
        .venv/bin/python "$PATCH_SCRIPT" "$task_cfg" "$tmp_cfg" "$variant" "$task_short_name" "$APIS_OVERRIDE"

        uv run --frozen python -m capx.envs.scripts.run_batch \
            --args.config-paths "$tmp_cfg" \
            --args.models "${MODEL}" \
            --args.server-url "${CLAUDE_SERVER_URL}" \
            --args.total-trials "${TOTAL_TRIALS}" \
            --args.output-dir "outputs/${OUTPUT_PREFIX}_${variant}" \
            >> "$worker_log" 2>&1

        echo "[${task_name}] === finished variant: ${variant} ===" | tee -a "$worker_log"
    done

    echo "[${task_name}] All variants done." | tee -a "$worker_log"
}

WORKER_PIDS=()
for i in "${!TASK_CONFIGS[@]}"; do
    run_task_worker "${TASK_CONFIGS[$i]}" "${TASK_NAMES[$i]}" &
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
