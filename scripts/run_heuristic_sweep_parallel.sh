#!/bin/bash
# Run star's hand-written S3 solvers (cube_restack + nut_assembly) against every named
# variant/preset combo of their respective tasks. Sibling of run_claude_code_sweep_parallel.sh:
# same shared-services setup, same PATCH_SCRIPT for variant yaml patching, same run_batch
# entry point -- only two differences:
#
#   1. The "LLM" is a stub HTTP server (capx/serving/static_solver_server.py) that just
#      returns the contents of heuristics/<task>/solver_s3.py wrapped in a ```python fence.
#      No claude-code CLI, no OAuth token, no per-request cost.
#   2. MODEL is set PER WORKER as "heuristic/<task_short_name>", so the stub routes each
#      task's requests to the right solver file.
#
# Usage: bash scripts/run_heuristic_sweep_parallel.sh [SHARED_SERVICES_GPU] [TOTAL_TRIALS] [STUB_PORT] [TASKS] [OUTPUT_PREFIX] [VARIANT_SET]
#   SHARED_SERVICES_GPU  GPU for the shared SAM3/SAM2/GraspNet/PyRoki/OWL-ViT (default: 0)
#   TOTAL_TRIALS         Trials per (task, variant) (default: 10). May also be a path to a
#                         TSV of "<task> <variant> <n>" lines to run a different count per
#                         cell (cells missing from the file or with n<=0 are skipped).
#   STUB_PORT            Port for the static-solver stub (default: 8118)
#   TASKS                Comma-separated task short names (default: cube_restack,nut_assembly).
#                         Must have a heuristics/<task>/solver_s3.py file.
#   OUTPUT_PREFIX         outputs/<prefix>_<variant>/... (default: heuristic)
#   VARIANT_SET           "singles" (default, 11 named presets), "pairwise" (33 two-axis
#                         combos), or "combined" (both).
set -e

cd "$(git rev-parse --show-toplevel)"

SHARED_SERVICES_GPU=${1:-0}
TOTAL_TRIALS=${2:-10}
STUB_PORT=${3:-8118}
TASKS_FILTER=${4:-cube_restack,nut_assembly}
OUTPUT_PREFIX=${5:-heuristic}
VARIANT_SET=${6:-singles}
STUB_SERVER_URL="http://127.0.0.1:${STUB_PORT}/chat/completions"

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

# Every way to have exactly 2 of the 6 independent perturbation axes active at once --
# see run_claude_code_sweep_parallel.sh's header for the axis-vs-preset explanation.
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

case "$VARIANT_SET" in
    baseline) ALL_VARIANTS=(baseline) ;;
    singles)  ALL_VARIANTS=("${SINGLE_VARIANTS[@]}") ;;
    pairwise) ALL_VARIANTS=("${PAIRWISE_VARIANTS[@]}") ;;
    combined) ALL_VARIANTS=("${SINGLE_VARIANTS[@]}" "${PAIRWISE_VARIANTS[@]}") ;;
    *)
        echo "ERROR: unknown VARIANT_SET '$VARIANT_SET'. Choices: baseline, singles, pairwise, combined"
        exit 1
        ;;
esac

declare -A TASK_CONFIG_MAP=(
    [cube_restack]="env_configs/cube_restack/franka_robosuite_cube_restack_variants.yaml"
    [nut_assembly]="env_configs/nut_assembly/franka_robosuite_nut_assembly_variants.yaml"
    [cube_lifting]="env_configs/cube_lifting/franka_robosuite_cube_lifting_variants.yaml"
    [cube_stack]="env_configs/cube_stack/franka_robosuite_cube_stack_variants.yaml"
    [spill_wipe]="env_configs/spill_wipe/franka_robosuite_spill_wipe_variants.yaml"
    [two_arm_handover]="env_configs/two_arm_handover/two_arm_handover_variants.yaml"
    [two_arm_lift]="env_configs/two_arm_lift/franka_robosuite_two_arm_lift_variants.yaml"
)
TASK_ORDER=(cube_restack nut_assembly cube_lifting cube_stack spill_wipe two_arm_handover two_arm_lift)

TASK_CONFIGS=()
TASK_NAMES=()
IFS=',' read -ra REQUESTED_TASKS <<< "$TASKS_FILTER"
for t in "${REQUESTED_TASKS[@]}"; do
    if [ -z "${TASK_CONFIG_MAP[$t]:-}" ]; then
        echo "ERROR: unknown or unsupported heuristic task '$t'. Available: ${TASK_ORDER[*]}"
        echo "       (a task needs heuristics/<task>/solver_s3.py to be runnable)"
        exit 1
    fi
    if [ ! -f "heuristics/${t}/solver_s3.py" ]; then
        echo "ERROR: heuristics/${t}/solver_s3.py not found -- can't run task '$t'"
        exit 1
    fi
    TASK_CONFIGS+=("${TASK_CONFIG_MAP[$t]}")
    TASK_NAMES+=("$t")
done

# --- Python helper: sets low_level.variation, empties api_servers, sets env.cfg.prompt to the
# corrected text for any preset with a known prompt/scene conflict. Identical to the one in
# run_claude_code_sweep_parallel.sh -- kept inline so this script is standalone.
PATCH_SCRIPT="/tmp/patch_capx_variant_yaml_heur.py"
cat > "$PATCH_SCRIPT" << 'PYEOF'
import sys
import yaml

sys.path.insert(0, ".")
from capx.envs.simulators.variant_prompts import resolve_prompt

src, dst, variation_arg, task_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
variation = variation_arg.split("+") if "+" in variation_arg else variation_arg
with open(src) as f:
    data = yaml.safe_load(f)

data["env"]["cfg"]["low_level"]["variation"] = variation
data["api_servers"] = []
# S3 tier: swap the higher-level FrankaControlApi (get_object_pose/goto_pose) for
# FrankaControlApiReduced, which exposes the low-level primitives star's solver needs
# (get_observation / plan_grasp / solve_ik / move_to_joints / open+close_gripper /
# get_oriented_bounding_box_from_3d_points). Every task listed in TASK_CONFIG_MAP is
# expected to work at S3 tier; the API class itself gates by task via internal flags.
# spill_wipe needs a shorter TCP offset (sponge attachment); switch API accordingly.
if task_name == "spill_wipe":
    data["env"]["cfg"]["apis"] = ["FrankaControlSpillWipeApiReduced"]
elif task_name == "two_arm_handover":
    data["env"]["cfg"]["apis"] = ["FrankaHandoverApiReduced"]
elif task_name == "two_arm_lift":
    data["env"]["cfg"]["apis"] = ["FrankaControlApiReducedBimanual"]
else:
    data["env"]["cfg"]["apis"] = ["FrankaControlApiReduced"]

corrected_prompt = resolve_prompt(task_name, variation)
if corrected_prompt is not None:
    data["env"]["cfg"]["prompt"] = corrected_prompt

with open(dst, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PYEOF

echo "=== ${#TASK_CONFIGS[@]} tasks x ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials = $(( ${#TASK_CONFIGS[@]} * ${#ALL_VARIANTS[@]} * TOTAL_TRIALS )) total trials ==="
echo "=== Backend: static-solver stub on port ${STUB_PORT}; shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

ALL_BG_PIDS=()
cleanup() {
    echo ""
    echo "=== Shutting down all background servers: ${ALL_BG_PIDS[*]} ==="
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

# --- Reuse ONE shared set of vision/planning services (same as claude-code sweep).
# If run_claude_code_sweep_parallel.sh is ALREADY running with the same GPU, its shared
# services on ports 8113-8117 will be reused -- this script will detect the existing bind
# via wait_for_port passing immediately and just move on.
echo "=== Starting shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="

start_or_reuse_service() {
    local name="$1" port="$2" module="$3"
    local logfile="$4"
    shift 4
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/openapi.json" > /dev/null 2>&1; then
        echo "  ${name} already up on port ${port} -- reusing"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m "$module" \
        "$@" > "$logfile" 2>&1 &
    ALL_BG_PIDS+=("$!")
    local pid=$!
    wait_for_port "$name" "$port" "$pid" "$logfile"
}

start_or_reuse_service "SAM3"     8114 capx.serving.launch_sam3_server \
    "/tmp/heur_sweep_shared_sam3.log" \
    --device cuda --port 8114 --host 127.0.0.1

start_or_reuse_service "GraspNet" 8115 capx.serving.launch_contact_graspnet_server \
    "/tmp/heur_sweep_shared_graspnet.log" \
    --port 8115 --host 127.0.0.1

start_or_reuse_service "PyRoki"   8116 capx.serving.launch_pyroki_server \
    "/tmp/heur_sweep_shared_pyroki.log"

start_or_reuse_service "SAM2"     8113 capx.serving.launch_sam2_server \
    "/tmp/heur_sweep_shared_sam2.log" \
    --port 8113 --host 127.0.0.1 --model-name facebook/sam2.1-hiera-large

start_or_reuse_service "OWL-ViT"  8117 capx.serving.launch_owlvit_server \
    "/tmp/heur_sweep_shared_owlvit.log" \
    --port 8117 --host 127.0.0.1 --model-name google/owlv2-large-patch14-ensemble

echo "=== Shared vision/planning services ready. ==="

# --- Start the static-solver stub (fresh port; own to this sweep).
STUB_LOG="/tmp/heur_sweep_stub.log"
setsid .venv/bin/python -m capx.serving.static_solver_server \
    --solver-root heuristics --host 127.0.0.1 --port "${STUB_PORT}" > "$STUB_LOG" 2>&1 &
ALL_BG_PIDS+=("$!")
STUB_PID=$!
wait_for_port "static-solver stub" "${STUB_PORT}" "$STUB_PID" "$STUB_LOG" "/health"
echo "=== static-solver stub ready (port ${STUB_PORT}). ==="

run_task_worker() {
    local task_cfg="$1"
    local task_short_name="$2"
    local task_name
    task_name=$(basename "$task_cfg" .yaml)
    local worker_log="/tmp/heur_sweep_${task_name}.log"
    : > "$worker_log"
    local MODEL="heuristic/${task_short_name}"

    echo "[${task_name}] Starting (model=${MODEL}) -- ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials" | tee -a "$worker_log"

    for variant in "${ALL_VARIANTS[@]}"; do
        local n_trials="$TOTAL_TRIALS"
        if [ -f "$TOTAL_TRIALS" ]; then
            n_trials=$(awk -v t="$task_short_name" -v v="$variant" '$1==t && $2==v {print $3}' "$TOTAL_TRIALS")
            if [ -z "$n_trials" ] || [ "$n_trials" -le 0 ]; then
                echo "[${task_name}] === variant: ${variant} -- nothing to run, skipping ===" | tee -a "$worker_log"
                continue
            fi
        fi
        echo "[${task_name}] === variant: ${variant} (${n_trials} trials) ===" | tee -a "$worker_log"
        local tmp_cfg="/tmp/heur_sweep_configs_${task_name}_${variant}.yaml"
        .venv/bin/python "$PATCH_SCRIPT" "$task_cfg" "$tmp_cfg" "$variant" "$task_short_name"

        uv run --frozen python -m capx.envs.scripts.run_batch \
            --args.config-paths "$tmp_cfg" \
            --args.models "${MODEL}" \
            --args.server-url "${STUB_SERVER_URL}" \
            --args.total-trials "${n_trials}" \
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
    echo "    tail -f /tmp/heur_sweep_${task_name}.log"
done

wait "${WORKER_PIDS[@]}"
echo "=== All task workers finished. ==="
