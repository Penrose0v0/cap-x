#!/bin/bash
# Shared pieces of the variant sweeps (sourced by run_heuristic_sweep_parallel.sh and
# run_claude_code_sweep_parallel.sh): the variant sets, the task -> yaml map, the shared
# vision/planning services, background-process cleanup, and the per-task worker loop.
#
# A sweep script sets these before calling the functions below:
#   SHARED_SERVICES_GPU   GPU for the shared services and the render workers' EGL contexts
#   TOTAL_TRIALS          trials per cell, or a path to a TSV of "<task> <variant> <n>" lines
#   OUTPUT_PREFIX         outputs/<prefix>_<variant>/...
#   SWEEP_LOG_PREFIX      /tmp/<prefix>_<task>.log (and <prefix>_shared_<service>.log)
#   SWEEP_SERVER_URL      chat/completions endpoint the run_batch workers query
#   SWEEP_APIS_OVERRIDE   "" (yaml default), "S3", or an explicit API class name

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
    camera_shift_mild
    camera_shift_strong
)

# Every way to have exactly 2 of the 6 independent perturbation axes (color, position,
# lighting, distractor, swap, language) active at once. Presets on the same axis (e.g.
# color_shift_mild/_strong) are not combined: one would just overwrite the other's config key.
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

# sweep_resolve_variant_set <singles|pairwise|combined|hard_vs_base> -> ALL_VARIANTS
# hard_vs_base runs baseline plus each task's own difficulty preset (DIFFICULTY_PRESETS in
# capx/envs/simulators/variants_common.py), looked up per task in sweep_run_task_worker.
sweep_resolve_variant_set() {
    case "$1" in
        singles)      ALL_VARIANTS=("${SINGLE_VARIANTS[@]}") ;;
        pairwise)     ALL_VARIANTS=("${PAIRWISE_VARIANTS[@]}") ;;
        combined)     ALL_VARIANTS=("${SINGLE_VARIANTS[@]}" "${PAIRWISE_VARIANTS[@]}") ;;
        hard_vs_base) ALL_VARIANTS=(baseline __task_harder__) ;;
        *)
            echo "ERROR: unknown VARIANT_SET '$1'. Choices: singles, pairwise, combined, hard_vs_base"
            exit 1
            ;;
    esac
}

# sweep_resolve_tasks <all|comma-separated task names> -> TASK_CONFIGS, TASK_NAMES
sweep_resolve_tasks() {
    TASK_CONFIGS=()
    TASK_NAMES=()
    local requested=()
    if [ "$1" = "all" ]; then
        requested=("${TASK_ORDER[@]}")
    else
        IFS=',' read -ra requested <<< "$1"
    fi
    local t
    for t in "${requested[@]}"; do
        if [ -z "${TASK_CONFIG_MAP[$t]:-}" ]; then
            echo "ERROR: unknown task '$t'. Choices: ${TASK_ORDER[*]}"
            exit 1
        fi
        TASK_CONFIGS+=("${TASK_CONFIG_MAP[$t]}")
        TASK_NAMES+=("$t")
    done
}

sweep_print_plan() {
    if [ -f "$TOTAL_TRIALS" ]; then
        echo "=== ${#TASK_CONFIGS[@]} tasks; per-cell variants and trial counts from ${TOTAL_TRIALS} ==="
    else
        echo "=== ${#TASK_CONFIGS[@]} tasks x ${#ALL_VARIANTS[@]} variants x ${TOTAL_TRIALS} trials = $(( ${#TASK_CONFIGS[@]} * ${#ALL_VARIANTS[@]} * TOTAL_TRIALS )) total trials ==="
    fi
    if [ -n "$SWEEP_APIS_OVERRIDE" ]; then
        echo "=== APIs override: ${SWEEP_APIS_OVERRIDE} (overrides whatever the yaml's env.cfg.apis says) ==="
    fi
}

# --- Background processes: everything started with setsid is its own process group, so
# signalling -$pid also reaches any children it spawned (multiprocessing pools, uv wrappers).
ALL_BG_PIDS=()
sweep_cleanup() {
    echo ""
    echo "=== Shutting down all background servers: ${ALL_BG_PIDS[*]} ==="
    local pid
    for pid in "${ALL_BG_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
    sleep 2
    for pid in "${ALL_BG_PIDS[@]}"; do kill -KILL -- "-${pid}" 2>/dev/null || true; done
    for pid in "${ALL_BG_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap sweep_cleanup EXIT

wait_for_port() {
    local name="$1" port="$2" pid="$3" logfile="$4" path="${5:-/openapi.json}"
    local i
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

# start_or_reuse_service <name> <port> <module> <logfile> [module args...]
# Reuses a service already listening on the port (e.g. another sweep's), else starts it on
# SHARED_SERVICES_GPU and waits for it.
start_or_reuse_service() {
    local name="$1" port="$2" module="$3" logfile="$4"
    shift 4
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/openapi.json" > /dev/null 2>&1; then
        echo "  ${name} already up on port ${port} -- reusing"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=${SHARED_SERVICES_GPU} setsid .venv/bin/python -m "$module" "$@" > "$logfile" 2>&1 &
    ALL_BG_PIDS+=("$!")
    wait_for_port "$name" "$port" "$!" "$logfile"
}

# The one shared set of vision/planning services every sweep needs (fixed ports; the
# per-variant yaml copies get api_servers emptied so each trial talks to these instead).
sweep_start_shared_services() {
    echo "=== Starting shared vision/planning services on GPU ${SHARED_SERVICES_GPU} ==="
    local lp="/tmp/${SWEEP_LOG_PREFIX}_shared"
    start_or_reuse_service "SAM3"     8114 capx.serving.launch_sam3_server "${lp}_sam3.log" \
        --device cuda --port 8114 --host 127.0.0.1
    start_or_reuse_service "GraspNet" 8115 capx.serving.launch_contact_graspnet_server "${lp}_graspnet.log" \
        --port 8115 --host 127.0.0.1
    # launch_pyroki_server's __main__ ignores CLI flags; its defaults (panda, port 8116) are what we want.
    start_or_reuse_service "PyRoki"   8116 capx.serving.launch_pyroki_server "${lp}_pyroki.log"
    start_or_reuse_service "SAM2"     8113 capx.serving.launch_sam2_server "${lp}_sam2.log" \
        --port 8113 --host 127.0.0.1 --model-name facebook/sam2.1-hiera-large
    start_or_reuse_service "OWL-ViT"  8117 capx.serving.launch_owlvit_server "${lp}_owlvit.log" \
        --port 8117 --host 127.0.0.1 --model-name google/owlv2-large-patch14-ensemble
    echo "=== Shared vision/planning services ready. ==="
}

# sweep_run_task_worker <task_cfg.yaml> <task_short_name> <model>
# Runs this task's variants sequentially: ALL_VARIANTS, or, with a TSV plan, exactly the
# cells the plan lists for the task (any preset the task's VARIATIONS table knows).
sweep_run_task_worker() {
    local task_cfg="$1" task_short_name="$2" model="$3"
    local task_name
    task_name=$(basename "$task_cfg" .yaml)
    local worker_log="/tmp/${SWEEP_LOG_PREFIX}_${task_name}.log"
    : > "$worker_log"

    local -a variants_to_run=("${ALL_VARIANTS[@]}")
    if [ -f "$TOTAL_TRIALS" ]; then
        mapfile -t variants_to_run < <(awk -v t="$task_short_name" '$1==t && $3>0 {print $2}' "$TOTAL_TRIALS")
    fi
    echo "[${task_name}] Starting (model=${model}) -- ${#variants_to_run[@]} variants x ${TOTAL_TRIALS} trials" | tee -a "$worker_log"

    local variant n_trials tmp_cfg
    for variant in "${variants_to_run[@]}"; do
        if [ "$variant" = "__task_harder__" ]; then
            variant=$(.venv/bin/python -c "from capx.envs.simulators.variants_common import DIFFICULTY_PRESETS as d; print(d.get('${task_short_name}', ''))")
            if [ -z "$variant" ]; then
                echo "[${task_name}] no difficulty preset for ${task_short_name}, skipping" | tee -a "$worker_log"
                continue
            fi
        fi
        n_trials="$TOTAL_TRIALS"
        if [ -f "$TOTAL_TRIALS" ]; then
            n_trials=$(awk -v t="$task_short_name" -v v="$variant" '$1==t && $2==v {print $3}' "$TOTAL_TRIALS")
        fi
        echo "[${task_name}] === variant: ${variant} (${n_trials} trials) ===" | tee -a "$worker_log"
        tmp_cfg="/tmp/${SWEEP_LOG_PREFIX}_configs_${task_name}_${variant}.yaml"
        .venv/bin/python -m capx.envs.scripts.patch_variant_config "$task_cfg" "$tmp_cfg" "$variant" "$task_short_name" \
            ${SWEEP_APIS_OVERRIDE:+--apis "$SWEEP_APIS_OVERRIDE"}

        # Pin the render workers' EGL contexts to the shared GPU: left unset, mujoco takes the
        # first EGL device that initializes, which on a shared box can be a GPU saturated by
        # someone else -- the workers then die on driver OOM and run_batch hangs on its dead pool.
        MUJOCO_EGL_DEVICE_ID="${SHARED_SERVICES_GPU}" \
        uv run --frozen python -m capx.envs.scripts.run_batch \
            --args.config-paths "$tmp_cfg" \
            --args.models "${model}" \
            --args.server-url "${SWEEP_SERVER_URL}" \
            --args.total-trials "${n_trials}" \
            --args.output-dir "outputs/${OUTPUT_PREFIX}_${variant}" \
            >> "$worker_log" 2>&1

        echo "[${task_name}] === finished variant: ${variant} ===" | tee -a "$worker_log"
    done
    echo "[${task_name}] All variants done." | tee -a "$worker_log"
}

# sweep_launch_workers <model-or-"per-task">
# One background worker per task. With "per-task" the model is "heuristic/<task>" (the
# static-solver stub routes on it); otherwise every worker uses the given model string.
sweep_launch_workers() {
    local model_spec="$1"
    WORKER_PIDS=()
    local i model
    for i in "${!TASK_CONFIGS[@]}"; do
        if [ "$model_spec" = "per-task" ]; then model="heuristic/${TASK_NAMES[$i]}"; else model="$model_spec"; fi
        sweep_run_task_worker "${TASK_CONFIGS[$i]}" "${TASK_NAMES[$i]}" "$model" &
        WORKER_PIDS+=("$!")
    done
    echo "=== All ${#TASK_CONFIGS[@]} task workers launched. Tail their logs with: ==="
    local task_cfg
    for task_cfg in "${TASK_CONFIGS[@]}"; do
        echo "    tail -f /tmp/${SWEEP_LOG_PREFIX}_$(basename "$task_cfg" .yaml).log"
    done
    # Wait on the workers only: the shared services are also background jobs of this shell
    # and never exit on their own, so a bare `wait` would hang forever.
    wait "${WORKER_PIDS[@]}"
    echo "=== All task workers finished. ==="
}
