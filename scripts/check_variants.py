"""Snapshot / regression check for the robustness-eval variant envs.

Builds every (task, preset) env, resets it with a fixed seed, and records what the policy
would see: the first observation's RGB/depth statistics, the camera pose, and the prompt
text. Writing a snapshot before a refactor and checking against it afterwards catches
any behaviour change in the variant machinery without running a single solver.

    python scripts/check_variants.py --write tests/snapshots/variants.json
    python scripts/check_variants.py --check tests/snapshots/variants.json
    python scripts/check_variants.py --check ... --tasks cube_stack,spill_wipe --presets baseline,camera_shift_mild

Every env is constructed fresh per preset (some presets act in __init__/_load_model).
Expect roughly 5-10 s per cell, ~170 cells for the full set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())

TASKS: dict[str, tuple[str, str]] = {
    "cube_lifting": ("robosuite_cube_lift_variants", "FrankaRobosuiteCubeLiftVariantsLowLevel"),
    "cube_stack": ("robosuite_cubes_variants", "FrankaRobosuiteCubesVariantsLowLevel"),
    "cube_restack": ("robosuite_cubes_restack_variants", "FrankaRobosuiteCubesRestackVariantsLowLevel"),
    "nut_assembly": ("robosuite_nut_assembly_variants", "FrankaRobosuiteNutAssemblyVariantsLowLevel"),
    "spill_wipe": ("robosuite_spill_wipe_variants", "FrankaRobosuiteSpillWipeVariantsLowLevel"),
    "two_arm_handover": ("robosuite_handover_variants", "RobosuiteHandoverVariantsEnv"),
    "two_arm_lift": ("robosuite_two_arm_lift_variants", "RobosuiteTwoArmLiftVariantsEnv"),
}

# Tolerances for --check. Rendering is deterministic on one machine, but driver/GPU
# differences can shift pixels slightly, so images are compared by channel means.
TOL = {"rgb_mean": 1.0, "depth": 0.005, "camera_pose": 1e-4}


def snapshot_cell(task: str, preset: str, seed: int) -> dict[str, Any]:
    module_name, class_name = TASKS[task]
    module = importlib.import_module(f"capx.envs.simulators.{module_name}")
    from capx.envs.simulators.variant_prompts import resolve_prompt

    env = getattr(module, class_name)(privileged=False, enable_render=True, seed=seed, variation=preset)
    env.reset()
    obs = env.get_observation()
    key = "robot0_robotview" if "robot0_robotview" in obs else "agentview"
    rgb = np.asarray(obs[key]["images"]["rgb"], dtype=np.float64)
    depth = np.asarray(obs[key]["images"]["depth"], dtype=np.float64)
    depth = depth[..., 0] if depth.ndim == 3 else depth
    finite = depth[np.isfinite(depth)]
    rs = env.robosuite_env
    cell = {
        "camera_key": key,
        "rgb_shape": list(rgb.shape),
        "rgb_mean": [round(float(v), 3) for v in rgb.reshape(-1, rgb.shape[-1]).mean(axis=0)],
        "rgb_md5": hashlib.md5(np.asarray(obs[key]["images"]["rgb"], dtype=np.uint8).tobytes()).hexdigest(),
        "depth_stats": [round(float(v), 4) for v in (finite.mean(), *np.percentile(finite, [10, 50, 90]))],
        "camera_pose": [round(float(v), 6) for v in np.asarray(obs[key]["pose"]).reshape(-1)],
        "scene_lang": str(rs.get_ep_meta().get("lang", "")),
        "prompt_fix": resolve_prompt(task, preset),
        "variation_config": json.loads(json.dumps(rs.variation_config, default=str, sort_keys=True)),
    }
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass
    return cell


def presets_for(task: str) -> list[str]:
    module = importlib.import_module(f"capx.envs.simulators.{TASKS[task][0]}")
    return list(module.VARIATIONS)


def compare(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    diffs = []
    for k in ("camera_key", "rgb_shape", "scene_lang", "prompt_fix", "variation_config"):
        if old.get(k) != new.get(k):
            diffs.append(f"{k}: {old.get(k)!r} -> {new.get(k)!r}")
    if max(abs(a - b) for a, b in zip(old["rgb_mean"], new["rgb_mean"])) > TOL["rgb_mean"]:
        diffs.append(f"rgb_mean: {old['rgb_mean']} -> {new['rgb_mean']}")
    if max(abs(a - b) for a, b in zip(old["depth_stats"], new["depth_stats"])) > TOL["depth"]:
        diffs.append(f"depth_stats: {old['depth_stats']} -> {new['depth_stats']}")
    if max(abs(a - b) for a, b in zip(old["camera_pose"], new["camera_pose"])) > TOL["camera_pose"]:
        diffs.append(f"camera_pose: {old['camera_pose']} -> {new['camera_pose']}")
    if old["rgb_md5"] != new["rgb_md5"] and not diffs:
        diffs.append("rgb_md5 differs (pixels changed within tolerance; inspect if unexpected)")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", metavar="PATH", help="build every cell and write the snapshot here")
    mode.add_argument("--check", metavar="PATH", help="rebuild cells and compare against this snapshot")
    parser.add_argument("--tasks", default="all", help="comma-separated task names (default: all)")
    parser.add_argument("--presets", default="all", help="comma-separated preset names (default: every preset of each task)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    wanted = None if args.presets == "all" else set(args.presets.split(","))
    old = json.load(open(args.check)) if args.check else {}
    result: dict[str, dict[str, Any]] = {}
    failures = 0
    started = time.time()
    for task in tasks:
        for preset in presets_for(task):
            if wanted and preset not in wanted:
                continue
            cell_id = f"{task}/{preset}"
            try:
                cell = snapshot_cell(task, preset, args.seed)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"ERROR {cell_id}: {type(exc).__name__}: {exc}", flush=True)
                failures += 1
                continue
            result[cell_id] = cell
            if args.check:
                if cell_id not in old:
                    print(f"NEW   {cell_id} (not in snapshot)", flush=True)
                    continue
                diffs = compare(old[cell_id], cell)
                if diffs:
                    failures += 1
                    print(f"DIFF  {cell_id}\n    " + "\n    ".join(diffs), flush=True)
                else:
                    print(f"ok    {cell_id}", flush=True)
            else:
                print(f"wrote {cell_id}", flush=True)

    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)), exist_ok=True)
        json.dump(result, open(args.write, "w"), indent=1, sort_keys=True)
        print(f"\nwrote {len(result)} cells to {args.write} in {time.time() - started:.0f}s ({failures} errors)")
    else:
        missing = sorted(set(old) - set(result)) if not wanted and args.tasks == "all" else []
        for cell_id in missing:
            print(f"GONE  {cell_id} (in snapshot, not rebuilt)")
        print(f"\nchecked {len(result)} cells in {time.time() - started:.0f}s: {failures} differing/erroring, {len(missing)} missing")
        sys.exit(1 if failures or missing else 0)


if __name__ == "__main__":
    main()
