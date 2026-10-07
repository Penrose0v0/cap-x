"""Success-rate tables from sweep outputs.

Walks a sweep output tree, reads each trial directory name
(`trial_NN_sandboxrc_<crash>_reward_<r>_taskcompleted_<0|1>`), and prints a task x variant
table of success rates with crash counts. Trials from several run prefixes are pooled.

    python scripts/summarize_sweeps.py                                  # heuristic solvers, singles
    python scripts/summarize_sweeps.py --presets camera                 # camera presets
    python scripts/summarize_sweeps.py --backend claude --runs claude_camera_hi_20260930
    python scripts/summarize_sweeps.py --format json > matrix.json

Heuristic trials are only counted if they were run with the task's current solver: each
solver has a "valid since" date (SOLVER_SINCE, overridable with --since task=YYYY-MM-DD),
and older trial directories are skipped. That keeps superseded solver versions, which
share the same output layout, out of the reported numbers.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys

TASKS = ["cube_lifting", "cube_stack", "cube_restack", "nut_assembly", "spill_wipe", "two_arm_handover", "two_arm_lift"]

PRESET_SETS = {
    "singles": ["baseline", "color_shift_mild", "color_shift_strong", "object_position_shift", "robot_position_shift",
                "lighting_bright", "lighting_dim", "distractor_object", "target_object_swap", "language_paraphrase", "compound_hard"],
    "camera": ["baseline", "camera_shift_mild", "camera_shift_strong"]
              + [f"camera_az{az:03d}_r{r}" for r in ("025", "100") for az in (0, 45, 90, 135, 180)],
    "difficulty": ["baseline", "lift_higher", "stack_3_cubes", "both_nuts", "harder_spill", "harder_handover", "harder_lift"],
}

# Date from which a task's heuristics/<task>/solver_s3.py has had its current content.
SOLVER_SINCE = {
    "cube_lifting": "2026-09-08", "cube_stack": "2026-09-08", "cube_restack": "2026-09-01",
    "nut_assembly": "2026-09-01", "spill_wipe": "2026-09-24", "two_arm_handover": "2026-09-09",
    "two_arm_lift": "2026-09-27T09:15",
}

TRIAL_RE = re.compile(r"trial_\d+_sandboxrc_(?P<crash>\d)_reward_(?P<reward>[0-9.]+)_taskcompleted_(?P<done>\d)$")
CFG_RE = re.compile(r"^(?P<backend>heur|claude)_sweep_configs_(?P<cfg>.+)$")


def task_and_preset(cfg_dir: str, presets: list[str]) -> tuple[str, str] | None:
    m = CFG_RE.match(cfg_dir)
    if not m:
        return None
    rest = m.group("cfg")
    for preset in sorted(presets, key=len, reverse=True):
        if rest.endswith("_" + preset):
            task = rest[: -len(preset) - 1].replace("franka_robosuite_", "").replace("_variants", "")
            return (task, preset) if task in TASKS else None
    return None


def collect(root: str, backend: str, presets: list[str], runs: list[str] | None, since: dict[str, float]):
    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    want_backend = "heur" if backend == "heuristic" else "claude"
    for dirpath, dirnames, _ in os.walk(root):
        name = os.path.basename(dirpath)
        tm = TRIAL_RE.match(name)
        if not tm:
            continue
        dirnames[:] = []
        cfg_dir = os.path.basename(os.path.dirname(dirpath))
        cm = CFG_RE.match(cfg_dir)
        if not cm or cm.group("backend") != want_backend:
            continue
        tp = task_and_preset(cfg_dir, presets)
        if tp is None:
            continue
        task, preset = tp
        rel = os.path.relpath(dirpath, root)
        if runs and not any(part.startswith(tuple(runs)) for part in rel.split(os.sep)):
            continue
        if backend == "heuristic" and os.path.getmtime(dirpath) < since.get(task, 0):
            continue
        cells[(task, preset)].append({"done": int(tm.group("done")), "crash": int(tm.group("crash")), "reward": float(tm.group("reward"))})
    return cells


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/current")
    p.add_argument("--backend", choices=["heuristic", "claude"], default="heuristic")
    p.add_argument("--presets", default="singles", help="a named set (singles/camera/difficulty) or a comma-separated list")
    p.add_argument("--tasks", default="all")
    p.add_argument("--runs", default=None, help="comma-separated run-directory prefixes to include (default: all)")
    p.add_argument("--since", action="append", default=[], metavar="TASK=YYYY-MM-DD", help="override a task's solver valid-since date")
    p.add_argument("--format", choices=["md", "json"], default="md")
    args = p.parse_args()

    presets = PRESET_SETS.get(args.presets) or args.presets.split(",")
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    since_dates = dict(SOLVER_SINCE)
    for item in args.since:
        task, date = item.split("=")
        since_dates[task] = date
    since = {t: dt.datetime.fromisoformat(d).timestamp() for t, d in since_dates.items()}
    cells = collect(args.root, args.backend, presets, args.runs.split(",") if args.runs else None, since)

    if args.format == "json":
        out = {t: {v: cells.get((t, v), []) for v in presets} for t in tasks}
        json.dump(out, sys.stdout, indent=1)
        return
    print("| task | " + " | ".join(presets) + " |")
    print("|---|" + "---|" * len(presets))
    for t in tasks:
        row = []
        for v in presets:
            trials = cells.get((t, v), [])
            if not trials:
                row.append("—")
                continue
            n, ok, crash = len(trials), sum(x["done"] for x in trials), sum(x["crash"] for x in trials)
            row.append(f"{100 * ok / n:.0f}% ({ok}/{n})" + (f" {crash}✗" if crash else ""))
        print(f"| {t} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
