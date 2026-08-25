"""Fixes the prompt/scene text conflicts documented in variants_common.py's module docstring.

Each affected task's real instruction text lives as a `PROMPT` class-attribute string on its
HIGH-level wrapper (capx/envs/tasks/franka/*.py) -- NOT on the low-level `*_variants.py` classes,
whose own CANONICAL_PROMPT/paraphrase-pool machinery is disconnected from what the model actually
sees (see the language_paraphrase note in variants_common.py's module docstring). That high-level
PROMPT is only overridable via `cfg.prompt` in a task's env_configs/*.yaml, which is unset by
default:

    # capx/envs/tasks/base.py:106-110
    # Prompt priority: YAML config (cfg.prompt) overrides the class attribute (self.prompt).
    self._task_prompt = cfg.prompt if cfg.prompt is not None else self.prompt

So the fix here is entirely additive and yaml-level: for each task with a known conflict, this
module defines a small set of word-level (find, replace) substitutions per conflicting preset.
`resolve_prompt(task, variation)` composes the right combination for a given `variation` (single
name or list, matching resolve_variation_config()'s signature); whoever builds a per-variant yaml
(a static env_configs/*.yaml default, or a sweep script's yaml-patching step) is expected to set
the result as `env.cfg.prompt`. No change to capx/envs/tasks/base.py or any *_variants.py file's
construction/reward logic -- this is a pure lookup table plus string substitution.

Two_arm_lift has no entry here: its prompt never names a color or handle shape (see
robosuite_two_arm_lift_variants.py's own note), so there's nothing to fix.

Fix-list design notes (why plain ordered find/replace, not a general templating engine):

- cube_stack / cube_restack: color_tint applies the SAME tint to both cubes, so a corrected
  description can't lean on color to tell them apart -- these switch to a SIZE-based
  description ("smaller"/"larger") instead, which stays true regardless of color_tint OR
  target_object_swap (cubeA's default half-size 0.02 stays smaller than cubeB's 0.025 even as
  the swapped cylinder, radius 0.021). Because that switch is unconditionally safe, every
  color/shape-conflicting preset for these two tasks uses the same base color-fix, with
  target_object_swap layering an extra shape-word fix on top.
- cube_lifting: single object, no disambiguation need -- color fix just drops the (now wrong)
  color adjective; shape fix swaps the now-wrong "cube" for a generic "object".
- nut_assembly: color_tint only affects the active nut, so "brown" (the peg's own
  "brown square block" wording is untouched) is what needs dropping under color_tint;
  target_object_swap needs both "brown" and "square" fixed -- the round nut isn't brown by
  default either (round-nut.xml uses a steel/grey texture vs. square-nut.xml's brass texture),
  not just a target_object_swap + color_shift combination issue. This text appears 5 times in
  the base prompt, including as the literal vision-grounding query strings the model is told to
  pass to get_object_pose/sample_grasp_pose (e.g. "extruded handle of the brown square nut"),
  so a stale word here isn't just a readability nit -- it's the actual text sent to SAM3/
  OWL-ViT's open-vocabulary grounding.
- spill_wipe: single color word ("brown"), single object, same drop-the-adjective fix as
  cube_lifting. dirt_pattern_swap's two_clusters split (singular "the spill" vs. now two
  disconnected regions) is deliberately NOT fixed here -- flagged as a softer, lower-priority
  issue in the original prompt-conflict audit, not worth the extra fix-list complexity yet.
- two_arm_handover: no color word at all in the prompt, so no color-related fix. The swap
  target's own docstring (robosuite_handover_variants.py) says the kwargs are chosen to "look
  like a visually-distinct tool (e.g. a mallet)" -- "mallet" is used here as the corrected noun
  (rather than a generic "tool") since that's literally the intended appearance, and it appears
  6 times across the Goal line and the Environment-details section, all fixed by one global
  substitution.

Fix lists are intentionally written to be safe in any order / any subset (including a
combination via resolve_variation_config()'s list support): each substitution's `find` text
matches either the pristine base prompt or an already-partially-fixed one, so applying an
inactive preset's fixes, or applying two together in either order, is idempotent rather than
double-transforming or silently missing a match.
"""

from __future__ import annotations

from capx.envs.tasks.franka.franka_cube_restack import PROMPT as _CUBE_RESTACK_PROMPT
from capx.envs.tasks.franka.franka_lift import PROMPT as _CUBE_LIFTING_PROMPT
from capx.envs.tasks.franka.franka_nut_assembly import PROMPT as _NUT_ASSEMBLY_PROMPT
from capx.envs.tasks.franka.franka_pick_place import PROMPT as _CUBE_STACK_PROMPT
from capx.envs.tasks.franka.franka_spill_wipe import PROMPT as _SPILL_WIPE_PROMPT
from capx.envs.tasks.franka.two_arm_handover import PROMPT as _TWO_ARM_HANDOVER_PROMPT

TASK_PROMPTS: dict[str, str] = {
    "cube_stack": _CUBE_STACK_PROMPT,
    "cube_restack": _CUBE_RESTACK_PROMPT,
    "cube_lifting": _CUBE_LIFTING_PROMPT,
    "nut_assembly": _NUT_ASSEMBLY_PROMPT,
    "spill_wipe": _SPILL_WIPE_PROMPT,
    "two_arm_handover": _TWO_ARM_HANDOVER_PROMPT,
}

_STACK_COLOR_FIX: list[tuple[str, str]] = [
    ("red cube", "smaller cube"),
    ("green cube", "larger cube"),
    ("green_center_z", "larger_center_z"),
    ("green_extent", "larger_extent"),
    ("red_extent", "smaller_extent"),
]
_STACK_SHAPE_FIX: list[tuple[str, str]] = [("smaller cube", "smaller object")]

_RESTACK_COLOR_FIX: list[tuple[str, str]] = [
    ("red cube", "smaller cube"),
    ("green cube", "larger cube"),
]
_RESTACK_SHAPE_FIX: list[tuple[str, str]] = [("smaller cube", "smaller object")]

_LIFT_COLOR_FIX: list[tuple[str, str]] = [("red ", "")]
_LIFT_SHAPE_FIX: list[tuple[str, str]] = [("cube", "object")]

_NUT_COLOR_FIX: list[tuple[str, str]] = [("brown square nut", "square nut")]
_NUT_SHAPE_FIX: list[tuple[str, str]] = [
    ("brown square nut", "round nut"),
    ("square nut", "round nut"),
]

_WIPE_COLOR_FIX: list[tuple[str, str]] = [("brown ", "")]

_HANDOVER_SHAPE_FIX: list[tuple[str, str]] = [("hammer", "mallet")]

TASK_PROMPT_FIXES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "cube_stack": {
        "color_shift_mild": _STACK_COLOR_FIX,
        "color_shift_strong": _STACK_COLOR_FIX,
        "compound_hard": _STACK_COLOR_FIX,
        "target_object_swap": _STACK_COLOR_FIX + _STACK_SHAPE_FIX,
    },
    "cube_restack": {
        "color_shift_mild": _RESTACK_COLOR_FIX,
        "color_shift_strong": _RESTACK_COLOR_FIX,
        "compound_hard": _RESTACK_COLOR_FIX,
        "target_object_swap": _RESTACK_COLOR_FIX + _RESTACK_SHAPE_FIX,
    },
    "cube_lifting": {
        "color_shift_mild": _LIFT_COLOR_FIX,
        "color_shift_strong": _LIFT_COLOR_FIX,
        "compound_hard": _LIFT_COLOR_FIX,
        "target_object_swap": _LIFT_SHAPE_FIX,
    },
    "nut_assembly": {
        "color_shift_mild": _NUT_COLOR_FIX,
        "color_shift_strong": _NUT_COLOR_FIX,
        "compound_hard": _NUT_COLOR_FIX,
        "target_object_swap": _NUT_SHAPE_FIX,
    },
    "spill_wipe": {
        "color_shift_mild": _WIPE_COLOR_FIX,
        "color_shift_strong": _WIPE_COLOR_FIX,
        "compound_hard": _WIPE_COLOR_FIX,
    },
    "two_arm_handover": {
        "target_object_swap": _HANDOVER_SHAPE_FIX,
    },
}


def resolve_prompt(task: str, variation: str | list[str]) -> str | None:
    """Returns the corrected prompt for `task` under `variation`, or None if `task` isn't
    registered (no known conflict) or none of the active preset(s) need a fix -- callers should
    treat None as "leave cfg.prompt unset, the class default is already correct."

    `variation` may be a single preset name or a list (matching resolve_variation_config's
    signature) to support the same preset-combination feature.
    """
    if task not in TASK_PROMPTS:
        return None

    names = [variation] if isinstance(variation, str) else list(variation)
    fixes = TASK_PROMPT_FIXES[task]
    active_fixes = [fixes[n] for n in names if n in fixes]
    if not active_fixes:
        return None

    text = TASK_PROMPTS[task]
    for fix_list in active_fixes:
        for find, replace in fix_list:
            text = text.replace(find, replace)
    return text
