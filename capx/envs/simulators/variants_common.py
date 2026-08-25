"""Shared helper for resolving named robustness-eval variation presets.

Each of the 7 `*_variants.py` task wrappers (robosuite_cubes_variants.py,
robosuite_cubes_restack_variants.py, robosuite_cube_lift_variants.py,
robosuite_nut_assembly_variants.py, robosuite_spill_wipe_variants.py,
robosuite_handover_variants.py, robosuite_two_arm_lift_variants.py) defines its
own module-level `VARIATIONS: dict[str, dict[str, Any]]` preset table, and used
to resolve a single preset name into `self.variation_config` with an identical
block of code duplicated in all 7 files. This factors that block out and extends
it to accept multiple preset names at once, merged into one combined config.

KNOWN ISSUE -- prompt/scene text conflicts (not fixed, documented only):

Every task's instruction text ("Goal: ..." in capx/envs/tasks/franka/*.py) is a
fixed string written once, naming a specific color or shape it expects to see.
Two preset dimensions change the scene without touching that text:

  - `color_tint`: 5 of 7 tasks (cube_stack, cube_restack, cube_lifting,
    nut_assembly, spill_wipe) have a prompt that names a specific color (e.g.
    "the red cube", "the brown spill"). Under a tint preset the named color is
    no longer accurate -- for cube_stack/cube_restack specifically, the SAME
    tint is applied to both objects, so they also stop being distinguishable
    from each other by color, not just mis-described. two_arm_lift and
    two_arm_handover's prompts don't name a color at all, so their color
    presets are unaffected.
  - `swap_target` / `swap_nut_type` / equivalent (the `target_object_swap`
    preset): 4 of 7 tasks (cube_stack, cube_lifting, nut_assembly,
    two_arm_handover) have a prompt that names the object's shape/type (e.g.
    "the cube", "square nut", "the hammer"), while the swap changes exactly
    that -- cube -> cylinder, square -> round, hammer -> mallet-shaped. The
    two_arm_handover swap's docstring (robosuite_handover_variants.py) is
    explicit that this is deliberate: the swap kwargs are chosen to "look like
    a visually-distinct tool," which is precisely what makes "the hammer"
    stop matching. two_arm_lift's swap (POT_CONFIG_SWAPS) is the one
    swap-preset written to avoid this -- it only changes material/thickness,
    and its prompt never names a handle shape or color.
  - `compound_hard` includes `color_tint` for every task that has one, so it
    inherits whichever conflict its host task already has.

This was surfaced during a real Claude-backed eval sweep (see git history around
the `claude_code_*` sweep scripts and the resulting outputs/ data) and rendered
for visual confirmation, not just reasoned about in the abstract.

Deliberately NOT fixed here. Fixing it for real needs either (a) rewriting each
task's prompt to drop color/shape words that aren't needed to disambiguate the
scene, which changes baseline semantics for tasks that rely on color to tell two
objects apart (cube_stack/cube_restack), or (b) making the prompt itself derive
from `variation_config` at resolve time, which first requires reconciling this
resolver's `self.variation_config` with the *separate*, currently-disconnected
prompt text set by each task's high-level wrapper (capx/envs/tasks/franka/*.py's
`self.prompt` / `cfg.prompt` -- see FrankaPickPlaceCodeEnv, which is NOT wired to
StackVariants's own `get_ep_meta()["lang"]`/paraphrase-pool machinery despite a
yaml comment in env_configs/cube_stack/*.yaml claiming otherwise). Until one of
those happens, treat any preset that stacks `color_tint` or `swap_target` (single
or via `compound_hard`) as testing a compound signal: perception-under-shift
*and* whether the policy over-trusts stale instruction text over what it
actually observes -- not a clean isolation of the former alone.
"""

from __future__ import annotations

from typing import Any, Sequence


def resolve_variation_config(
    variations: dict[str, dict[str, Any]],
    variation: str | Sequence[str],
    variation_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one or more named presets from `variations` into a merged config dict.

    Args:
        variations: The task's own `VARIATIONS` preset table (name -> override dict).
        variation: A single preset name, or a list/tuple of names to combine. When
            combining, presets are merged in order via `dict.update` -- a key set by
            a later name overrides the same key set by an earlier one.
        variation_overrides: Optional ad-hoc overrides applied last, on top of the
            merged preset(s).

    Returns:
        (label, config) where `label` is a string identifying the resolved
        variation(s) -- the single name unchanged, or names joined with "+" for a
        combination -- and `config` is the merged override dict.
    """
    names = [variation] if isinstance(variation, str) else list(variation)
    if not names:
        raise ValueError("variation must be a non-empty string or a non-empty list of strings")

    unknown = [n for n in names if n not in variations]
    if unknown:
        raise ValueError(f"Unknown variation(s) {unknown}. Choose from {list(variations)}")

    config: dict[str, Any] = {}
    for name in names:
        config.update(variations[name])
    if variation_overrides:
        config.update(variation_overrides)

    label = "+".join(names)
    return label, config
