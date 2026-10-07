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

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, find_elements

from capx.envs.simulators.variant_camera import apply_camera_orbit, capture_camera_reference

# Each task's difficulty-knob preset (raises the task's own success bar; not a perturbation).
# Sweep scripts' hard_vs_base mode reads this table.
DIFFICULTY_PRESETS: dict[str, str] = {
    "cube_lifting": "lift_higher",
    "cube_stack": "stack_3_cubes",
    "cube_restack": "stack_3_cubes",
    "nut_assembly": "both_nuts",
    "spill_wipe": "harder_spill",
    "two_arm_handover": "harder_handover",
    "two_arm_lift": "harder_lift",
}

# Each task's reduced low-level API class (the S3 tier the hard-coded heuristics/ solvers use).
S3_API_CLASSES: dict[str, str] = {
    "spill_wipe": "FrankaControlSpillWipeApiReduced",
    "two_arm_handover": "FrankaHandoverApiReduced",
    "two_arm_lift": "FrankaControlApiReducedBimanual",
}
S3_API_DEFAULT = "FrankaControlApiReduced"


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


class VariantMixin:
    """The perturbation plumbing every `*Variants` robosuite env shares.

    Mix in before the robosuite task class (``class LiftVariants(VariantMixin, Lift)``) and
    set the class attributes below. The mixin owns preset resolution, the perturbations whose
    implementation is the same for every task (lighting, single-arm robot base offset,
    placement-range bias), the observation-camera orbit, distractor body ids, and the
    paraphrased instruction. Task-specific perturbations (colour tint, object swaps,
    distractor construction, two-arm robot offsets) stay in each env, which may also override
    any hook here.

    Class attributes:
        PRESETS: the module's VARIATIONS table.
        OBS_CAMERA: name of the camera the capx wrapper observes through, perturbed by the
            camera_orbit presets at every reset; None if the wrapper handles the camera itself.
        CANONICAL_PROMPT / PROMPT_PARAPHRASES: instruction text and its paraphrase pool.
    """

    PRESETS: dict[str, dict[str, Any]] = {}
    OBS_CAMERA: str | None = "robot0_robotview"
    CANONICAL_PROMPT: str = ""
    PROMPT_PARAPHRASES: Sequence[str] = ()

    def _init_variant(self, variation: str | Sequence[str], variation_overrides: dict[str, Any] | None) -> None:
        """Call at the top of __init__, before the robosuite task's own __init__ runs _load_model."""
        self.variation, self.variation_config = resolve_variation_config(self.PRESETS, variation, variation_overrides)
        self.distractor_objects: list[Any] = []
        self.language_instruction: str | None = None

    # ----------------------- Shared perturbations (called from _load_model) -----------------------

    def _apply_lighting(self) -> None:
        diffuse_scale = self.variation_config.get("light_diffuse_scale")
        ambient_scale = self.variation_config.get("light_ambient_scale")
        if diffuse_scale is None and ambient_scale is None:
            return
        base_diffuse = np.array([0.7, 0.7, 0.7])
        base_ambient = np.array([0.15, 0.15, 0.15])
        lights = find_elements(root=self.model.mujoco_arena.worldbody, tags="light", return_first=False) or []
        for light in lights:
            if diffuse_scale is not None:
                light.set("diffuse", array_to_string(np.clip(base_diffuse * diffuse_scale, 0, 1)))
            if ambient_scale is not None:
                light.set("ambient", array_to_string(np.clip(base_ambient * ambient_scale, 0, 1)))

    def _apply_robot_offset(self) -> None:
        """Single-arm table tasks: shift the robot base by robot_base_offset. Two-arm envs override."""
        offset = self.variation_config.get("robot_base_offset")
        if offset is None:
            return
        base_pos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        self.robots[0].robot_model.set_base_xpos(base_pos + np.array(offset))

    def _apply_default_placement_range(self) -> None:
        self.placement_initializer.x_range = list(self._DEFAULT_X_RANGE)
        self.placement_initializer.y_range = list(self._DEFAULT_Y_RANGE)

    def _apply_object_position_bias(self) -> None:
        """Re-centre/shrink a UniformRandomSampler's x/y ranges inside the task's default bin.
        Envs whose sampler isn't a plain range (restack, nut_assembly, spill_wipe, handover) override."""
        pos_cfg = self.variation_config.get("object_position_bias")
        if pos_cfg is None:
            return
        sampler = self.placement_initializer
        bin_x_half, bin_y_half = -self._DEFAULT_X_RANGE[0], -self._DEFAULT_Y_RANGE[0]
        bias_x, bias_y = pos_cfg.get("bias", (0.0, 0.0))
        scale = pos_cfg.get("scale", 1.0)
        center_x, center_y = bias_x * bin_x_half, bias_y * bin_y_half
        half_x, half_y = bin_x_half * scale, bin_y_half * scale
        sampler.x_range = [
            float(np.clip(center_x - half_x, -bin_x_half, bin_x_half)),
            float(np.clip(center_x + half_x, -bin_x_half, bin_x_half)),
        ]
        sampler.y_range = [
            float(np.clip(center_y - half_y, -bin_y_half, bin_y_half)),
            float(np.clip(center_y + half_y, -bin_y_half, bin_y_half)),
        ]

    # ----------------------- Sim references / reset -----------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        self._distractor_body_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self.distractor_objects]
        if self.OBS_CAMERA is not None:
            self._camera_orbit_ref = capture_camera_reference(self.sim, self.OBS_CAMERA)

    def _reset_internal(self) -> None:
        super()._reset_internal()
        if self.OBS_CAMERA is not None:
            apply_camera_orbit(self.sim, self._camera_orbit_ref, self.variation_config.get("camera_orbit"))
        self._reset_variant_extras()
        self.language_instruction = self._sample_language()

    def _reset_variant_extras(self) -> None:
        """Hook for per-reset work (e.g. placing distractors) between the camera orbit and the prompt."""

    def _sample_language(self) -> str:
        if self.variation_config.get("paraphrase_language", False):
            return str(self.rng.choice(self.PROMPT_PARAPHRASES))
        return self.CANONICAL_PROMPT

    def get_ep_meta(self) -> dict[str, Any]:
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.language_instruction
        return ep_meta
