"""Config-driven robustness-eval perturbations of robosuite's TwoArmHandover task.

New sibling env only: robosuite's TwoArmHandover (capx/third_party/robosuite/robosuite/
environments/manipulation/two_arm_handover.py) and capx's RobosuiteHandoverEnv
(robosuite_handover.py) are both left untouched, so existing benchmark configs are
unaffected. Mirrors the pattern already validated for the single-arm Stack task
(StackVariants in robosuite_cubes_variants.py): a `variation` preset selects a named
combination of perturbations, applied on top of the vanilla task.

TwoArmHandover is bimanual (two Panda arms, `env_configuration="opposed"` by default in
capx), unlike Stack's single arm. See `TwoArmHandoverVariants._apply_robot_offset` for the
reasoning behind how "robot_position_shift" is interpreted for two robots.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, HammerObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variant_camera import (
    CAMERA_ORBIT_PRESETS,
    apply_camera_orbit,
    capture_camera_reference,
)
from capx.envs.simulators.variants_common import VariantMixin
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")

# Distractor object classes usable as a task-irrelevant addition to the scene. Restricted to
# cylinders (as in StackVariants): TwoArmHandover's default hammer placement sampler uses a
# *fixed* (non-random) rotation ("rotation=pi/2" about the y or z axis, see
# TwoArmHandover._load_model) applied to every object it samples, including any distractor
# added to the same sampler. A cylinder looks identical under that rotation; a box would look
# oddly tilted.
DISTRACTOR_OBJECT_CLASSES = {"cylinder": CylinderObject}

# "target_object_swap" analogues for the hammer. The hammer is a HammerObject
# (CompositeObject) whose handle/head geoms, `handle_length`, `head_halfsize`,
# `handle_geoms`, and `top_offset`/`bottom_offset` are all read directly by
# TwoArmHandover._get_task_info/_check_success/_setup_observables. Swapping in an unrelated
# object class would break the reward and success logic. Instead, each entry below is a dict
# of HammerObject constructor kwargs that changes its *shape/color* enough to look like a
# visually-distinct tool (e.g. a mallet: cylindrical handle, dark solid head) while remaining
# a HammerObject so every attribute the task logic relies on keeps working unmodified.
SWAP_CONFIGS: dict[str, dict[str, Any]] = {
    "mallet": dict(
        handle_shape="cylinder",
        handle_radius=(0.022, 0.026),
        handle_length=(0.22, 0.28),
        head_density_ratio=1.1,
        use_texture=False,
        rgba_handle=(0.55, 0.35, 0.15, 1.0),  # wood-brown cylindrical handle
        rgba_head=(0.12, 0.12, 0.12, 1.0),  # dark, rubber-mallet-like head
        rgba_face=(0.12, 0.12, 0.12, 1.0),
        rgba_claw=(0.12, 0.12, 0.12, 1.0),
    ),
}

CANONICAL_PROMPT = (
    "Arm 0 should pick up the hammer, lift it, and hand it over to Arm 1. "
    "Arm 1 should then grasp the hammer handle. Quaternions are WXYZ."
)
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Have Arm 0 grasp the hammer, lift it off the table, and pass it to Arm 1, "
    "which should take hold of the handle. Quaternions are WXYZ.",
    "Arm 0 picks up the hammer and hands it to Arm 1; Arm 1 should grip the handle "
    "once it receives the hammer. Quaternions are WXYZ.",
    "Transfer the hammer from Arm 0 to Arm 1: Arm 0 lifts it from the table and "
    "Arm 1 grasps the handle. Quaternions are WXYZ.",
]

# Named variation presets. Each maps to a dict of overrides consumed by TwoArmHandoverVariants.
# "baseline" reproduces the original TwoArmHandover task and is useful as a control condition
# when measuring policy robustness against the other ten perturbations.
# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt only ever says "the hammer" / "hammer handle" / "hammer head" -- no color word, so
# color_tint is unaffected. But target_object_swap (swap_target: "mallet", SWAP_CONFIGS
# above) is explicitly built to "look like a visually-distinct tool" -- exactly what makes
# "the hammer" stop matching what's actually in the scene.
VARIATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "color_shift_mild": {"color_tint": (1.0, 0.85, 0.6, 1.0)},
    "color_shift_strong": {"color_tint": (0.2, 0.9, 1.0, 1.0)},
    "object_position_shift": {"object_position_bias": {"bias": (0.6, 0.6), "scale": 0.5}},
    "robot_position_shift": {"robot_base_offset": (0.05, 0.05, 0.0)},
    "lighting_bright": {"light_diffuse_scale": 2.0, "light_ambient_scale": 2.5},
    "lighting_dim": {"light_diffuse_scale": 0.3, "light_ambient_scale": 0.3},
    "distractor_object": {"distractor_classes": ["cylinder"]},
    "target_object_swap": {"swap_target": "mallet"},
    "language_paraphrase": {"paraphrase_language": True},
    "compound_hard": {
        "color_tint": (1.0, 0.85, 0.6, 1.0),
        "light_diffuse_scale": 0.5,
        "light_ambient_scale": 0.5,
        "distractor_classes": ["cylinder"],
        "paraphrase_language": True,
    },
    # Difficulty knob, not a robustness perturbation: raises the handover height bar from the
    # upstream default (0.1) to 0.16 -- less margin for a sloppy or low handoff before Arm 1's
    # grasp stops counting as a success. See the __init__ override note above.
    "harder_handover": {"handover_height_threshold": 0.16},
}


VARIATIONS.update(CAMERA_ORBIT_PRESETS)


class TwoArmHandoverVariants(VariantMixin, TwoArmHandover):
    """TwoArmHandover with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint multiplier applied to the hammer's material(s)
          (wood handle material + metal head/face/claw material).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the hammer is sampled on the table, instead of the default
          centered range.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to BOTH robot
          bases -- see `_apply_robot_offset` for why both (not just one) are shifted.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES to additionally
          place on the table. Never scored and have no role in the handover.
        - swap_target (str): key of SWAP_CONFIGS; rebuilds the hammer with different
          shape/color HammerObject kwargs (see SWAP_CONFIGS docstring for why this is a
          parameter swap rather than an object-class swap).
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool
          of paraphrases instead of the fixed canonical wording.
        - handover_height_threshold (float): overrides upstream TwoArmHandover's
          height_threshold (default 0.1). A difficulty knob, not a perturbation -- see
          VARIATIONS["harder_handover"].

    Pick one of the ~10 named presets in `VARIATIONS` via `variation`, optionally layering
    ad-hoc tweaks via `variation_overrides`.
    """

    PRESETS = VARIATIONS
    OBS_CAMERA = None  # the capx wrapper pins agentview itself and applies the camera orbit there
    CANONICAL_PROMPT = CANONICAL_PROMPT
    PROMPT_PARAPHRASES = PROMPT_PARAPHRASES

    def __init__(
        self,
        variation: str | Sequence[str] = "baseline",
        variation_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._init_variant(variation, variation_overrides)

        super().__init__(**kwargs)

        # Difficulty knob, not a robustness perturbation: height_threshold is a plain instance
        # attribute TwoArmHandover.__init__ sets (not a constructor kwarg), read (not
        # recomputed) by both reward() and _check_success() -- overridable here with no method
        # override needed, unlike cube_lifting's lift_height_threshold which had to override
        # both methods since upstream Lift writes its margin as an inline literal instead of an
        # attribute. See VARIATIONS["harder_handover"].
        handover_threshold = self.variation_config.get("handover_height_threshold")
        if handover_threshold is not None:
            self.height_threshold = handover_threshold

    # ----------------------- Scene construction -----------------------

    def _load_model(self) -> None:
        # `_add_distractors` may have replaced self.placement_initializer with a
        # SequentialCompositeSampler on a previous call (hard_reset=True means _load_model runs
        # again on every env.reset()). TwoArmHandover._load_model's own
        # "if self.placement_initializer is not None" branch assumes a plain
        # ObjectPositionSampler -- its `.add_objects()` call raises AttributeError on a
        # composite sampler (see SequentialCompositeSampler.add_objects). Reset it to None here
        # so the hammer sampler is always rebuilt fresh.
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            self.placement_initializer = None

        swap_target = self.variation_config.get("swap_target")
        if swap_target is None:
            super()._load_model()
        else:
            self._load_model_with_swap(swap_target)

        # All mutations below operate on XML elements/samplers already attached to self.model /
        # self.placement_initializer -- the sim isn't compiled from this tree until later in the
        # environment's load/reset sequence, so it's safe to edit them here.
        self._apply_color_tint()
        self._apply_lighting()
        self._apply_robot_offset()
        # _add_distractors may replace self.placement_initializer with a composite sampler (see
        # its docstring), so it must run before _apply_object_position_bias, which needs to know
        # which sampler object (plain or composite sub-sampler) currently owns the hammer.
        self._add_distractors()
        self._apply_object_position_bias()

    def _load_model_with_swap(self, swap_target: str) -> None:
        """Reproduces TwoArmHandover._load_model, substituting the hammer's shape/color.

        TwoArmHandover builds the hammer inline in _load_model rather than via a separate
        overridable hook, so a shape/color swap has to re-derive this method instead of
        composing on top of it (same situation as Stack's cubeA, see StackVariants).
        """
        super(TwoArmHandover, self)._load_model()

        # Adjust base pose(s) accordingly (verbatim from TwoArmHandover._load_model; the
        # actual perturbation, if any, is layered on afterwards by _apply_robot_offset).
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                for robot, rotation, offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_true_size, table_friction=self.table_friction, table_offset=self.table_offset
        )
        mujoco_arena.set_origin([0, 0, 0])
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494],
            quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        )

        swap_kwargs = SWAP_CONFIGS[swap_target]
        self.hammer = HammerObject(name="hammer", rng=self.rng, **swap_kwargs)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hammer)
        else:
            rotation_axis = "y" if self.prehensile else "z"
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hammer,
                x_range=[-0.1, 0.1],
                y_range=[-0.05, 0.05],
                rotation=np.pi / 2.0,
                rotation_axis=rotation_axis,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hammer,
        )

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        materials = find_elements(root=self.hammer.asset, tags="material", return_first=False) or []
        for material in materials:
            material.set("rgba", rgba_str)

    def _apply_robot_offset(self) -> None:
        """Perturb the robot base position(s) -- see class docstring for the config key.

        Design choice for the two-robot case: the SAME (x, y, z) delta is added to BOTH
        robot bases (not a mirrored delta, and not just one arm). Reasoning:

          - TwoArmHandover needs the two arms' reachable workspaces to overlap enough for the
            hammer to actually pass from one gripper to the other. A delta applied to only one
            arm (or mirrored so the arms move apart/together) risks pushing them out of mutual
            reach, which would make the episode physically infeasible rather than "harder" --
            not a useful robustness signal.
          - Applying the same delta to both bases instead models a whole-workcell placement
            shift (e.g. the two-robot station is bolted down a few cm off from where a
            scene-camera/vision calibration expects it): it perturbs both robots' absolute
            poses (as seen by `agentview`) while preserving the relative arm-to-arm geometry
            the handover itself depends on.
          - Arm 0 is the primary/initiating manipulator (it must localize and grasp the hammer
            first -- see `_get_task_info`/`reward`), so its position matters most for grasp
            success; Arm 1 mainly needs to remain within handoff range of Arm 0. Since both
            still must stay mutually reachable, the offset is applied to both rather than only
            Arm 0.

        Re-derives the base-position formulas from TwoArmHandover._load_model (there is no
        accessor to read back the pose after `set_base_xpos`/`set_base_ori`, which only write
        into the XML and don't expose a getter) and adds `offset` on top before re-setting.
        """
        offset = self.variation_config.get("robot_base_offset")
        if offset is None:
            return
        offset = np.array(offset, dtype=float)

        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(np.array(xpos) + offset)
        elif self.env_configuration == "opposed":
            for robot, rotation, base_offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.25, 0.25)):
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                rot = np.array((0, 0, rotation))
                xpos = T.euler2mat(rot) @ np.array(xpos)
                xpos = xpos + np.array((0, base_offset, 0)) + offset
                robot.robot_model.set_base_xpos(xpos)
                robot.robot_model.set_base_ori(rot)
        else:  # "parallel"
            for robot, base_offset in zip(self.robots, (-0.6, 0.6)):
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                xpos = np.array(xpos) + np.array((0, base_offset, 0)) + offset
                robot.robot_model.set_base_xpos(xpos)

    # Default hammer sampling range from TwoArmHandover._load_model, used to scale/clip the
    # object_position_bias config below.
    _DEFAULT_X_RANGE = (-0.1, 0.1)
    _DEFAULT_Y_RANGE = (-0.05, 0.05)
    # Disjoint region used for distractors -- see _add_distractors for why this can't share the
    # hammer's own (very tight) drop-zone.
    _DISTRACTOR_X_RANGE = (-0.35, -0.25)
    _DISTRACTOR_Y_RANGE = (-0.1, 0.1)

    @property

    def _hammer_sampler(self) -> UniformRandomSampler:
        """The UniformRandomSampler that actually owns hammer placement.

        Usually `self.placement_initializer` itself, but `_add_distractors` may have replaced
        it with a `SequentialCompositeSampler` wrapping a `"HammerSampler"` sub-sampler.
        """
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            return self.placement_initializer.samplers["HammerSampler"]
        return self.placement_initializer

    def _apply_object_position_bias(self) -> None:
        pos_cfg = self.variation_config.get("object_position_bias")
        if pos_cfg is None:
            return
        sampler = self._hammer_sampler
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

    def _add_distractors(self) -> None:
        distractor_names = self.variation_config.get("distractor_classes", [])
        if not distractor_names:
            self.distractor_objects = []
            return

        self.distractor_objects = []
        for i, name in enumerate(distractor_names):
            obj_cls = DISTRACTOR_OBJECT_CLASSES[name]
            obj_name = f"Distractor{name}{i}"
            self.distractor_objects.append(obj_cls(name=obj_name, size=[0.02, 0.02]))

        self.model.merge_objects(self.distractor_objects)

        # TwoArmHandover's default hammer sampler covers a narrow, off-center drop-zone tuned
        # for exactly one object (x in [-0.1, 0.1], y in [-0.05, 0.05] -- the table itself is
        # only ~0.3m wide since TwoArmHandover shrinks it to a narrow bridge between the two
        # arms). Adding a second object into that same tight region makes
        # `ensure_valid_placement` unable to find a non-overlapping placement
        # (`RandomizationError: Cannot place all objects`). Instead, give the distractor(s) a
        # disjoint region of the table via a SequentialCompositeSampler -- same pattern used for
        # NutAssemblyVariants's distractor nut (robosuite_nut_assembly_variants.py).
        hammer_sampler = self._hammer_sampler
        composite = SequentialCompositeSampler(name="ObjectSampler")
        composite.append_sampler(
            sampler=UniformRandomSampler(
                name="HammerSampler",
                x_range=list(hammer_sampler.x_range),
                y_range=list(hammer_sampler.y_range),
                rotation=hammer_sampler.rotation,
                rotation_axis=hammer_sampler.rotation_axis,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rng=self.rng,
            )
        )
        composite.add_objects_to_sampler(sampler_name="HammerSampler", mujoco_objects=self.hammer)
        composite.append_sampler(
            sampler=UniformRandomSampler(
                name="DistractorSampler",
                x_range=list(self._DISTRACTOR_X_RANGE),
                y_range=list(self._DISTRACTOR_Y_RANGE),
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rng=self.rng,
            )
        )
        composite.add_objects_to_sampler(
            sampler_name="DistractorSampler", mujoco_objects=self.distractor_objects
        )
        self.placement_initializer = composite

    # ----------------------- Sim references / reset -----------------------


class RobosuiteHandoverVariantsEnv(RobosuiteBaseEnv):
    """RobosuiteHandoverEnv, backed by TwoArmHandoverVariants instead of plain TwoArmHandover.

    Identical control/observation interface to RobosuiteHandoverEnv (robosuite_handover.py);
    the only difference is the extra `variation`/`variation_overrides` kwargs forwarded to
    TwoArmHandoverVariants for robustness-eval perturbations. `variation="baseline"`
    reproduces the original TwoArmHandover task exactly.

    Bimanual quirks replicated verbatim from RobosuiteHandoverEnv (both robots need
    independent tracking that RobosuiteBaseEnv's single-arm helpers don't provide):
      - Two robots means two controller configs (`[controller_config, controller_config]`),
        two base links (`fixed_mount0_base`/`fixed_mount1_base`), two gripper links
        (`gripper0_right_eef`/`gripper1_right_eef`), and two independent gripper fraction
        targets (`_gripper_fraction_0`/`_gripper_fraction_1`).
      - The action vector is the concatenation of two 8-dim per-robot actions (7 joints + 1
        gripper each) rather than RobosuiteBaseEnv._build_action's single 7+1 action, so
        `move_to_joints_blocking*` build `np.concatenate([robot0_action, robot1_action])`
        directly instead of using `_build_action`/`_do_robosuite_step`.
    """

    _SUBSAMPLE_RATE = 1

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 5000,
        seed: int | None = None,
        viser_debug: bool = False,  # TODO: move the viser visualization manager into a separate class, low level env agnostic
        privileged: bool = False,
        enable_render: bool = False,
        variation: str | Sequence[str] = "baseline",
        variation_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.save_camera_name = "agentview"  # Scene-level camera to show both arms
        self.render_camera_names = ["agentview"]  # Scene-level camera for observations
        self.segmentation_level = "instance"

        self._render_width = 512
        self._render_height = 512

        # TwoArmHandover requires 2 robots or 1 bimanual robot
        # Load controller config for both robots (same config for both)
        controller_config = load_composite_controller_config(controller=self.controller_cfg)

        common_kwargs: dict[str, Any] = dict(
            # Forward the seed to robosuite so object placement and the robot's
            # joint-init noise are deterministic: robosuite's MujocoEnv.__init__
            # does self.rng = np.random.default_rng(seed) (environments/base.py),
            # and this env already hands that same self.rng to every placement
            # sampler. Without this line `seed` reached only RobosuiteBaseEnv's
            # own _rng and the scene was effectively unseeded.
            seed=seed,
            variation=variation,
            variation_overrides=variation_overrides,
            robots=["Panda", "Panda"],
            env_configuration="opposed",
            controller_configs=[controller_config, controller_config],
            horizon=max_steps,
            prehensile=True,
            reward_shaping=True,
            use_object_obs=True,
        )

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = TwoArmHandoverVariants(
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    **common_kwargs,
                )
            else:
                self.robosuite_env = TwoArmHandoverVariants(
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    camera_depths=True,
                    use_camera_obs=True,
                    **common_kwargs,
                )
        else:
            self.robosuite_env = TwoArmHandoverVariants(
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_segmentations=self.segmentation_level,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                camera_depths=True,
                use_camera_obs=True,
                **common_kwargs,
            )

        # Get camera ID and modify its position and orientation
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]
        self._apply_camera_orbit()

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._subsample_rate = 1

        # Robot link indices for transforms (robot0 and robot1)
        # Base links are fixed, so we cache their transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx_0 = self.robosuite_env.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx_0 = self.robosuite_env.sim.model.body_name2id("gripper0_right_eef")
        self.base_link_idx_1 = self.robosuite_env.sim.model.body_name2id("fixed_mount1_base")
        self.gripper_link_idx_1 = self.robosuite_env.sim.model.body_name2id("gripper1_right_eef")

        # Cache base transforms (these are constant, base doesn't move)
        self.base_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )
        self.base_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_1],
            ]
        )

        # Gripper state (read from robosuite when needed, not stored)
        self._gripper_fraction_0 = 1.0  # Target gripper state for robot0
        self._gripper_fraction_1 = 1.0  # Target gripper state for robot1

        # Temporary viser debugging
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.urdf_vis_arm0 = None
            self.urdf_vis_arm1 = None
            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.mjcf_ee_frame_handle_arm1 = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.handle_frame_handle = None
            self.grasp_pose_handle_arm0 = None
            self.grasp_pose_handle_arm1 = None
            self.gripper_metric_length = 0.0584

            # Load URDF for both robots
            self.urdf = load_robot_description("panda_description")
            # Create URDF visualization for arm0 first (load immediately)
            # Arm1 will be loaded lazily on first update to avoid blocking initial page load
            print("Loading robot0 URDF visualization (this may take a moment)...")
            self.urdf_vis_arm0 = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
            print("Robot0 URDF loaded. Viser server ready - you can now open the browser.")
            self._viser_init_check()

            # Initialize grasp pose storage (set by API when grasp is sampled)
            self.grasp_sample_tf_arm0 = None
            self.grasp_sample_tf_arm1 = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()

        # Re-apply camera adjustment after reset (in case model was reloaded)
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]
        self._apply_camera_orbit()

        self._step_count = 0
        self._sim_step_count = 0

        obs = self.get_observation()
        info = {"task_prompt": self.robosuite_env.get_ep_meta()["lang"]}
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        # This is a fallback; normally FrankaControlApi methods are used
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move robot0 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            if self._sim_step_count >= self.max_steps:
                break
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            # Build Robosuite action for both robots
            # Each robot action = [7 joints, 1 gripper] = 8 dims
            robot0_action = np.concatenate([target, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self.robosuite_env.step(action)
            else:
                self.robosuite_env.step(action, skip_render_images=True)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def move_to_joints_blocking_arm1(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move robot1 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            if self._sim_step_count >= self.max_steps:
                break
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)
            robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            # Build Robosuite action for both robots
            robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([target, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self.robosuite_env.step(action)
            else:
                self.robosuite_env.step(action, skip_render_images=True)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def move_to_joints_blocking_both(
        self, joints0: np.ndarray, joints1: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move both robot0 and robot1 to target joint positions simultaneously using Robosuite's controller.

        Args:
            joints0: (7,) target joint positions for robot0 in radians
            joints1: (7,) target joint positions for robot1 in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target0 = np.asarray(joints0, dtype=np.float64).reshape(7)
        target1 = np.asarray(joints1, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            if self._sim_step_count >= self.max_steps:
                break
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current0 = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            current1 = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence for both arms
            error0 = np.linalg.norm(current0 - target0)
            error1 = np.linalg.norm(current1 - target1)
            if error0 < tolerance and error1 < tolerance:
                break

            # Build Robosuite action for both robots
            # Each robot action = [7 joints, 1 gripper] = 8 dims
            robot0_action = np.concatenate([target0, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([target1, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self.robosuite_env.step(action)
            else:
                self.robosuite_env.step(action, skip_render_images=True)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def _set_gripper(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot0.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_0 = float(np.clip(fraction, 0.0, 1.0))

    def _set_gripper_arm1(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot1.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_1 = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """Execute one simulation step maintaining current joint positions and gripper states."""
        # Get current joint positions from robosuite
        robosuite_obs = self.robosuite_env._get_observations()
        robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

        # Build action for both robots (maintain current joints, apply gripper targets)
        robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
        robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
        action = np.concatenate([robot0_action, robot1_action])

        self.robosuite_env.step(action)
        self._sim_step_count += 1

        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def _hammer_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get hammer poses in robot base frame.

        Returns:
            Dict with "hammer_pos" and "handle_pos" keys, each containing
            [x, y, z, qw, qx, qy, qz] (7-element list) for hammer body and handle position
        """
        # Get transforms
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )

        # Get hammer position and quaternion (convert from xyzw to wxyz)
        hammer_pos_world = robosuite_obs.get("hammer_pos", np.zeros(3))
        hammer_quat_xyzw = robosuite_obs.get("hammer_quat", np.array([0, 0, 0, 1]))
        # Convert quaternion from xyzw to wxyz
        hammer_quat_wxyz = np.array(
            [hammer_quat_xyzw[3], hammer_quat_xyzw[0], hammer_quat_xyzw[1], hammer_quat_xyzw[2]]
        )

        # Get handle position
        handle_pos_world = robosuite_obs.get("handle_xpos", np.zeros(3))

        # Transform hammer to robot base frame
        hammer_world = vtf.SE3(wxyz_xyz=np.concatenate([hammer_quat_wxyz, hammer_pos_world]))
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        hammer_robot_base = base_transform @ hammer_world

        # Transform handle position (use same rotation as hammer)
        handle_world = vtf.SE3.from_translation(handle_pos_world)
        handle_robot_base = base_transform @ handle_world

        return {
            "hammer": [
                float(x)
                for x in np.concatenate(
                    [hammer_robot_base.translation(), hammer_robot_base.rotation().wxyz]
                )
            ],
            "handle": [
                float(x)
                for x in np.concatenate(
                    [
                        handle_robot_base.translation(),
                        hammer_robot_base.rotation().wxyz,
                    ]  # Use hammer rotation for handle
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse handover reward.

        Returns:
            1.0 if handover is successful (only Arm 1 gripping handle, lifted above threshold), 0.0 otherwise
        """
        # Use the robosuite environment's built-in reward function
        # It returns 2.0 for success when reward_shaping=False, normalized by reward_scale/2.0
        # So max reward is 1.0 (when reward_scale=1.0)
        reward = float(self.robosuite_env.reward())

        # The robosuite reward is already normalized to [0, 1.0] for success
        return reward

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        # Ensure camera position is set before capturing observations
        # (robosuite's _get_observations() uses the current camera position)
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]
        self._apply_camera_orbit()
        # Forward the simulation to update camera transforms before capturing observations
        self.robosuite_env.sim.forward()
        # Force update to ensure fresh camera images are captured with the new camera position
        robosuite_obs = self.robosuite_env._get_observations(force_update=True)
        pose_dict = self._hammer_pose_dict(robosuite_obs)

        # Store hammer poses explicitly for hammer-specific API
        hammer_pose_array = np.stack(
            [
                np.asarray(pose_dict["hammer"], dtype=np.float32),
                np.asarray(pose_dict["handle"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["hammer_poses"] = {
            "hammer": hammer_pose_array[0],
            "handle": hammer_pose_array[1],
        }

        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(
                        self.robosuite_env.sim.data.get_camera_xmat(camera_name)
                    ).wxyz,
                    self.robosuite_env.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )

            robosuite_obs[camera_name]["pose"] = np.concatenate(
                [
                    cam_robot_tf.translation(),
                    cam_robot_tf.rotation().wxyz,
                ]
            )
            robosuite_obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs[camera_name]["intrinsics"] = K

            robosuite_obs[camera_name]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs[camera_name]["images"]["rgb"] = robosuite_obs[camera_name + "_image"][
                    ::-1
                ]
            if camera_name + "_depth" in robosuite_obs:
                # converts openGL z buffer to metric
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )

                robosuite_obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in robosuite_obs:
                robosuite_obs[camera_name]["images"]["segmentation"] = robosuite_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

        # Compute gripper pose for robot0
        gripper_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_0],
            ]
        )
        gripper_robot_base_0 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_0)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Compute gripper pose for robot1 in robot0 frame
        gripper_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_1],
            ]
        )

        # First compute in robot1's base frame
        gripper_robot_base_1_local = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_1)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Transform from robot1's base frame to world frame
        gripper_world_1 = vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1) @ gripper_robot_base_1_local

        # Transform from world frame to robot0's base frame
        gripper_robot_base_1 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse() @ gripper_world_1
        )

        robosuite_obs["robot0_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_0.translation(),
                gripper_robot_base_0.rotation().wxyz,
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

        # Now robot1_cartesian_pos is in robot0's base frame
        robosuite_obs["robot1_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_1.translation(),
                gripper_robot_base_1.rotation().wxyz,
                [robosuite_obs["robot1_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

        if len(self.render_camera_names) == 1:
            robosuite_obs["robot0_robotview"] = robosuite_obs[self.render_camera_names[0]]

        return robosuite_obs

    # ------------------------- Video Capture -------------------------

    def _apply_camera_orbit(self) -> None:
        # This wrapper pins agentview to a fixed pose (above) after every reset/observation,
        # so the orbit has to be applied here, on top of that pinned pose, not inside the
        # robosuite env's _reset_internal where it would be overwritten.
        sim = self.robosuite_env.sim
        reference = capture_camera_reference(sim, "agentview")
        apply_camera_orbit(sim, reference, self.robosuite_env.variation_config.get("camera_orbit"))

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            # Ensure camera position is set and capture first frame with correct position
            agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
            self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
            self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]
            self._apply_camera_orbit()
            self.robosuite_env.sim.forward()
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def _record_frame(self) -> None:
        if not self._record_frames:
            return

        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._frame_buffer.append(frame[::-1])  # Flip vertically

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    # Temporary viser debugging

    def _update_viser_server(
        self,
    ) -> None:
        try:
            obs = self.get_observation()
            if self.viser_server is not None:
                self._viser_init_check()

                if "robot0_joint_pos" in obs:
                    if self.urdf_vis_arm0 is None:
                        print("Warning: urdf_vis_arm0 is None, reinitializing...")
                        self.urdf_vis_arm0 = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
                    robot0_joints = obs["robot0_joint_pos"].copy()
                    if "robot0_gripper_qpos" in obs:
                        gripper_pos = obs["robot0_gripper_qpos"][0] / self.gripper_metric_length
                        robot0_joints_full = np.concatenate([robot0_joints, [gripper_pos]])
                    else:
                        robot0_joints_full = np.concatenate([robot0_joints, [0.0]])
                    try:
                        self.urdf_vis_arm0.update_cfg(robot0_joints_full)
                    except Exception as e:
                        print(f"Warning: Failed to update robot0 URDF: {e}")

                if "robot1_joint_pos" in obs:
                    if self.urdf_vis_arm1 is None:
                        print("Loading robot1 URDF visualization (this may take a moment)...")
                        self.urdf_vis_arm1 = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
                        print("Robot1 URDF loaded.")
                    robot1_joints = obs["robot1_joint_pos"].copy()
                    if "robot1_gripper_qpos" in obs:
                        gripper_pos = obs["robot1_gripper_qpos"][0] / self.gripper_metric_length
                        robot1_joints_full = np.concatenate([robot1_joints, [gripper_pos]])
                    else:
                        robot1_joints_full = np.concatenate([robot1_joints, [0.0]])
                    try:
                        self.urdf_vis_arm1.update_cfg(robot1_joints_full)
                    except Exception as e:
                        print(f"Warning: Failed to update robot1 URDF: {e}")

                obs_cartesian_arm0 = obs["robot0_cartesian_pos"][:-1]
                self.mjcf_ee_frame_handle.position = obs_cartesian_arm0[:3]
                self.mjcf_ee_frame_handle.wxyz = obs_cartesian_arm0[3:]

                obs_cartesian_arm1 = obs["robot1_cartesian_pos"][:-1]
                self.mjcf_ee_frame_handle_arm1.position = obs_cartesian_arm1[:3]
                self.mjcf_ee_frame_handle_arm1.wxyz = obs_cartesian_arm1[3:]

                if hasattr(self, "grasp_sample_tf_arm0") and self.grasp_sample_tf_arm0 is not None:
                    if self.grasp_pose_handle_arm0 is None:
                        self.grasp_pose_handle_arm0 = self.viser_server.scene.add_frame(
                            "/grasp_pose_arm0",
                            position=self.grasp_sample_tf_arm0.wxyz_xyz[-3:],
                            wxyz=self.grasp_sample_tf_arm0.wxyz_xyz[:4],
                            axes_length=0.15,
                            axes_radius=0.005,
                        )
                    else:
                        self.grasp_pose_handle_arm0.position = self.grasp_sample_tf_arm0.wxyz_xyz[-3:]
                        self.grasp_pose_handle_arm0.wxyz = self.grasp_sample_tf_arm0.wxyz_xyz[:4]
                        self.grasp_pose_handle_arm0.visible = True
                elif self.grasp_pose_handle_arm0 is not None:
                    self.grasp_pose_handle_arm0.visible = False

                if hasattr(self, "grasp_sample_tf_arm1") and self.grasp_sample_tf_arm1 is not None:
                    if self.grasp_pose_handle_arm1 is None:
                        self.grasp_pose_handle_arm1 = self.viser_server.scene.add_frame(
                            "/grasp_pose_arm1",
                            position=self.grasp_sample_tf_arm1.wxyz_xyz[-3:],
                            wxyz=self.grasp_sample_tf_arm1.wxyz_xyz[:4],
                            axes_length=0.15,
                            axes_radius=0.005,
                        )
                    else:
                        self.grasp_pose_handle_arm1.position = self.grasp_sample_tf_arm1.wxyz_xyz[-3:]
                        self.grasp_pose_handle_arm1.wxyz = self.grasp_sample_tf_arm1.wxyz_xyz[:4]
                        self.grasp_pose_handle_arm1.visible = True
                elif self.grasp_pose_handle_arm1 is not None:
                    self.grasp_pose_handle_arm1.visible = False

                if "hammer_poses" in obs and "handle" in obs["hammer_poses"]:
                    handle_pose = obs["hammer_poses"]["handle"]
                    self.handle_frame_handle.position = handle_pose[:3]
                    self.handle_frame_handle.wxyz = handle_pose[3:]

                rbg_imgs = obs_get_rgb(obs)
                for image_key in rbg_imgs:
                    self.viser_img_handle.image = rbg_imgs[image_key]

                    if "pose" in obs[image_key]:
                        self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                        self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                        self.image_frustum_handle.image = rbg_imgs[image_key]
                    else:
                        self.image_frustum_handle.visible = False

                camera_key = "robot0_robotview" if "robot0_robotview" in obs else "agentview"
                if camera_key in obs and "images" in obs[camera_key] and "depth" in obs[camera_key]["images"]:
                    points, colors = depth_color_to_pointcloud(
                        obs[camera_key]["images"]["depth"][:, :, 0],
                        rbg_imgs.get(camera_key, rbg_imgs.get("agentview", None)),
                        obs[camera_key]["intrinsics"],
                    )
                    if points is not None and colors is not None:
                        if "pose_mat" in obs[camera_key]:
                            points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
                            points_world = (obs[camera_key]["pose_mat"] @ points_hom.T).T[:, :3]
                        else:
                            points_world = points

                        self.viser_server.scene.add_point_cloud(
                            f"{camera_key}/point_cloud",
                            points_world,
                            colors,
                            point_size=0.001,
                            point_shape="square",
                        )
        except Exception as e:
            import traceback
            print(f"Warning: Viser update failed: {e}")
            traceback.print_exc()

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        # Initialize robot0 (arm0) end-effector frame
        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/robot0_ee_frame", axes_length=0.15, axes_radius=0.005
            )

        # Initialize robot1 (arm1) end-effector frame
        if self.mjcf_ee_frame_handle_arm1 is None:
            self.mjcf_ee_frame_handle_arm1 = self.viser_server.scene.add_frame(
                "/robot1_ee_frame", axes_length=0.15, axes_radius=0.005
            )

        # Note: robot1 URDF is loaded lazily in _update_viser_server to avoid blocking initial page load

        # Initialize handle frame (useful for grasping visualization)
        if self.handle_frame_handle is None:
            self.handle_frame_handle = self.viser_server.scene.add_frame(
                "/handle_frame", axes_length=0.08, axes_radius=0.002
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="agentview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = ["TwoArmHandoverVariants", "RobosuiteHandoverVariantsEnv", "VARIATIONS"]
