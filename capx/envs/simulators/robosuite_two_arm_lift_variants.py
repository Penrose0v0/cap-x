"""Config-driven robustness-eval perturbations of robosuite's bimanual TwoArmLift task.

New sibling env only: robosuite's TwoArmLift (capx/third_party/robosuite/robosuite/
environments/manipulation/two_arm_lift.py) and capx's RobosuiteTwoArmLiftEnv
(robosuite_two_arm_lift.py) are both left untouched, so existing benchmark configs are
unaffected. Mirrors the pattern already validated for robosuite's Stack task
(StackVariants, robosuite_cubes_variants.py): a `variation` preset selects a named
combination of perturbations, applied on top of the vanilla task.

Two things differ from the single-arm Stack case, both driven by TwoArmLift's specifics:

  - The lifted object is PotWithHandlesObject, a CompositeObject whose two named sites
    ("handle0"/"handle1") and geom groups (handle0_geoms/handle1_geoms) are load-bearing:
    TwoArmLift._setup_references, reward(), and _check_success() all key off of them by
    name/role. Swapping to an unrelated object class (as StackVariants does for cubeA,
    e.g. box -> cylinder) would have no such handles and would break grasping, reward
    shaping, and success detection outright. So "target_object_swap" here instead swaps
    PotWithHandlesObject's *construction kwargs* for a meaningfully different pot (smaller
    body, thicker solid handles instead of hollow wire handles, plain untextured grey
    instead of the textured red/green/blue) -- a different-but-same-role object, which is
    the closest feasible analogue given the composite/site-dependent object.
  - "robot_position_shift" perturbs both arms. Under env_configuration="opposed" (the only
    configuration capx's wrapper uses), TwoArmLift._load_model positions each robot by
    computing its default table-relative base position and then rotating it by +-90 deg
    around z so the two arms face each other. We add the configured offset in each robot's
    own pre-rotation ("facing the table") frame, then apply the same per-robot rotation --
    i.e. an identical local-frame offset for both arms rather than distinct per-arm
    offsets. This keeps the perturbation well-defined and symmetric under the mirrored
    "opposed" layout, similar to a single consistent rig/calibration error affecting both
    arm mounts equally, rather than arbitrarily privileging one arm over the other.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import viser
import viser.transforms as vtf
import robosuite.utils.transform_utils as T
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, PotWithHandlesObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variant_camera import CAMERA_ORBIT_PRESETS
from capx.envs.simulators.variants_common import VariantMixin
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

# Object classes usable as task-irrelevant distractors placed alongside the pot.
DISTRACTOR_OBJECT_CLASSES = {"cylinder": CylinderObject}

# PotWithHandlesObject construction-kwarg overrides usable as a "different but same-role"
# stand-in for the default pot (see module docstring for why a literal class swap isn't
# feasible here). Smaller body, shorter/thicker solid handles (vs. the default hollow wire
# handles), plain untextured grey instead of the textured red body / green+blue handles.
POT_CONFIG_SWAPS: dict[str, dict[str, Any]] = {
    "compact_solid_handle": dict(
        body_half_size=(0.05, 0.05, 0.05),
        handle_radius=0.016,
        handle_length=0.06,
        handle_width=0.06,
        solid_handle=True,
        use_texture=False,
        rgba_body=(0.5, 0.5, 0.5, 1.0),
        rgba_handle_0=(0.5, 0.5, 0.5, 1.0),
        rgba_handle_1=(0.5, 0.5, 0.5, 1.0),
    ),
}

CANONICAL_PROMPT = (
    "Two robotic arms (Panda) should coordinate to lift the pot. Arm 0 should grasp one "
    "handle, and Arm 1 should grasp the other handle."
)
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Coordinate both Panda arms to lift the pot together: one arm grasps each handle.",
    "Using both arms, grasp the pot's two handles and lift it jointly off the table.",
    "Have arm 0 take one pot handle and arm 1 take the other, then lift the pot as a team.",
]

# Named variation presets. Each maps to a dict of overrides consumed by TwoArmLiftVariants.
# "baseline" reproduces the original TwoArmLift task and is useful as a control condition
# when measuring policy robustness against the other nine perturbations.
# Unlike the other 6 task files, this one is NOT affected by the prompt/scene conflict
# documented in variants_common.py's module docstring: the prompt only says "grasp one
# handle... grasp the other handle" (no color or handle-shape word), and
# target_object_swap (POT_CONFIG_SWAPS) only changes material/thickness -- both were
# deliberately written this way (see the module docstring above) because the pot's named
# sites/geoms are load-bearing for reward/success detection, so an unrelated-class swap
# wasn't an option here anyway.
VARIATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "color_shift_mild": {"color_tint": (1.0, 0.85, 0.85, 1.0)},
    "color_shift_strong": {"color_tint": (0.2, 1.0, 0.3, 1.0)},
    "object_position_shift": {"object_position_bias": {"bias": (0.6, 0.6), "scale": 0.5}},
    "robot_position_shift": {"robot_base_offset": (0.06, 0.06, 0.0)},
    "lighting_bright": {"light_diffuse_scale": 2.0, "light_ambient_scale": 2.5},
    "lighting_dim": {"light_diffuse_scale": 0.3, "light_ambient_scale": 0.3},
    "distractor_object": {"distractor_classes": ["cylinder"]},
    "target_object_swap": {"swap_pot_config": "compact_solid_handle"},
    "language_paraphrase": {"paraphrase_language": True},
    "compound_hard": {
        "color_tint": (1.0, 0.85, 0.85, 1.0),
        "light_diffuse_scale": 0.5,
        "light_ambient_scale": 0.5,
        "distractor_classes": ["cylinder"],
        "paraphrase_language": True,
    },
    # Difficulty knob, not a robustness perturbation: unlike the other 6 tasks' difficulty
    # knobs, this one needed a full reward()/_check_success() override -- both the 0.10m
    # height margin and the 30-degree tilt tolerance are inline literals in the method bodies
    # upstream, not even instance attributes (see the overrides above). Raises the lift height
    # to 0.18m and tightens the tilt tolerance to 15 degrees -- both arms have to lift higher
    # and keep the pot noticeably more level.
    "harder_lift": {"lift_height_threshold": 0.18, "tilt_tolerance_deg": 15.0},
}


VARIATIONS.update(CAMERA_ORBIT_PRESETS)


class TwoArmLiftVariants(VariantMixin, TwoArmLift):
    """TwoArmLift with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint multiplier applied to the pot's material(s).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the pot is sampled on the table, instead of TwoArmLift's own
          default +-0.03 m centered range.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied, in each robot's
          own pre-rotation frame, to both robots' base positions (see module docstring).
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the
          arena light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES to additionally
          place on the table. Never scored and have no grasping role.
        - swap_pot_config (str): key of POT_CONFIG_SWAPS; overrides PotWithHandlesObject's
          construction kwargs for a differently-sized/colored/handled pot while keeping its
          handle0/handle1 sites and geoms intact (see module docstring for why a literal
          class swap isn't used here).
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool
          of paraphrases instead of the fixed canonical wording.
        - lift_height_threshold (float) / tilt_tolerance_deg (float): override upstream
          TwoArmLift's hardcoded 0.10m success margin and 30-degree tilt tolerance. Difficulty
          knobs, not perturbations -- see `reward()`/`_check_success()` overrides below and
          VARIATIONS["harder_lift"].

    Pick one of the ~10 named presets in `VARIATIONS` via `variation`, optionally layering
    ad-hoc tweaks via `variation_overrides`.
    """

    PRESETS = VARIATIONS
    # The observation image is the one robosuite renders during reset(), from the arena's own
    # agentview pose (the capx wrapper later pins agentview elsewhere for subsequent frames, but
    # without a forward, so the reset observation and its extrinsics both come from the arena
    # pose). Perturb that camera, before the reset render, so image and extrinsics stay consistent.
    OBS_CAMERA = "agentview"
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

    # ----------------------- Scene construction -----------------------

    def _load_model(self) -> None:
        # With hard_reset (the default), _load_model() runs again on every env.reset(). Our
        # own composite wrapping from a previous call (added below in _add_distractors, only
        # when distractor_classes is set) isn't a plain sampler that _load_model_base knows
        # how to reuse/re-add the pot to (SequentialCompositeSampler.add_objects() raises by
        # design) -- discard it here so pot (re)construction always starts from a clean
        # slate; _add_distractors rebuilds a fresh composite every call anyway.
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            self.placement_initializer = None

        swap_key = self.variation_config.get("swap_pot_config")
        pot_overrides = POT_CONFIG_SWAPS[swap_key] if swap_key else None
        self._load_model_base(pot_overrides=pot_overrides)

        # All mutations below operate on XML elements/samplers already attached to self.model /
        # self.placement_initializer -- the sim isn't compiled from this tree until later in the
        # environment's load/reset sequence, so it's safe to edit them here.
        self._apply_default_placement_range()
        self._apply_color_tint()
        self._apply_lighting()
        self._apply_robot_offset()
        self._apply_object_position_bias()
        self._add_distractors()

    # Matches TwoArmLift's own default sampler range ([-0.03, 0.03] on both axes). Applied
    # unconditionally so variation="baseline" reproduces the vanilla task's placement exactly.
    _DEFAULT_X_RANGE = (-0.03, 0.03)
    _DEFAULT_Y_RANGE = (-0.03, 0.03)

    def _load_model_base(self, pot_overrides: dict[str, Any] | None = None) -> None:
        """Reproduces TwoArmLift._load_model, optionally overriding the pot's construction kwargs.

        TwoArmLift builds the pot inline in _load_model rather than via a separate
        overridable hook (unlike PickPlace's _construct_objects), so a construction-kwarg
        swap has to re-derive this method instead of composing on top of it -- same
        situation StackVariants hit with Stack's cubeA/cubeB.
        """
        super(TwoArmLift, self)._load_model()

        # Adjust base pose(s) accordingly (verbatim from TwoArmLift._load_model)
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest, optionally with swapped construction kwargs
        pot_kwargs: dict[str, Any] = dict(pot_overrides) if pot_overrides else {}
        self.pot = PotWithHandlesObject(name="pot", **pot_kwargs)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.pot,
        )

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        materials = find_elements(root=self.pot.asset, tags="material", return_first=False) or []
        for material in materials:
            material.set("rgba", rgba_str)

    def _apply_robot_offset(self) -> None:
        offset = self.variation_config.get("robot_base_offset")
        if offset is None:
            return
        offset = np.array(offset)
        if self.env_configuration == "single-robot":
            base_pos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
            self.robots[0].robot_model.set_base_xpos(base_pos + offset)
        elif self.env_configuration == "opposed":
            # Apply the same local-frame offset to both arms, then rotate -- see module
            # docstring for why this (rather than distinct per-arm offsets) was chosen.
            for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                base_pos = np.array(robot.robot_model.base_xpos_offset["table"](self.table_full_size[0]))
                rot = np.array((0, 0, rotation))
                xpos = T.euler2mat(rot) @ (base_pos + offset)
                robot.robot_model.set_base_xpos(xpos)
        else:  # "parallel" configuration setting
            for robot, lateral in zip(self.robots, (-0.25, 0.25)):
                base_pos = np.array(robot.robot_model.base_xpos_offset["table"](self.table_full_size[0]))
                base_pos = base_pos + np.array((0, lateral, 0)) + offset
                robot.robot_model.set_base_xpos(base_pos)

    def _add_distractors(self) -> None:
        distractor_names = self.variation_config.get("distractor_classes", [])
        if not distractor_names:
            self.distractor_objects = []
            return

        self.distractor_objects = []
        for i, name in enumerate(distractor_names):
            obj_cls = DISTRACTOR_OBJECT_CLASSES[name]
            obj_name = f"Distractor{name}{i}"
            if obj_cls is CylinderObject:
                self.distractor_objects.append(obj_cls(name=obj_name, size=[0.02, 0.02]))
            else:
                self.distractor_objects.append(obj_cls(name=obj_name))

        self.model.merge_objects(self.distractor_objects)

        # The pot's own sampling region (+-0.03 m on both axes by default, possibly re-biased
        # by object_position_bias) is far too small to also fit a second object: the pot's
        # horizontal_radius alone (~0.23 m, from its body plus handles) already exceeds the
        # entire region, so ensure_valid_placement could never find a non-overlapping spot
        # for both objects sharing that one sampler. So distractors get their own sub-sampler
        # over a separate patch of the table, well clear of the pot's region, and the two
        # samplers are combined via SequentialCompositeSampler (the pot is still sampled
        # first, so distractor placement is checked against the pot's actual sampled pose).
        pot_sampler = self.placement_initializer
        distractor_sampler = UniformRandomSampler(
            name="DistractorSampler",
            mujoco_objects=self.distractor_objects,
            x_range=[0.28, 0.34],
            y_range=[0.28, 0.34],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )
        composite = SequentialCompositeSampler(name="TwoArmLiftVariantsSampler")
        composite.append_sampler(pot_sampler)
        composite.append_sampler(distractor_sampler)
        self.placement_initializer = composite

    # ----------------------- Sim references / reset -----------------------

    # ----------------------- Difficulty override -----------------------

    def _check_success(self) -> bool:
        """Verbatim from TwoArmLift._check_success, parameterized on lift_height_threshold
        instead of the hardcoded 0.10m margin (see VARIATIONS["harder_lift"]). Upstream's own
        success check has no tilt component (only reward() does, via direction_coef), so this
        only reads lift_height_threshold, not tilt_tolerance_deg."""
        threshold = self.variation_config.get("lift_height_threshold")
        if threshold is None:
            return super()._check_success()
        pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]
        return pot_bottom_height > table_height + threshold

    def reward(self, action=None) -> float:
        """Verbatim from TwoArmLift.reward, parameterized on lift_height_threshold and
        tilt_tolerance_deg instead of the hardcoded 0.10m margin / 30-degree tilt tolerance --
        both inline literals upstream, not even instance attributes like
        two_arm_handover's height_threshold (see VARIATIONS["harder_lift"])."""
        threshold = self.variation_config.get("lift_height_threshold")
        tilt_deg = self.variation_config.get("tilt_tolerance_deg")
        if threshold is None and tilt_deg is None:
            return super().reward(action)

        threshold = threshold if threshold is not None else 0.10
        tilt_deg = tilt_deg if tilt_deg is not None else 30.0

        reward = 0

        # check if the pot is tilted more than tilt_deg
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_tolerance = np.cos(np.deg2rad(tilt_deg))
        direction_coef = 1 if cos_z >= cos_tolerance else 0

        if self._check_success():
            reward = 3.0 * direction_coef
        elif self.reward_shaping:
            pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - (threshold - 0.05), 0), threshold + 0.05)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            (g0, g1) = (
                (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
                if self.env_configuration == "single-robot"
                else (self.robots[0].gripper, self.robots[1].gripper)
            )

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward


class RobosuiteTwoArmLiftVariantsEnv(RobosuiteBaseEnv):
    """RobosuiteTwoArmLiftEnv, backed by TwoArmLiftVariants instead of plain TwoArmLift.

    Identical bimanual control/observation interface to RobosuiteTwoArmLiftEnv
    (robosuite_two_arm_lift.py); the only difference is the extra `variation`/
    `variation_overrides` kwargs forwarded to TwoArmLiftVariants for robustness-eval
    perturbations, and constructing TwoArmLiftVariants directly rather than via
    suite.make("TwoArmLift", ...) since TwoArmLiftVariants isn't registered with
    robosuite's env registry. `variation="baseline"` reproduces the original task exactly:
    suite.make("TwoArmLift", **kwargs) itself just resolves the registry to the TwoArmLift
    class and constructs TwoArmLift(**kwargs), so constructing TwoArmLiftVariants(**kwargs)
    (a TwoArmLift subclass) directly with the same kwargs accepts the same arguments and
    produces the same scene/robots/observations when variation="baseline".
    """

    _SUBSAMPLE_RATE = 4

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl2.json",
        max_steps: int = 5000,
        seed: int | None = None,
        viser_debug: bool = False,  # TODO: move the viser visualization manager into a separate class, low level env agnostic
        privileged: bool = True,
        enable_render: bool = False,
        variation: str | Sequence[str] = "baseline",
        variation_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        self.save_camera_name = "agentview"  # Scene-level camera to show both arms
        self.render_camera_names = ["agentview"]  # Scene-level camera for observations
        self.segmentation_level = "instance"

        # Initialize Robosuite environment
        # TwoArmLiftVariants requires 2 robots
        # Load controller config for both robots (same config for both)
        controller_config = load_composite_controller_config(controller=self.controller_cfg)

        common_kwargs: dict[str, Any] = dict(
            variation=variation,
            variation_overrides=variation_overrides,
            robots=["Panda", "Panda"],
            env_configuration="opposed",
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=[controller_config, controller_config],
            horizon=max_steps,
            use_object_obs=True,
            reward_shaping=True,
        )

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = TwoArmLiftVariants(
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    **common_kwargs,
                )
            else:
                self.robosuite_env = TwoArmLiftVariants(
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    use_camera_obs=True,
                    camera_depths=True,
                    **common_kwargs,
                )
        else:
            self.robosuite_env = TwoArmLiftVariants(
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                use_camera_obs=True,
                camera_depths=True,
                **common_kwargs,
            )

        # Get camera ID and modify its position and orientation
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.653, 0.271, 0.271, 0.653]

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

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle_0 = None
            self.mjcf_ee_frame_handle_1 = None
            self.handle0_frame_handle = None
            self.handle1_frame_handle = None
            self.det_handle0_frame_handle = None
            self.det_handle1_frame_handle = None
            self.det_handle0_bbox_center_handle = None
            self.det_handle1_bbox_center_handle = None
            self.detected_poses = {}
            self.detected_bbox_centers = {}

            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584

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

    # ----------------------- FrankaControlApi Interface -----------------------

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
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

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
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

    def move_to_joints_blocking_both(
        self,
        joints0: np.ndarray,
        joints1: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        """Move both robots to target joint positions simultaneously using Robosuite's controller.

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
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current0 = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            current1 = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence for both
            error0 = np.linalg.norm(current0 - target0)
            error1 = np.linalg.norm(current1 - target1)

            if error0 < tolerance and error1 < tolerance:
                break

            # Build Robosuite action for both robots
            robot0_action = np.concatenate([target0, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([target1, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)
            self._sim_step_count += 1

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1

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

    def _pot_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get pot poses in robot base frame.

        Returns:
            Dict with "pot_pos", "handle0_pos", "handle1_pos" keys
        """
        # Get transforms
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )

        # Get pot position and quaternion
        pot_pos_world = robosuite_obs.get("pot_pos", np.zeros(3))
        pot_quat_xyzw = robosuite_obs.get("pot_quat", np.array([0, 0, 0, 1]))
        # Convert quaternion from xyzw to wxyz
        pot_quat_wxyz = np.array(
            [pot_quat_xyzw[3], pot_quat_xyzw[0], pot_quat_xyzw[1], pot_quat_xyzw[2]]
        )

        # Get handle positions
        # Robosuite usually provides handle_xpos but for TwoArmLift, need to check keys.
        # Assuming handle0_xpos and handle1_xpos exist for TwoArmLift
        # If not present, we might need to infer them from pot pose.
        handle0_pos_world = robosuite_obs.get(
            "handle0_xpos", pot_pos_world + np.array([0.1, 0, 0])
        )  # Fallback/Guess
        handle1_pos_world = robosuite_obs.get(
            "handle1_xpos", pot_pos_world + np.array([-0.1, 0, 0])
        )  # Fallback/Guess

        # Transform pot to robot base frame
        pot_world = vtf.SE3(wxyz_xyz=np.concatenate([pot_quat_wxyz, pot_pos_world]))
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        pot_robot_base = base_transform @ pot_world

        # Transform handle positions (use same rotation as pot for now, or identity if points)
        handle0_world = vtf.SE3.from_translation(handle0_pos_world)
        handle0_robot_base = base_transform @ handle0_world

        handle1_world = vtf.SE3.from_translation(handle1_pos_world)
        handle1_robot_base = base_transform @ handle1_world

        return {
            "pot": [
                float(x)
                for x in np.concatenate(
                    [pot_robot_base.translation(), pot_robot_base.rotation().wxyz]
                )
            ],
            "handle0": [
                float(x)
                for x in np.concatenate(
                    [
                        handle0_robot_base.translation(),
                        pot_robot_base.rotation().wxyz,
                    ]  # Using pot rotation
                )
            ],
            "handle1": [
                float(x)
                for x in np.concatenate(
                    [
                        handle1_robot_base.translation(),
                        pot_robot_base.rotation().wxyz,
                    ]  # Using pot rotation
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse lift reward."""
        # Use the robosuite environment's built-in reward function
        reward = float(self.robosuite_env.reward())
        return reward

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._pot_pose_dict(robosuite_obs)

        # Store pot poses explicitly
        pot_pose_array = np.stack(
            [
                np.asarray(pose_dict["pot"], dtype=np.float32),
                np.asarray(pose_dict["handle0"], dtype=np.float32),
                np.asarray(pose_dict["handle1"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["pot_poses"] = {
            "pot": pot_pose_array[0],
            "handle0": pot_pose_array[1],
            "handle1": pot_pose_array[2],
        }

        # Camera pose/intrinsics/images -- identical logic to RobosuiteTwoArmLiftEnv's own
        # per-camera loop, expressed in robot0's base frame; reused from RobosuiteBaseEnv
        # since the computation is generic given an explicit base_wxyz_xyz.
        self._process_camera_observations(robosuite_obs, base_wxyz_xyz=self.base_link_wxyz_xyz_0)

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
        else:
            raise ValueError("Unhandle case, with multiple camera names")

        return robosuite_obs

    # Temporary viser debugging (bimanual-specific; overrides RobosuiteBaseEnv's single-arm
    # implementation, which assumes one gripper/base frame)

    def _update_viser_server(
        self,
    ) -> None:
        try:
            obs = self.get_observation()
            if self.viser_server is not None:
                self._viser_init_check()

                obs_cartesian_0 = obs["robot0_cartesian_pos"][:-1]
                obs_cartesian_1 = obs["robot1_cartesian_pos"][:-1]

                self.mjcf_ee_frame_handle_0.position = obs_cartesian_0[:3]
                self.mjcf_ee_frame_handle_0.wxyz = obs_cartesian_0[3:]

                self.mjcf_ee_frame_handle_1.position = obs_cartesian_1[:3]
                self.mjcf_ee_frame_handle_1.wxyz = obs_cartesian_1[3:]

                # Add handle visualization
                if "pot_poses" in obs:
                    handle0_pose = obs["pot_poses"]["handle0"]
                    if self.handle0_frame_handle is not None:
                        self.handle0_frame_handle.position = handle0_pose[:3]
                        self.handle0_frame_handle.wxyz = handle0_pose[3:]

                    handle1_pose = obs["pot_poses"]["handle1"]
                    if self.handle1_frame_handle is not None:
                        self.handle1_frame_handle.position = handle1_pose[:3]
                        self.handle1_frame_handle.wxyz = handle1_pose[3:]

                if hasattr(self, "detected_poses") and "handle0" in self.detected_poses:
                    pos, wxyz = self.detected_poses["handle0"]
                    if self.det_handle0_frame_handle is not None:
                        self.det_handle0_frame_handle.position = pos
                        self.det_handle0_frame_handle.wxyz = wxyz

                if hasattr(self, "detected_poses") and "handle1" in self.detected_poses:
                    pos, wxyz = self.detected_poses["handle1"]
                    if self.det_handle1_frame_handle is not None:
                        self.det_handle1_frame_handle.position = pos
                        self.det_handle1_frame_handle.wxyz = wxyz

                # Update bbox center visualizations
                if hasattr(self, "detected_bbox_centers"):
                    if "handle0" in self.detected_bbox_centers:
                        bbox_pos = self.detected_bbox_centers["handle0"]
                        if self.det_handle0_bbox_center_handle is not None:
                            self.det_handle0_bbox_center_handle.position = bbox_pos
                            # Use identity quaternion for bbox center (no rotation)
                            self.det_handle0_bbox_center_handle.wxyz = np.array(
                                [1.0, 0.0, 0.0, 0.0]
                            )

                    if "handle1" in self.detected_bbox_centers:
                        bbox_pos = self.detected_bbox_centers["handle1"]
                        if self.det_handle1_bbox_center_handle is not None:
                            self.det_handle1_bbox_center_handle.position = bbox_pos
                            # Use identity quaternion for bbox center (no rotation)
                            self.det_handle1_bbox_center_handle.wxyz = np.array(
                                [1.0, 0.0, 0.0, 0.0]
                            )

                rbg_imgs = obs_get_rgb(obs)
                for image_key in rbg_imgs:
                    if self.viser_img_handle is not None:
                        self.viser_img_handle.image = rbg_imgs[image_key]

                    if "pose" in obs[image_key]:
                        self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                        self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                        self.image_frustum_handle.image = rbg_imgs[image_key]
                    else:
                        self.image_frustum_handle.visible = False

                # Temporary hardcode to visualise some stuff for debugging
                if "depth" in obs["agentview"]["images"] and "agentview" in rbg_imgs:
                    depth_img = obs["agentview"]["images"]["depth"]
                    if depth_img.ndim == 3:
                        depth_img = depth_img[:, :, 0]

                    points, colors = depth_color_to_pointcloud(
                        depth_img,
                        rbg_imgs["agentview"],
                        obs["agentview"]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        "agentview/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )
        except Exception:
            # Silently catch viser update errors to prevent blocking execution
            # The visualization is non-critical, so we don't want it to stop the robot
            pass

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle_0 is None:
            self.mjcf_ee_frame_handle_0 = self.viser_server.scene.add_frame(
                "/panda0_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.mjcf_ee_frame_handle_1 is None:
            self.mjcf_ee_frame_handle_1 = self.viser_server.scene.add_frame(
                "/panda1_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.handle0_frame_handle is None:
            self.handle0_frame_handle = self.viser_server.scene.add_frame(
                "/ground_truth/handle0", axes_length=0.15, axes_radius=0.005
            )
        if self.handle1_frame_handle is None:
            self.handle1_frame_handle = self.viser_server.scene.add_frame(
                "/ground_truth/handle1", axes_length=0.15, axes_radius=0.005
            )

        if self.det_handle0_frame_handle is None:
            self.det_handle0_frame_handle = self.viser_server.scene.add_frame(
                "/detected/handle0", axes_length=0.15, axes_radius=0.005
            )
        if self.det_handle1_frame_handle is None:
            self.det_handle1_frame_handle = self.viser_server.scene.add_frame(
                "/detected/handle1", axes_length=0.15, axes_radius=0.005
            )

        if self.det_handle0_bbox_center_handle is None:
            self.det_handle0_bbox_center_handle = self.viser_server.scene.add_frame(
                "/detected/handle0_bbox_center", axes_length=0.1, axes_radius=0.003
            )
        if self.det_handle1_bbox_center_handle is None:
            self.det_handle1_bbox_center_handle = self.viser_server.scene.add_frame(
                "/detected/handle1_bbox_center", axes_length=0.1, axes_radius=0.003
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


__all__ = ["TwoArmLiftVariants", "RobosuiteTwoArmLiftVariantsEnv", "VARIATIONS"]
