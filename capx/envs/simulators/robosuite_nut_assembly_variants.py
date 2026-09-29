"""Config-driven robustness-eval perturbations of robosuite's NutAssemblySquare task.

New sibling env only: robosuite's NutAssembly/NutAssemblySquare (capx/third_party/robosuite/
robosuite/environments/manipulation/nut_assembly.py) and capx's FrankaRobosuiteNutAssembly
(robosuite_nut_assembly.py) are both left untouched, so existing benchmark configs are
unaffected. Mirrors the pattern already validated for robosuite's Stack task (StackVariants,
robosuite_cubes_variants.py): a `variation` preset selects a named combination of
perturbations, applied on top of the vanilla task.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import viser
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.nut_assembly import NutAssembly, NutAssemblySquare
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variants_common import resolve_variation_config
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")

# NutAssembly's own upstream `_load_model` always constructs *both* a SquareNut and a RoundNut
# and registers both with the (per-nut-type) placement initializer -- NutAssemblySquare's
# single_object_mode=2 doesn't remove the round nut from the model, it just teleports it to
# (10, 10, 10) every reset via `clear_objects` (see environments/base.py). So unlike Stack
# (which has no spare object lying around and has to merge a brand new mesh into the model),
# the natural "distractor" here is simply the *existing* round nut: instead of banishing it,
# resample its pose from its own dedicated sub-sampler ("RoundNutSampler") and leave it near
# the table, while `_check_success` is overridden so it can never be scored.
DISTRACTOR_NUT_NAMES: dict[str, str] = {"round_nut": "RoundNut", "square_nut": "SquareNut"}

# Matches NutAssembly._load_model's own per-nut-type default sampler ranges exactly, so
# variation="baseline" reproduces today's nut_assembly task exactly and so object_position_bias
# has a well-defined default drop-zone to bias/rescale.
_NUT_SAMPLER_DEFAULT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "SquareNut": {"x_range": (-0.115, -0.11), "y_range": (0.11, 0.225)},
    "RoundNut": {"x_range": (-0.115, -0.11), "y_range": (-0.225, -0.11)},
}

# Handle-site offset (relative to the nut's own body frame) baked into each nut's MJCF, read
# directly from models/assets/objects/{square,round}-nut.xml's "handle_site" <site pos=.../>.
_NUT_HANDLE_OFFSETS: dict[str, np.ndarray] = {
    "square": np.array([0.054, 0.0, 0.0]),
    "round": np.array([0.06, 0.0, 0.0]),
}

CANONICAL_PROMPT_TEMPLATE = "Insert the {nut} nut onto the small {nut} peg. Quaternions are WXYZ."
PROMPT_PARAPHRASE_TEMPLATES = [
    CANONICAL_PROMPT_TEMPLATE,
    "Place the {nut} nut onto its matching {nut} peg. Quaternions are WXYZ.",
    "Fit the {nut} nut over the {nut} peg. Quaternions are WXYZ.",
    "Assemble the {nut} nut onto the {nut}-shaped peg. Quaternions are WXYZ.",
]

# Named variation presets. Each maps to a dict of overrides consumed by NutAssemblyVariants.
# "baseline" reproduces the original NutAssemblySquare task and is useful as a control condition
# when measuring policy robustness against the other nine perturbations.
# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt says "brown square nut"; color_tint can make it a different color entirely, and
# target_object_swap (swap_nut_type: "round") makes it a round nut -- "brown" and "square"
# both go stale, and neither is corrected in the prompt text.
VARIATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "color_shift_mild": {"color_tint": (1.0, 0.75, 0.75, 1.0)},
    "color_shift_strong": {"color_tint": (0.2, 1.0, 0.3, 1.0)},
    "object_position_shift": {"object_position_bias": {"bias": (0.6, 0.6), "scale": 0.5}},
    "robot_position_shift": {"robot_base_offset": (0.06, 0.06, 0.0)},
    "lighting_bright": {"light_diffuse_scale": 2.0, "light_ambient_scale": 2.5},
    "lighting_dim": {"light_diffuse_scale": 0.3, "light_ambient_scale": 0.3},
    "distractor_object": {"distractor_classes": ["round_nut"]},
    "target_object_swap": {"swap_nut_type": "round"},
    "language_paraphrase": {"paraphrase_language": True},
    "compound_hard": {
        "color_tint": (1.0, 0.75, 0.75, 1.0),
        "light_diffuse_scale": 0.5,
        "light_ambient_scale": 0.5,
        "distractor_classes": ["round_nut"],
        "paraphrase_language": True,
    },
    # Difficulty knob, not a robustness perturbation: single_object_mode is already a
    # first-class upstream NutAssembly constructor kwarg (mode 0 puts both nuts in the scene,
    # both scored -- reward()/_check_success() already loop over self.nuts generically, no
    # override needed), so this is genuinely free unlike stack_3_cubes/lift_higher. Not
    # composable with color_tint/object_position_bias/distractor_classes/swap_nut_type -- those
    # all target "the active nut" (self.active_nut_type/_active_nut_name), a concept that
    # doesn't apply once both nuts are in play; combining both_nuts with one of those is a
    # silent no-op for that perturbation rather than an error (self.active_nut_type still
    # defaults to "square" so _active_nut_name stays valid, it just doesn't correspond to
    # anything scored).
    "both_nuts": {"both_nuts": True},
}


class NutAssemblyVariants(NutAssemblySquare):
    """NutAssemblySquare with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint multiplier applied to the active nut's material(s).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the active nut is sampled within its own drop-zone, instead of the
          default (narrow, off-center) upstream range.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to the robot's base position.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_NUT_NAMES; the corresponding nut
          (round or square) is left near the table instead of being cleared out of the scene.
          Never scored -- see `_check_success` -- and has no assembly role.
        - swap_nut_type ("square" or "round"): NutAssembly already exposes `nut_type` as a
          first-class constructor kwarg for selecting which nut is the active/scored one, so
          target-object swap is implemented by threading this straight through to it rather than
          re-deriving `_load_model` (unlike Stack, which has to rebuild cubeA from scratch).
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool of
          paraphrases instead of the fixed canonical wording.
        - both_nuts (bool): if True, uses upstream single_object_mode=0 (both the square and
          round nut present, both must be inserted). A difficulty knob, not a perturbation --
          see the VARIATIONS["both_nuts"] note on why it doesn't compose with the other presets.

    Pick one of the ~10 named presets in `VARIATIONS` via `variation`, optionally layering
    ad-hoc tweaks via `variation_overrides`.
    """

    def __init__(
        self,
        variation: str | Sequence[str] = "baseline",
        variation_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.variation, self.variation_config = resolve_variation_config(
            VARIATIONS, variation, variation_overrides
        )
        self.language_instruction: str | None = None

        swap_nut_type = self.variation_config.get("swap_nut_type", "square")
        assert swap_nut_type in ("square", "round"), "swap_nut_type must be 'square' or 'round'"
        self.active_nut_type = swap_nut_type

        # NutAssemblySquare.__init__ hard-asserts "single_object_mode"/"nut_type" aren't in
        # kwargs and always forces nut_type="square" -- bypass it (going straight to the
        # grandparent NutAssembly.__init__ instead of calling super().__init__()) so that
        # target_object_swap can select nut_type="round" via that same upstream mechanism.
        # This reproduces exactly what NutAssemblySquare.__init__ does, parameterized on nut_type.
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        if self.variation_config.get("both_nuts", False):
            # nut_type is "only used if single_object_mode is 2" (NutAssembly's own docstring)
            # -- omitted here since mode 0 ignores it, matching upstream's own convention.
            NutAssembly.__init__(self, single_object_mode=0, **kwargs)
        else:
            NutAssembly.__init__(self, single_object_mode=2, nut_type=swap_nut_type, **kwargs)

    @property
    def _active_nut_name(self) -> str:
        return "SquareNut" if self.active_nut_type == "square" else "RoundNut"

    # ----------------------- Scene construction -----------------------

    def _load_model(self) -> None:
        super()._load_model()

        # All mutations below operate on XML elements/samplers already attached to self.model /
        # self.placement_initializer -- the sim isn't compiled from this tree until later in the
        # environment's load/reset sequence, so it's safe to edit them here.
        self._apply_color_tint()
        self._apply_lighting()
        self._apply_robot_offset()
        self._apply_object_position_bias()

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        nut_name = self._active_nut_name
        nut_obj = next(obj for obj in self.nuts if obj.name == nut_name)
        materials = find_elements(root=nut_obj.asset, tags="material", return_first=False) or []
        for material in materials:
            material.set("rgba", rgba_str)

    def _apply_lighting(self) -> None:
        diffuse_scale = self.variation_config.get("light_diffuse_scale")
        ambient_scale = self.variation_config.get("light_ambient_scale")
        if diffuse_scale is None and ambient_scale is None:
            return
        base_diffuse = np.array([0.7, 0.7, 0.7])
        base_ambient = np.array([0.15, 0.15, 0.15])
        lights = (
            find_elements(root=self.model.mujoco_arena.worldbody, tags="light", return_first=False)
            or []
        )
        for light in lights:
            if diffuse_scale is not None:
                light.set("diffuse", array_to_string(np.clip(base_diffuse * diffuse_scale, 0, 1)))
            if ambient_scale is not None:
                light.set("ambient", array_to_string(np.clip(base_ambient * ambient_scale, 0, 1)))

    def _apply_robot_offset(self) -> None:
        offset = self.variation_config.get("robot_base_offset")
        if offset is None:
            return
        base_pos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        self.robots[0].robot_model.set_base_xpos(base_pos + np.array(offset))

    def _apply_object_position_bias(self) -> None:
        pos_cfg = self.variation_config.get("object_position_bias")
        if pos_cfg is None:
            return
        if not isinstance(self.placement_initializer, SequentialCompositeSampler):
            return

        nut_name = self._active_nut_name
        sampler = self.placement_initializer.samplers[f"{nut_name}Sampler"]
        default_x = _NUT_SAMPLER_DEFAULT_RANGES[nut_name]["x_range"]
        default_y = _NUT_SAMPLER_DEFAULT_RANGES[nut_name]["y_range"]
        bias_x, bias_y = pos_cfg.get("bias", (0.0, 0.0))
        scale = pos_cfg.get("scale", 1.0)

        for axis_range, bias, attr in ((default_x, bias_x, "x_range"), (default_y, bias_y, "y_range")):
            lo, hi = axis_range
            center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            new_center = center + bias * half
            new_half = half * scale
            setattr(
                sampler,
                attr,
                [
                    float(np.clip(new_center - new_half, lo, hi)),
                    float(np.clip(new_center + new_half, lo, hi)),
                ],
            )

    # ----------------------- Reset / success -----------------------

    def _reset_internal(self) -> None:
        super()._reset_internal()

        distractor_names = self.variation_config.get("distractor_classes", [])
        if distractor_names and isinstance(self.placement_initializer, SequentialCompositeSampler):
            for key in distractor_names:
                nut_name = DISTRACTOR_NUT_NAMES[key]
                if nut_name == self.obj_to_use:
                    continue  # already the active/scored nut; nothing to un-clear
                sampler = self.placement_initializer.samplers[f"{nut_name}Sampler"]
                obj_pos, obj_quat, obj = sampler.sample()[nut_name]
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

        paraphrase = self.variation_config.get("paraphrase_language", False)
        templates = PROMPT_PARAPHRASE_TEMPLATES if paraphrase else [CANONICAL_PROMPT_TEMPLATE]
        template = str(self.rng.choice(templates)) if paraphrase else CANONICAL_PROMPT_TEMPLATE
        self.language_instruction = template.format(nut=self.active_nut_type)

    def _check_success(self) -> bool:
        success = super()._check_success()
        if self.variation_config.get("distractor_classes"):
            # Base _check_success (single_object_mode > 0) returns True if ANY nut in self.nuts
            # is on its peg -- normally safe because the non-active nut is cleared out to
            # (10, 10, 10) by _reset_internal. Since distractor nuts are deliberately left in the
            # task space here, restrict success to just the active/scored nut so a decorative
            # distractor landing on its own peg by chance can never trigger a false success.
            return bool(self.objects_on_pegs[self.nut_id])
        return success

    def get_ep_meta(self) -> dict[str, Any]:
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.language_instruction
        return ep_meta


class FrankaRobosuiteNutAssemblyVariantsLowLevel(RobosuiteBaseEnv):
    """FrankaRobosuiteNutAssembly, backed by NutAssemblyVariants instead of plain NutAssemblySquare.

    Identical control/observation interface to FrankaRobosuiteNutAssembly (robosuite_nut_assembly.py);
    the only difference is the extra `variation`/`variation_overrides` kwargs forwarded to
    NutAssemblyVariants for robustness-eval perturbations. `variation="baseline"` reproduces the
    original NutAssemblySquare task exactly.
    """

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1000,
        seed: int | None = None,
        viser_debug: bool = False,
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

        self.save_camera_name = "birdview"
        self.render_camera_names = [self.save_camera_name]

        # Initialize Robosuite environment
        self.privileged = privileged
        common_kwargs: dict[str, Any] = dict(
            variation=variation,
            variation_overrides=variation_overrides,
            robots=["Panda"],
            reward_shaping=True,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            camera_heights=self._render_height,
            camera_widths=self._render_width,
        )
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = NutAssemblyVariants(
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    **common_kwargs,
                )
            else:
                self.robosuite_env = NutAssemblyVariants(
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    **common_kwargs,
                )
        else:
            self.robosuite_env = NutAssemblyVariants(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_depths=True,
                renderer="mujoco",
                **common_kwargs,
            )

        # nut type
        self.nut_type = self.robosuite_env.active_nut_type
        self.nut_id = self.robosuite_env.nut_to_id[self.nut_type]
        self.nuts = self.robosuite_env.nuts

        self._init_robot_links()

        # Precompute fast joint qpos addresses for Panda (avoid heavy _get_observations in tight loops)
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self._panda_joint_qpos_addrs: list[int] = []
        for jn in joint_names:
            addr = self.robosuite_env.sim.model.get_joint_qpos_addr(jn)
            if isinstance(addr, tuple):
                addr = addr[0]
            self._panda_joint_qpos_addrs.append(int(addr))
        self.home_joint_position: np.ndarray | None = None

        # Temporary viser debugging
        self.viser_debug = viser_debug
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.mjcf_gripper_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.urdf = load_robot_description("panda_description")
            self.urdf_vis = ViserUrdf(self.viser_server, urdf_or_path=self.urdf, load_meshes=True)
            self._viser_init_check()

            self.cube_points = None
            self.cube_color = None
            self.cube_center = None
            self.cube_rot = None
            self.grasp_frame_position = None
            self.grasp_frame_orientation = None
        else:
            self.viser_server = None

        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        first_obs = self.robosuite_env.reset()
        self.home_joint_position = np.array(first_obs["robot0_joint_pos"], dtype=np.float64)
        self.robosuite_env.sim.data.qpos[:7] = np.array(
            [0, -1.585, 0, -2.645, 0, 1, 0.785]
        )  # should be out of the view from top camera
        self.robosuite_env.sim.forward()
        self._current_joints = self.robosuite_env.sim.data.qpos[:7].copy()

        self._step_count = 0
        self._sim_step_count = 0

        self._gripper_fraction = 1.0

        # NOTE: This seems very important: post reset there seems to be a bit of noise on the joint velocity
        # also the blocks are reset to be in air, so if we sample from the initial state, the blocks are not at the correct height
        for _ in range(100):
            self._step_once()

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        # nut_type is fixed for the lifetime of the env (set at construction from the variation
        # config), but re-read it here in case a future variation ever randomizes it per-episode.
        self.nut_type = self.robosuite_env.active_nut_type
        self.nut_id = self.robosuite_env.nut_to_id[self.nut_type]

        info = {"task_prompt": self.robosuite_env.get_ep_meta()["lang"]}

        return obs, info

    # Override _step_once to include the double _sim_step_count increment from original
    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        action = self._build_action()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self.robosuite_env.step(action[:-1])
        else:
            self.robosuite_env.step(action[:-1], skip_render_images=True)

        self._sim_step_count += 1

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

        self._sim_step_count += 1

    # Override move_to_joints_blocking to use fast qpos path
    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move to target joint positions using Robosuite's controller."""
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            # Fast path via direct qpos indices
            current = np.array(
                self.robosuite_env.sim.data.qpos[self._panda_joint_qpos_addrs], dtype=np.float64
            )

            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0

            need_render = (self._record_frames and self._sim_step_count % self._subsample_rate == 0) or self.viser_debug
            if need_render:
                self.robosuite_env.step(action[:-1])
            else:
                self.robosuite_env.step(action[:-1], skip_render_images=True)

            self._sim_step_count += 1

            if self.viser_debug and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def _get_nut_pose(self, robosuite_obs: dict[str, Any]) -> dict[str, Any]:
        """Get the active nut's (and its handle's, and its peg's) pose in the robot base frame."""
        nut_type = self.nut_type
        nut_name = "SquareNut" if nut_type == "square" else "RoundNut"
        peg_body_id = self.robosuite_env.peg1_body_id if nut_type == "square" else self.robosuite_env.peg2_body_id
        handle_offset = _NUT_HANDLE_OFFSETS[nut_type]

        invert_grasp_pose = vtf.SE3.from_matrix(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        _nut_xyzw = np.asarray(robosuite_obs[f"{nut_name}_quat"], dtype=np.float64)
        _nut_wxyz = np.array([_nut_xyzw[3], _nut_xyzw[0], _nut_xyzw[1], _nut_xyzw[2]], dtype=np.float64)
        nut_world = vtf.SE3(wxyz_xyz=np.concatenate([_nut_wxyz, robosuite_obs[f"{nut_name}_pos"]]))
        nut_handle_world = nut_world @ vtf.SE3.from_translation(handle_offset)

        peg_world = vtf.SE3(
            wxyz_xyz=np.concatenate(
                [
                    self.robosuite_env.sim.data.xquat[peg_body_id],
                    self.robosuite_env.sim.data.xpos[peg_body_id],
                ]
            )
        )

        rotate_by_180 = vtf.SE3.from_rotation(rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi))

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        nut_robot_base = base_transform @ nut_world @ rotate_by_180 @ invert_grasp_pose
        nut_handle_robot_base = base_transform @ nut_handle_world @ rotate_by_180 @ invert_grasp_pose
        peg_height = 0.1
        peg_robot_base = (
            base_transform
            @ peg_world
            @ vtf.SE3.from_translation(translation=np.array([0, 0, peg_height]))
            @ invert_grasp_pose
        )

        return {
            "nut_handle_to_center_offset": handle_offset,
            f"{nut_type}_nut": np.concatenate(
                [nut_robot_base.translation(), nut_robot_base.rotation().wxyz]
            ),
            f"{nut_type}_nut_handle": np.concatenate(
                [
                    nut_handle_robot_base.translation(),
                    nut_handle_robot_base.rotation().wxyz,
                ]
            ),
            f"{nut_type}_peg": np.concatenate(
                [peg_robot_base.translation(), peg_robot_base.rotation().wxyz]
            ),
        }

    def compute_reward(self) -> float:
        """Compute reward from the Robosuite environment."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaRobosuiteNutAssembly format."""
        robosuite_obs = self.robosuite_env._get_observations(force_update=True)
        pose_dict = self._get_nut_pose(robosuite_obs)

        robosuite_obs["nut_poses"] = pose_dict

        # Nut assembly uses birdview camera but stores results under "robot0_robotview" key
        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}
            if "robot0_robotview" not in robosuite_obs:
                robosuite_obs["robot0_robotview"] = {}

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
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
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

            robosuite_obs["robot0_robotview"]["pose"] = (
                np.concatenate(
                    [
                        cam_robot_tf.translation(),
                        cam_robot_tf.rotation().wxyz,
                    ]
                )
            )
            robosuite_obs["robot0_robotview"]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs["robot0_robotview"]["intrinsics"] = K

            robosuite_obs["robot0_robotview"]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs["robot0_robotview"]["images"]["rgb"] = robosuite_obs[
                    camera_name + "_image"
                ][::-1]
            if camera_name + "_depth" in robosuite_obs:
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )
                robosuite_obs["robot0_robotview"]["images"]["depth"] = depth_metric

        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs

    # Nut assembly has its own viser debugging logic
    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if self.viser_debug:
            self._viser_init_check()

            obs["robot_cartesian_pos"][:-1]

            action_joint_copy = obs["robot_joint_pos"].copy()
            action_joint_copy[-1] /= self.gripper_metric_length

            self.urdf_vis.update_cfg(action_joint_copy)

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                # Point cloud from depth
                if "depth" in obs[image_key].get("images", {}):
                    points_robot, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points_robot,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )

            nut_key = f"{self.nut_type}_nut"
            handle_key = f"{self.nut_type}_nut_handle"
            peg_key = f"{self.nut_type}_peg"
            if nut_key in obs["nut_poses"]:
                self.viser_server.scene.add_frame(
                    f"{nut_key}_frame",
                    position=obs["nut_poses"][nut_key][:3],
                    wxyz=obs["nut_poses"][nut_key][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )
                self.viser_server.scene.add_frame(
                    f"{handle_key}_frame",
                    position=obs["nut_poses"][handle_key][:3],
                    wxyz=obs["nut_poses"][handle_key][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )
                self.viser_server.scene.add_frame(
                    "peg_frame",
                    position=obs["nut_poses"][peg_key][:3],
                    wxyz=obs["nut_poses"][peg_key][3:],
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_center is not None and self.cube_rot is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/cube_frame",
                    position=self.cube_center,
                    wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_points is not None and self.cube_color is not None:
                self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/cube_point_cloud",
                    self.cube_points,
                    self.cube_color,
                    point_size=0.001,
                    point_shape="square",
                )

            if self.grasp_frame_position is not None and self.grasp_frame_orientation is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=self.grasp_frame_position,
                    wxyz=self.grasp_frame_orientation,
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

            if hasattr(self, "grasp_sample") and self.grasp_sample is not None:
                grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                    np.array([0, 0, 0.1])
                )
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=grasp_tf.wxyz_xyz[-3:],
                    wxyz=grasp_tf.wxyz_xyz[:4],
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

            self.mjcf_gripper_frame_handle = self.viser_server.scene.add_frame(
                "/panda_gripper_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = [
    "NutAssemblyVariants",
    "FrankaRobosuiteNutAssemblyVariantsLowLevel",
    "VARIATIONS",
]
