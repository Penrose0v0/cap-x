"""Config-driven robustness-eval perturbations of robosuite's Lift task.

New sibling env only: robosuite's Lift (capx/third_party/robosuite/robosuite/environments/
manipulation/lift.py) and capx's FrankaRobosuiteCubeLiftLowLevel (robosuite_cube_lift.py) are
both left untouched, so existing benchmark configs are unaffected. Mirrors the pattern already
validated for robosuite's Stack task (StackVariants in robosuite_cubes_variants.py): a
`variation` preset selects a named combination of perturbations, applied on top of the vanilla
task.

Unlike Stack, capx's FrankaRobosuiteCubeLiftLowLevel constructs plain `Lift(...)` with no custom
placement_initializer -- it inherits Lift's own built-in default sampling range ([-0.03, 0.03] on
both axes) unmodified. So, unlike StackVariants (which had to unconditionally reapply capx's own
wider range so that "baseline" would be a true no-op relative to FrankaRobosuiteCubesLowLevel),
LiftVariants needs no such correction: calling Lift._load_model() unmodified already reproduces
the existing cube_lift task exactly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variant_camera import CAMERA_ORBIT_PRESETS
from capx.envs.simulators.variants_common import VariantMixin

# Object classes usable as a visually-similar swap for the lift cube, or as task-irrelevant
# distractors. Lift only has a single scored object (self.cube), unlike Stack's cubeA/cubeB.
SIMILAR_OBJECT_CLASSES = {"cylinder": CylinderObject}
DISTRACTOR_OBJECT_CLASSES = {"cylinder": CylinderObject}

# NOTE: this exact wording (including "top of the secondary cube", a leftover from the Stack
# task) is what FrankaRobosuiteCubeLiftLowLevel.reset() currently hardcodes as `task_prompt`
# (robosuite_cube_lift.py), even though Lift only ever has one cube and no stacking role. It
# reads like a copy/paste bug, but it is preserved verbatim here so that variation="baseline"
# is a true no-op relative to the existing task. `language_paraphrase` is the only variation
# that actually deviates from it, using Lift-appropriate wording instead.
CANONICAL_PROMPT = "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Lift the cube up off the table surface. Quaternions are WXYZ.",
    "Pick up the cube and raise it into the air. Quaternions are WXYZ.",
    "Grasp the cube and lift it clear of the table. Quaternions are WXYZ.",
]

# Named variation presets. Each maps to a dict of overrides consumed by LiftVariants.
# "baseline" reproduces the original Lift task and is useful as a control condition when
# measuring policy robustness against the other nine perturbations.
# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt says "the red cube"; color_tint can recolor it to something else entirely (e.g.
# color_shift_strong is a green tint), and target_object_swap turns it into a cylinder --
# neither updates the prompt text.
VARIATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "color_shift_mild": {"color_tint": (1.0, 0.75, 0.75, 1.0)},
    "color_shift_strong": {"color_tint": (0.2, 1.0, 0.3, 1.0)},
    "object_position_shift": {"object_position_bias": {"bias": (0.6, 0.6), "scale": 0.5}},
    "robot_position_shift": {"robot_base_offset": (0.06, 0.06, 0.0)},
    "lighting_bright": {"light_diffuse_scale": 2.0, "light_ambient_scale": 2.5},
    "lighting_dim": {"light_diffuse_scale": 0.3, "light_ambient_scale": 0.3},
    "distractor_object": {"distractor_classes": ["cylinder"]},
    "target_object_swap": {"swap_target": "cylinder"},
    "language_paraphrase": {"paraphrase_language": True},
    "compound_hard": {
        "color_tint": (1.0, 0.75, 0.75, 1.0),
        "light_diffuse_scale": 0.5,
        "light_ambient_scale": 0.5,
        "distractor_classes": ["cylinder"],
        "paraphrase_language": True,
    },
    # Difficulty knob, not a robustness perturbation: upstream Lift hardcodes its 0.04m success
    # margin as a literal in both reward() and _check_success() (no constructor kwarg), so
    # raising it needs a method override rather than a passthrough kwarg -- see
    # lift_height_threshold below and the two method overrides that consume it.
    "lift_higher": {"lift_height_threshold": 0.12},
}


VARIATIONS.update(CAMERA_ORBIT_PRESETS)


class LiftVariants(VariantMixin, Lift):
    """Lift with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint multiplier applied to the cube's material(s).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the cube is sampled on the table, instead of the default centered
          range.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to the robot's base
          position.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES to additionally
          place on the table near the lift target. Never scored and have no role in the task.
        - swap_target (str): key of SIMILAR_OBJECT_CLASSES; replaces the cube's shape with a
          visually-similar but different object while keeping its lift-target role.
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool of
          paraphrases instead of the fixed canonical wording.
        - lift_height_threshold (float): overrides upstream Lift's hardcoded 0.04m
          success/reward-shaping margin above the table. A difficulty knob, not a
          perturbation -- see `reward()`/`_check_success()` overrides below.

    Pick one of the ~10 named presets in `VARIATIONS` via `variation`, optionally layering
    ad-hoc tweaks via `variation_overrides`.
    """

    PRESETS = VARIATIONS
    OBS_CAMERA = "robot0_robotview"
    CANONICAL_PROMPT = CANONICAL_PROMPT
    PROMPT_PARAPHRASES = PROMPT_PARAPHRASES

    # Matches Lift's own internal default sampling range. Unlike Stack, capx's
    # FrankaRobosuiteCubeLiftLowLevel never constructs a custom placement_initializer, so there
    # is no wrapper-vs-upstream mismatch to correct here (see module docstring). This constant is
    # used only as the reference half-extent for object_position_bias scaling below.
    _DEFAULT_X_RANGE = (-0.03, 0.03)
    _DEFAULT_Y_RANGE = (-0.03, 0.03)

    def __init__(
        self,
        variation: str | Sequence[str] = "baseline",
        variation_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._init_variant(variation, variation_overrides)
        self._distractor_sampler: UniformRandomSampler | None = None

        super().__init__(**kwargs)

    # ----------------------- Scene construction -----------------------

    def _load_model(self) -> None:
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
        self._apply_object_position_bias()
        self._add_distractors()

    def _load_model_with_swap(self, swap_target: str) -> None:
        """Reproduces Lift._load_model, substituting the cube's class for a visually-similar object.

        Lift builds its cube inline in _load_model rather than via a separate overridable hook
        (same situation as Stack's cubeA/cubeB), so a shape swap has to re-derive this method
        instead of composing on top of it.
        """
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        swap_cls = SIMILAR_OBJECT_CLASSES[swap_target]
        if swap_cls is CylinderObject:
            # radius/half-length chosen at the midpoint of Lift's own cube size range
            # (size_min=[0.020]*3, size_max=[0.022]*3), for a visually-similar footprint.
            self.cube = swap_cls(
                name="cube",
                size=[0.021, 0.021],
                rgba=[1, 0, 0, 1],
                material=redwood,
                rng=self.rng,
            )
        else:
            self.cube = swap_cls(
                name="cube",
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.022, 0.022, 0.022],
                rgba=[1, 0, 0, 1],
                material=redwood,
                rng=self.rng,
            )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=list(self._DEFAULT_X_RANGE),
                y_range=list(self._DEFAULT_Y_RANGE),
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        materials = find_elements(root=self.cube.asset, tags="material", return_first=False) or []
        for material in materials:
            material.set("rgba", rgba_str)

    def _add_distractors(self) -> None:
        distractor_names = self.variation_config.get("distractor_classes", [])
        if not distractor_names:
            self.distractor_objects = []
            self._distractor_sampler = None
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
        # Lift's own default sampling region ([-0.03, 0.03] on both axes) is only 6cm across --
        # unlike Stack's much larger region, it's too tight to also fit a distractor alongside
        # the cube under ensure_valid_placement (this reliably raised RandomizationError in
        # testing). So the distractor gets its own sampler in a region just past the cube's,
        # still on the tabletop and near the lift target rather than sharing its exact bin.
        self._distractor_sampler = UniformRandomSampler(
            name="DistractorSampler",
            mujoco_objects=self.distractor_objects,
            x_range=[0.05, 0.08],
            y_range=[-0.03, 0.03],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

    # ----------------------- Sim references / reset -----------------------

    def _reset_variant_extras(self) -> None:
        if not self.deterministic_reset and self._distractor_sampler is not None:
            distractor_placements = self._distractor_sampler.sample()
            for obj_pos, obj_quat, obj in distractor_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    # ----------------------- Difficulty override -----------------------

    def _check_success(self) -> bool:
        """Verbatim from Lift._check_success, parameterized on lift_height_threshold instead
        of the hardcoded 0.04m margin (see VARIATIONS["lift_higher"])."""
        threshold = self.variation_config.get("lift_height_threshold")
        if threshold is None:
            return super()._check_success()
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return cube_height > table_height + threshold

    def reward(self, action=None) -> float:
        """Verbatim from Lift.reward, parameterized on lift_height_threshold (both the
        success check and the shaping normalization use it, matching upstream's own use of
        0.04 for both) instead of the hardcoded margin."""
        threshold = self.variation_config.get("lift_height_threshold")
        if threshold is None:
            return super().reward(action)

        reward = 0.0
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
            table_height = self.model.mujoco_arena.table_offset[2]
            reward = min((cube_height - table_height) / threshold, 1.0)
        return reward


class FrankaRobosuiteCubeLiftVariantsLowLevel(RobosuiteBaseEnv):
    """FrankaRobosuiteCubeLiftLowLevel, backed by LiftVariants instead of plain Lift.

    Identical control/observation interface to FrankaRobosuiteCubeLiftLowLevel
    (robosuite_cube_lift.py); the only difference is the extra `variation`/`variation_overrides`
    kwargs forwarded to LiftVariants for robustness-eval perturbations. variation="baseline"
    reproduces the original cube_lift task exactly.
    """

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
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

        common_kwargs: dict[str, Any] = dict(
            variation=variation,
            variation_overrides=variation_overrides,
            robots=["Panda"],
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
        )

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = LiftVariants(
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
                self.robosuite_env = LiftVariants(
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    **common_kwargs,
                )
        else:
            self.robosuite_env = LiftVariants(
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                **common_kwargs,
            )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)
        # Additional viser attributes specific to cube_lift
        if viser_debug:
            self.grasp_sample = None
            self.grasp_scores = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        # Adjust initial orientation
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        # We do this because for some reason modifying qpos does not update robot0_joint_pos
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {"task_prompt": self.robosuite_env.get_ep_meta()["lang"]}
        return obs, info

    def _cube_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get cube pose in robot base frame."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cube_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cube_robot_base = base_transform @ cube_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cube_robot_base.translation(), cube_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse/shaped lift reward."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Check if task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs

    # Viser debugging - uses the _update_viser_server from cube_lift's original
    # which checks hasattr(self, "viser_server") instead of self.viser_server is not None

    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if hasattr(self, "viser_server"):
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            from capx.utils.camera_utils import obs_get_rgb
            from capx.utils.depth_utils import depth_color_to_pointcloud
            import viser.transforms as vtf

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                if "depth" in obs[image_key].get("images", {}):
                    points, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
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

            if self.grasp_sample is not None:
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


__all__ = ["LiftVariants", "FrankaRobosuiteCubeLiftVariantsLowLevel", "VARIATIONS"]
