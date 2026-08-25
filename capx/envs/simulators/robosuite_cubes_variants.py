"""Config-driven robustness-eval perturbations of robosuite's Stack task.

New sibling env only: robosuite's Stack (capx/third_party/robosuite/robosuite/environments/
manipulation/stack.py) and capx's FrankaRobosuiteCubesLowLevel (robosuite_cubes.py) are both
left untouched, so existing benchmark configs are unaffected. Mirrors the pattern already
validated for robosuite's PickPlace task (PickPlaceVariants on the Penrose0v0/robosuite@geniac
branch): a `variation` preset selects a named combination of perturbations, applied on top of
the vanilla task.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variants_common import resolve_variation_config

# Object classes usable as a visually-similar swap for cubeA, or as task-irrelevant distractors.
SIMILAR_OBJECT_CLASSES = {"cylinder": CylinderObject}
DISTRACTOR_OBJECT_CLASSES = {"cylinder": CylinderObject}

CANONICAL_PROMPT = "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Stack the first cube on top of the second cube. Quaternions are WXYZ.",
    "Put the primary block on top of the other block. Quaternions are WXYZ.",
    "Set the first cube down onto the second cube. Quaternions are WXYZ.",
]

# Named variation presets. Each maps to a dict of overrides consumed by StackVariants.
# "baseline" reproduces the original Stack task and is useful as a control condition when
# measuring policy robustness against the other nine perturbations.
# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt says "red cube"/"green cube"; color_tint tints BOTH cubes to the same color, and
# target_object_swap turns cubeA into a cylinder. Neither updates the prompt text.
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
}


class StackVariants(Stack):
    """Stack with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint multiplier applied to cubeA/cubeB's material(s).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where cubes are sampled on the table, instead of the default centered range.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to the robot's base position.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES to additionally
          place on the table. Never scored and have no stacking role.
        - swap_target (str): key of SIMILAR_OBJECT_CLASSES; replaces cubeA's shape with a
          visually-similar but different object while keeping its stacking role.
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool of
          paraphrases instead of the fixed canonical wording.

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
        self.distractor_objects: list[Any] = []
        self.language_instruction: str | None = None

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
        self._apply_default_placement_range()
        self._apply_color_tint()
        self._apply_lighting()
        self._apply_robot_offset()
        self._apply_object_position_bias()
        self._add_distractors()

    # Matches FrankaRobosuiteCubesLowLevel's existing sampling range (robosuite_cubes.py), which
    # is wider than upstream Stack's own internal default ([-0.08, 0.08] on both axes). Applied
    # unconditionally so variation="baseline" reproduces today's cube_stack task exactly.
    _DEFAULT_X_RANGE = (-0.18, 0.18)
    _DEFAULT_Y_RANGE = (-0.12, 0.12)

    def _apply_default_placement_range(self) -> None:
        self.placement_initializer.x_range = list(self._DEFAULT_X_RANGE)
        self.placement_initializer.y_range = list(self._DEFAULT_Y_RANGE)

    def _load_model_with_swap(self, swap_target: str) -> None:
        """Reproduces Stack._load_model, substituting cubeA's class for a visually-similar object.

        Stack builds cubeA/cubeB inline in _load_model rather than via a separate overridable
        hook (unlike PickPlace's _construct_objects), so a shape swap has to re-derive this
        method instead of composing on top of it.
        """
        super(Stack, self)._load_model()

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        swap_cls = SIMILAR_OBJECT_CLASSES[swap_target]
        if swap_cls is CylinderObject:
            self.cubeA = swap_cls(
                name="cubeA",
                size=[self.cube_A_length, self.cube_A_length],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
        else:
            self.cubeA = swap_cls(
                name="cubeA",
                size_min=[self.cube_A_length] * 3,
                size_max=[self.cube_A_length] * 3,
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[self.cube_B_length] * 3,
            size_max=[self.cube_B_length] * 3,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
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
            mujoco_objects=cubes,
        )

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        for obj in (self.cubeA, self.cubeB):
            materials = find_elements(root=obj.asset, tags="material", return_first=False) or []
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
        # Distractors share cubeA/cubeB's sampling region; ensure_valid_placement (set on the
        # sampler above) keeps them from overlapping the task-relevant objects.
        self.placement_initializer.add_objects(self.distractor_objects)

    # ----------------------- Sim references / reset -----------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        self._distractor_body_ids = [
            self.sim.model.body_name2id(obj.root_body) for obj in self.distractor_objects
        ]

    def _reset_internal(self) -> None:
        super()._reset_internal()
        paraphrase = self.variation_config.get("paraphrase_language", False)
        self.language_instruction = (
            str(self.rng.choice(PROMPT_PARAPHRASES)) if paraphrase else CANONICAL_PROMPT
        )

    def get_ep_meta(self) -> dict[str, Any]:
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.language_instruction
        return ep_meta


class FrankaRobosuiteCubesVariantsLowLevel(RobosuiteBaseEnv):
    """FrankaRobosuiteCubesLowLevel, backed by StackVariants instead of plain Stack.

    Identical control/observation interface to FrankaRobosuiteCubesLowLevel (robosuite_cubes.py);
    the only difference is the extra `variation`/`variation_overrides` kwargs forwarded to
    StackVariants for robustness-eval perturbations. `variation="baseline"` reproduces the
    original Stack task exactly.
    """

    _SUBSAMPLE_RATE = 5

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

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = StackVariants(
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
                self.robosuite_env = StackVariants(
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
            self.robosuite_env = StackVariants(
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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
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
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cubeA_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cubeA_quat"], robosuite_obs["cubeA_pos"]])
        )
        cubeB_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cubeB_quat"], robosuite_obs["cubeB_pos"]])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cubeA_robot_base = base_transform @ cubeA_world
        cubeB_robot_base = base_transform @ cubeB_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cubeA_robot_base.translation(), cubeA_robot_base.rotation().wxyz]
                )
            ],
            "secondary": [
                float(x)
                for x in np.concatenate(
                    [cubeB_robot_base.translation(), cubeB_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        return self.robosuite_env.reward(action=None)

    def task_completed(self) -> bool:
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
                np.asarray(pose_dict["secondary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
            "secondary": cube_pose_array[1],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs


__all__ = ["StackVariants", "FrankaRobosuiteCubesVariantsLowLevel", "VARIATIONS"]
