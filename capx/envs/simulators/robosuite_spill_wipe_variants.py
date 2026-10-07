"""Config-driven robustness-eval perturbations of robosuite's Wipe task.

New sibling env only: robosuite's Wipe (capx/third_party/robosuite/robosuite/environments/
manipulation/wipe.py) and capx's FrankaRobosuiteSpillWipeLowLevel (robosuite_spill_wipe.py)
are both left untouched, so existing benchmark configs are unaffected. Mirrors the pattern
used for Stack (StackVariants in robosuite_cubes_variants.py): a `variation` preset selects a
named combination of perturbations, applied on top of the vanilla task.

Wipe is structurally very different from Stack: there are no discrete graspable objects.
Instead, WipeArena (capx/third_party/robosuite/robosuite/models/arenas/wipe_arena.py)
procedurally generates a path of `num_markers` small visual-only CylinderObject "dirt"
markers across the table (a correlated random walk, optionally split into two independent
clusters via `two_clusters`), sharing a single "dirt_mat" CustomMaterial. The robot must wipe
these away with its WipingGripper; reward is the fraction of markers wiped, and marker
positions are (re-)sampled directly into the compiled sim on every reset via
`WipeArena.reset_arena()` -- not just once at XML-build time. Perturbations below account for
this: several mutate `Wipe`/`WipeArena` instance attributes that are only read once, but the
position-bias perturbation has to monkeypatch the arena's own sampling methods so it keeps
applying on every subsequent reset, not just the first one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variant_camera import CAMERA_ORBIT_PRESETS
from capx.envs.simulators.variants_common import VariantMixin

# Wipe has no discrete graspable/target object the way Stack has cubeA/cubeB -- the "target"
# here is the procedurally generated dirt/spill marker path. So there's no natural
# SIMILAR_OBJECT_CLASSES registry to mirror StackVariants' cubeA shape-swap; see
# `target_object_swap` below for how that preset is adapted instead. Distractors, however, map
# straightforwardly: any ordinary physical object placed on the table works as an unrelated,
# never-scored obstacle.
DISTRACTOR_OBJECT_CLASSES = {"box": BoxObject}

CANONICAL_PROMPT = "Wipe all of the dirt off the table with the sponge."
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Clean all the dirt off of the tabletop using the wiping tool.",
    "Use the sponge to remove every bit of dirt from the table surface.",
    "Sweep the wiping tool across the table until no dirt remains.",
]

# Named variation presets. Each maps to a dict of overrides consumed by WipeVariants.
# "baseline" reproduces the original Wipe task and is useful as a control condition when
# measuring policy robustness against the other ten perturbations.
# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt says "the 'brown spill'"; color_tint retints the shared dirt_mat material (e.g.
# color_shift_strong is navy blue), so "brown" goes stale. dirt_pattern_swap (the
# target_object_swap preset) splits the spill into two disconnected clusters while the
# prompt still refers to it as singular -- a softer, grammatical version of the same issue.
VARIATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "color_shift_mild": {"color_tint": (1.0, 0.75, 0.75, 1.0)},
    "color_shift_strong": {"color_tint": (0.15, 0.35, 1.0, 1.0)},
    "object_position_shift": {"object_position_bias": {"bias": (0.6, 0.6), "scale": 0.5}},
    "robot_position_shift": {"robot_base_offset": (0.06, 0.06, 0.0)},
    "lighting_bright": {"light_diffuse_scale": 2.0, "light_ambient_scale": 2.5},
    "lighting_dim": {"light_diffuse_scale": 0.3, "light_ambient_scale": 0.3},
    "distractor_object": {"distractor_classes": ["box"]},
    "target_object_swap": {"dirt_pattern_swap": "two_clusters"},
    "language_paraphrase": {"paraphrase_language": True},
    "compound_hard": {
        "color_tint": (1.0, 0.75, 0.75, 1.0),
        "light_diffuse_scale": 0.5,
        "light_ambient_scale": 0.5,
        "distractor_classes": ["box"],
        "paraphrase_language": True,
    },
    # Difficulty knob, not a robustness perturbation: num_markers/coverage_factor/two_clusters
    # are all plain instance attributes upstream Wipe copies from task_config at __init__ time
    # (self.num_markers, self.coverage_factor, self.two_clusters -- same mechanism
    # dirt_pattern_swap above already uses for two_clusters alone), and reward() is already a
    # wiped/total ratio that adapts to whatever num_markers ends up being, so this is genuinely
    # free like nut_assembly's both_nuts -- no override needed beyond setting the attributes
    # before _load_model builds the WipeArena.
    "harder_spill": {"spill_difficulty": {"num_markers": 260, "coverage_factor": 0.92, "two_clusters": True}},
}


VARIATIONS.update(CAMERA_ORBIT_PRESETS)


class WipeVariants(VariantMixin, Wipe):
    """Wipe with composable, independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba tint applied to the shared "dirt_mat" material used by
          every dirt marker (the closest analogue Wipe has to Stack's cubeA/cubeB material).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the dirt path is generated on the table, instead of the default
          centered coverage region. Implemented by wrapping the arena's own
          `sample_start_pos`/`sample_path_pos` so the bias also applies on every reset (dirt
          positions are resampled into the compiled sim on every `reset_arena()` call, not just
          once at XML-build time).
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to the robot's base
          position.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES to additionally
          place on the table as a fixed, unrelated physical obstacle. Never scored and has no
          wiping role; welded to the table (joints=None), same as the dirt markers themselves.
        - dirt_pattern_swap (str): Wipe has no discrete "object" to swap the way Stack swaps
          cubeA's shape. The closest honest analogue is changing the *shape/distribution* of
          the wipe target itself: the only supported value, "two_clusters", switches the dirt
          pattern from one continuous line to two independently-sampled clusters (mirroring how
          `WipeArena`/`Wipe` already natively support `two_clusters` via task_config). This
          changes the topology of what must be wiped while preserving its role, analogous to
          swapping cubeA's box shape for a cylinder.
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool of
          paraphrases instead of the fixed canonical wording.
        - spill_difficulty (dict with optional "num_markers"/"coverage_factor"/"two_clusters"):
          overrides upstream Wipe's dirt-generation attributes directly. A difficulty knob, not
          a perturbation -- see VARIATIONS["harder_spill"].

    Pick one of the ten named presets in `VARIATIONS` via `variation`, optionally layering
    ad-hoc tweaks via `variation_overrides`.
    """

    PRESETS = VARIATIONS
    OBS_CAMERA = "robot0_robotview"
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
        # dirt_pattern_swap mutates self.two_clusters, which Wipe._load_model reads (via
        # self.task_config-derived instance attrs, set in Wipe.__init__) when constructing a
        # fresh WipeArena below -- so this has to be set *before* calling super()._load_model().
        swap = self.variation_config.get("dirt_pattern_swap")
        if swap is not None:
            if swap != "two_clusters":
                raise ValueError(f"Unknown dirt_pattern_swap '{swap}'. Only 'two_clusters' is supported.")
            self.two_clusters = True

        # Same mechanism as dirt_pattern_swap above -- set the instance attrs Wipe._load_model
        # reads before calling it, rather than passing a merged task_config at construction.
        difficulty = self.variation_config.get("spill_difficulty")
        if difficulty is not None:
            if "num_markers" in difficulty:
                self.num_markers = difficulty["num_markers"]
            if "coverage_factor" in difficulty:
                self.coverage_factor = difficulty["coverage_factor"]
            if "two_clusters" in difficulty:
                self.two_clusters = difficulty["two_clusters"]

        super()._load_model()

        # All mutations below operate on XML elements/samplers already attached to self.model /
        # self.model.mujoco_arena -- the sim isn't compiled from this tree until later in the
        # environment's load/reset sequence, so it's safe to edit them here.
        self._apply_color_tint()
        self._apply_lighting()
        self._apply_robot_offset()
        self._apply_object_position_bias()
        self._add_distractors()

    def _apply_color_tint(self) -> None:
        tint = self.variation_config.get("color_tint")
        if tint is None:
            return
        rgba_str = array_to_string(tint)
        # All dirt markers share a single "dirt_mat" CustomMaterial (constructed once in
        # WipeArena.configure_location and merged once per marker, deduplicated by name), so
        # tinting this one material tints the whole dirt/spill pattern -- Wipe's closest
        # analogue to Stack's cubeA/cubeB material tint.
        material = find_elements(
            root=self.model.mujoco_arena.asset,
            tags="material",
            attribs={"name": "dirt_mat"},
            return_first=True,
        )
        if material is not None:
            material.set("rgba", rgba_str)

    def _apply_object_position_bias(self) -> None:
        pos_cfg = self.variation_config.get("object_position_bias")
        if pos_cfg is None:
            return

        arena = self.model.mujoco_arena
        bias_x, bias_y = pos_cfg.get("bias", (0.0, 0.0))
        scale = pos_cfg.get("scale", 1.0)

        # WipeArena.sample_start_pos/sample_path_pos always sample within a region centered on
        # the table origin, sized by table_half_size * coverage_factor. There's no native way to
        # off-center this region, so we wrap the arena instance's own bound methods: "scale"
        # shrinks the walk toward the center, "bias" then re-centers it. This has to be done by
        # wrapping (not by reimplementing WipeArena) because marker positions are resampled into
        # the compiled sim on *every* reset via WipeArena.reset_arena(), not just once here.
        max_x = arena.table_half_size[0] * arena.coverage_factor - arena.line_width / 2
        max_y = arena.table_half_size[1] * arena.coverage_factor - arena.line_width / 2
        offset = np.array([bias_x * max_x * (1.0 - scale), bias_y * max_y * (1.0 - scale)])

        orig_sample_start_pos = arena.sample_start_pos
        orig_sample_path_pos = arena.sample_path_pos

        def biased_start_pos() -> np.ndarray:
            pos = np.asarray(orig_sample_start_pos(), dtype=np.float64) * scale + offset
            pos[0] = float(np.clip(pos[0], -max_x, max_x))
            pos[1] = float(np.clip(pos[1], -max_y, max_y))
            return pos

        def biased_path_pos(pos: np.ndarray) -> np.ndarray:
            # Undo the affine map to recover the arena's own "unbiased" coordinate before
            # calling through, so its internal correlated-random-walk state (self.direction)
            # and boundary checks stay correct, then re-apply the map to the result.
            unbiased = (np.asarray(pos, dtype=np.float64) - offset) / scale if scale else np.asarray(pos) - offset
            new_pos = np.asarray(orig_sample_path_pos(unbiased), dtype=np.float64) * scale + offset
            new_pos[0] = float(np.clip(new_pos[0], -max_x, max_x))
            new_pos[1] = float(np.clip(new_pos[1], -max_y, max_y))
            return new_pos

        arena.sample_start_pos = biased_start_pos
        arena.sample_path_pos = biased_path_pos

    def _add_distractors(self) -> None:
        distractor_names = self.variation_config.get("distractor_classes", [])
        self.distractor_objects = []
        if not distractor_names:
            return

        arena = self.model.mujoco_arena
        table_body = find_elements(
            root=arena.worldbody, tags="body", attribs={"name": "table"}, return_first=True
        )
        for i, name in enumerate(distractor_names):
            obj_cls = DISTRACTOR_OBJECT_CLASSES[name]
            half_size = np.array([0.025, 0.025, 0.02])
            obj = obj_cls(
                name=f"Distractor{name}{i}",
                size=half_size.tolist(),
                rgba=[0.15, 0.25, 0.85, 1.0],
                joints=None,
            )
            # Fixed corner position, expressed in the table body's own local frame (same frame
            # dirt markers use for their body_pos), away from the centered dirt-sampling region
            # so it reads as an unrelated obstacle rather than part of the task.
            obj_body = obj.get_obj()
            obj_body.set(
                "pos",
                array_to_string(
                    np.array(
                        [
                            0.7 * arena.table_half_size[0],
                            0.7 * arena.table_half_size[1],
                            arena.table_half_size[2] + half_size[2],
                        ]
                    )
                ),
            )
            arena.merge_assets(obj)
            table_body.append(obj_body)
            self.distractor_objects.append(obj)

    # ----------------------- Sim references / reset -----------------------


class FrankaRobosuiteSpillWipeVariantsLowLevel(RobosuiteBaseEnv):
    """FrankaRobosuiteSpillWipeLowLevel, backed by WipeVariants instead of plain Wipe.

    Identical control/observation interface to FrankaRobosuiteSpillWipeLowLevel
    (robosuite_spill_wipe.py); the only difference is the extra `variation`/
    `variation_overrides` kwargs forwarded to WipeVariants for robustness-eval perturbations,
    and that `task_prompt` is sourced from `get_ep_meta()["lang"]` (a task-appropriate wiping
    prompt, with paraphrase support) rather than the original file's hardcoded string -- which
    was actually a leftover cube-stacking prompt, not a Wipe-specific one.
    `variation="baseline"` reproduces the original Wipe task exactly.
    """

    _SUBSAMPLE_RATE = 10
    _ACTION_SLICE = -2  # Wipe env uses action[:-2] instead of action[:-1]

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 4000,
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
                self.robosuite_env = WipeVariants(
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
                self.robosuite_env = WipeVariants(
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
            self.robosuite_env = WipeVariants(
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
        # Additional viser attributes specific to spill_wipe
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

        self._sim_step_count = 0

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {"task_prompt": self.robosuite_env.get_ep_meta()["lang"]}
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._sim_step_count += 1
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._sim_step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.005, max_steps: int = 10
    ) -> None:
        """Move to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        # Override with spill_wipe-specific defaults
        super().move_to_joints_blocking(joints, tolerance=tolerance, max_steps=max_steps)

    def compute_reward(self) -> float:
        """Compute dense spill wipe reward."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()

        self._process_camera_observations(robosuite_obs)

        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        robosuite_obs["robot_joint_pos"] = np.concatenate(
            [
                robosuite_obs["robot0_joint_pos"],
            ]
        )
        robosuite_obs["robot_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base.translation(),
                gripper_robot_base.rotation().wxyz,
            ]
        )

        return robosuite_obs

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
            del self._frame_buffer
            self._frame_buffer = []
        return frames

    # Override viser update to handle the different obs_cartesian slicing (no gripper in cartesian)

    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if self.viser_server is not None:
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            from capx.utils.camera_utils import obs_get_rgb
            from capx.utils.depth_utils import depth_color_to_pointcloud

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
                    "cube_frame",
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


__all__ = ["WipeVariants", "FrankaRobosuiteSpillWipeVariantsLowLevel", "VARIATIONS"]
