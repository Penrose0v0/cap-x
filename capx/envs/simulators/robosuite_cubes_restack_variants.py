"""Config-driven robustness-eval perturbations of robosuite's cube_restack task.

New sibling env only: robosuite's Stack (capx/third_party/robosuite/robosuite/environments/
manipulation/stack.py) and capx's FrankaRobosuiteCubesRestackLowLevel
(robosuite_cubes_restack.py) are both left untouched, so existing benchmark configs are
unaffected. Mirrors the pattern already validated for cube_stack (StackVariants,
robosuite_cubes_variants.py): a `variation` preset selects a named combination of
perturbations, applied on top of the vanilla task.

cube_restack reuses robosuite's Stack class exactly like cube_stack does, but capx's
FrankaRobosuiteCubesRestackLowLevel overrides Stack's own cube objects and
placement_initializer after construction with:
  - plain (untextured, rgba-only) BoxObjects instead of Stack's own CustomMaterial-textured
    cubes -- there is no <material> element to find/mutate for a color tint the way
    StackVariants does, so color has to be set directly on cube construction instead.
  - a custom StackedObjectRandomSampler (copied here from robosuite_cubes_restack.py
    unchanged) that places cubeA and cubeB PRE-STACKED (cubeB directly on top of cubeA) at
    reset, rather than side-by-side like plain Stack.
  - reward()/task_completed() add a "secondary check" (in the capx wrapper, not the
    robosuite env) that zeroes out success/reward if BOTH cubes are elevated above the
    table (z > 0.04) -- i.e. the task requires the cubes to end up re-stacked resting on
    the table, not just held together mid-air.

Since Stack's own default cube construction gets discarded regardless (capx's existing
wrapper always replaces it), _load_model here fully re-derives Stack's arena/robot
construction and builds restack's own cube objects + sampler directly, rather than
composing on top of Stack's default cube-building the way StackVariants does for plain
cube_stack.
"""

from __future__ import annotations

import collections.abc
from copy import copy
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
from robosuite.utils import RandomizationError
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.transform_utils import quat_multiply

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.variant_camera import (
    CAMERA_ORBIT_PRESETS,
    apply_camera_orbit,
    capture_camera_reference,
)
from capx.envs.simulators.variants_common import resolve_variation_config


class StackedObjectRandomSampler(ObjectPositionSampler):
    """Places the first two objects pre-stacked (second directly on top of the first),
    then any remaining objects independently in the same region.

    Copied unchanged from capx/envs/simulators/robosuite_cubes_restack.py -- this is the
    exact sampler capx's existing cube_restack task relies on for its "cubes start already
    stacked" initial condition; duplicated here rather than imported so this file has no
    import-time dependency on the non-variant module.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        rng=None,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
            rng=rng,
        )

    def _sample_x(self, object_horizontal_radius):
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        if self.rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = self.rng.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            raise ValueError(
                f"Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {self.rotation_axis}"
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert reference in placed_objects, (
                f"Invalid reference received. Current options are: {placed_objects.keys()}, requested: {reference}"
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, (
                f"Invalid reference received. Should be (x,y,z) 3-tuple, but got: {base_offset}"
            )

        def _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
            if not self.ensure_valid_placement:
                return True
            for (x, y, z), _, other_obj in placed_objects.values():
                if (
                    np.linalg.norm((object_x - x, object_y - y))
                    <= other_obj.horizontal_radius + horizontal_radius
                ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                    return False
            return True

        mujoco_objects = list(self.mujoco_objects)

        if len(mujoco_objects) >= 2:
            first_obj = mujoco_objects[0]
            second_obj = mujoco_objects[1]

            assert first_obj.name not in placed_objects, (
                f"Object '{first_obj.name}' has already been sampled!"
            )
            assert second_obj.name not in placed_objects, (
                f"Object '{second_obj.name}' has already been sampled!"
            )

            first_hr = first_obj.horizontal_radius
            first_bo = first_obj.bottom_offset
            second_hr = second_obj.horizontal_radius
            second_bo = second_obj.bottom_offset

            success = False
            for _ in range(5000):
                object_x = self._sample_x(first_hr) + base_offset[0]
                object_y = self._sample_y(first_hr) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= first_bo[-1]

                if not _location_valid(object_x, object_y, object_z, first_hr, first_bo):
                    continue

                quat1 = self._sample_quat()
                if hasattr(first_obj, "init_quat"):
                    quat1 = quat_multiply(quat1, first_obj.init_quat)
                pos1 = (object_x, object_y, object_z)
                placed_objects[first_obj.name] = (pos1, quat1, first_obj)

                z_gap = max(float(self.z_offset), 0.01)
                object2_x = object_x - 0.01
                object2_y = object_y
                object2_z = object_z + first_obj.top_offset[-1] - second_bo[-1] + z_gap

                if not _location_valid(object2_x, object2_y, object2_z, second_hr, second_bo):
                    placed_objects.pop(first_obj.name, None)
                    continue

                quat2 = self._sample_quat()
                if hasattr(second_obj, "init_quat"):
                    quat2 = quat_multiply(quat2, second_obj.init_quat)
                pos2 = (object2_x, object2_y, object2_z)
                placed_objects[second_obj.name] = (pos2, quat2, second_obj)

                success = True
                break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            remaining_objects = mujoco_objects[2:]
        else:
            remaining_objects = mujoco_objects

        for obj in remaining_objects:
            assert obj.name not in placed_objects, f"Object '{obj.name}' has already been sampled!"

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for _ in range(5000):
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                if _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
                    quat = self._sample_quat()
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


# Object classes usable as a visually-similar swap for cubeA, or as task-irrelevant
# distractors (same choice as StackVariants).
SIMILAR_OBJECT_CLASSES = {"cylinder": CylinderObject}
DISTRACTOR_OBJECT_CLASSES = {"cylinder": CylinderObject}

CANONICAL_PROMPT = "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
PROMPT_PARAPHRASES = [
    CANONICAL_PROMPT,
    "Stack the first cube on top of the second cube. Quaternions are WXYZ.",
    "Put the primary block on top of the other block. Quaternions are WXYZ.",
    "Set the first cube down onto the second cube. Quaternions are WXYZ.",
]

# Matches FrankaRobosuiteCubesRestackLowLevel's own sampling range
# (robosuite_cubes_restack.py), which differs from cube_stack's own default.
_DEFAULT_X_RANGE = (-0.12, 0.16)
_DEFAULT_Y_RANGE = (-0.12, 0.12)

# KNOWN ISSUE (documented, not fixed -- see variants_common.py's module docstring): the
# prompt says "red cube"/"green cube"; color_tint sets the SAME rgba on both cubeA and
# cubeB (see rgba_a/rgba_b below), so neither word stays true and the color is no longer
# distinguishing. target_object_swap turns cubeA into a cylinder ("cube" goes stale too).
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
    # Difficulty knob, not a robustness perturbation -- same idea as StackVariants'
    # stack_3_cubes (robosuite_cubes_variants.py), adapted for the extra wrapper-level
    # "secondary check" this task has (see FrankaRobosuiteCubesRestackVariantsLowLevel):
    # cubeA on cubeB on cubeC, base (cubeC) must rest on the table for success to count.
    "stack_3_cubes": {"stack_3_cubes": True},
}


VARIATIONS.update(CAMERA_ORBIT_PRESETS)


class RestackVariants(Stack):
    """cube_restack (Stack + pre-stacked initial condition) with composable,
    independently-toggleable perturbations for robustness eval.

        - color_tint (4-tuple): rgba applied directly to cubeA/cubeB at construction (these
          are plain untextured BoxObjects, unlike cube_stack's CustomMaterial cubes, so there
          is no <material> element to mutate after the fact).
        - object_position_bias (dict with "bias" 2-tuple in [-1, 1] and "scale" float): biases
          and rescales where the (still pre-stacked) cubes are sampled, instead of the
          default `_DEFAULT_X_RANGE`/`_DEFAULT_Y_RANGE` region.
        - robot_base_offset (3-tuple): additional (x, y, z) offset applied to the robot's base.
        - light_diffuse_scale / light_ambient_scale (float): multiplicative scale on the arena
          light's diffuse/ambient brightness.
        - distractor_classes (list of str): keys of DISTRACTOR_OBJECT_CLASSES placed in the
          same region as cubeA/cubeB (StackedObjectRandomSampler samples any object beyond
          the first two independently, so no separate sub-sampler region is needed here,
          unlike several of the other task variants).
        - swap_target (str): key of SIMILAR_OBJECT_CLASSES; replaces cubeA's shape.
        - paraphrase_language (bool): if True, `get_ep_meta()["lang"]` is sampled from a pool
          of paraphrases instead of the fixed canonical wording.
        - stack_3_cubes (bool): if True, adds a third cube (cubeC, blue) placed independently
          of the pre-stacked cubeA/cubeB pair; success/reward require a full 3-cube tower
          (cubeA on cubeB on cubeC). A difficulty knob, not a perturbation -- see
          `reward()`/`_check_success()` overrides below and
          FrankaRobosuiteCubesRestackVariantsLowLevel's wrapper-level "secondary check".

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
        self.cubeC: Any = None
        self.language_instruction: str | None = None

        kwargs.setdefault("cube_A_length", 0.02)
        kwargs.setdefault("cube_B_length", 0.02)
        super().__init__(**kwargs)

    # ----------------------- Scene construction -----------------------

    def _load_model(self) -> None:
        """Fully re-derives Stack's arena/robot construction with restack's own cube
        objects and StackedObjectRandomSampler -- Stack's own default cube-building (called
        via a plain super()._load_model()) gets discarded regardless by capx's existing
        cube_restack wrapper, so this skips straight past it rather than composing on top,
        same as StackVariants._load_model_with_swap does for the swap-only case.
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

        tint = self.variation_config.get("color_tint")
        rgba_a = list(tint) if tint is not None else [1, 0, 0, 1]
        rgba_b = list(tint) if tint is not None else [0, 1, 0, 1]

        swap_target = self.variation_config.get("swap_target")
        swap_cls = SIMILAR_OBJECT_CLASSES[swap_target] if swap_target else None
        if swap_cls is CylinderObject:
            self.cubeA = CylinderObject(
                name="cubeA", size=[self.cube_A_length, self.cube_A_length], rgba=rgba_a
            )
        else:
            self.cubeA = BoxObject(name="cubeA", size=[self.cube_A_length] * 3, rgba=rgba_a)
        self.cubeB = BoxObject(name="cubeB", size=[self.cube_B_length] * 3, rgba=rgba_b)

        cubes = [self.cubeA, self.cubeB]
        if self.variation_config.get("stack_3_cubes"):
            # StackedObjectRandomSampler only pre-stacks the first two objects it's given
            # (cubeA on cubeB) and places anything beyond that independently (see its own
            # docstring/the module docstring above) -- so cubeC rides along as a 3rd,
            # separately-placed object for free, same mechanism _add_distractors already
            # relies on below for distractor objects, just included here instead so it's a
            # real scoreable object rather than a never-scored one.
            self.cubeC = BoxObject(name="cubeC", size=[self.cube_A_length] * 3, rgba=[0, 0, 1, 1])
            cubes.append(self.cubeC)
        else:
            self.cubeC = None

        self.placement_initializer = StackedObjectRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=list(_DEFAULT_X_RANGE),
            y_range=list(_DEFAULT_Y_RANGE),
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

        # All mutations below operate on XML elements/samplers already attached to self.model /
        # self.placement_initializer -- the sim isn't compiled from this tree until later in the
        # environment's load/reset sequence, so it's safe to edit them here.
        self._apply_lighting()
        self._apply_robot_offset()
        self._apply_object_position_bias()
        self._add_distractors()

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
        bias_x, bias_y = pos_cfg.get("bias", (0.0, 0.0))
        scale = pos_cfg.get("scale", 1.0)
        # _DEFAULT_X_RANGE/_DEFAULT_Y_RANGE aren't centered on 0 (e.g. x is [-0.12, 0.16]), so
        # bias/scale are applied around each axis's own center rather than assuming symmetry
        # (same generalized approach as NutAssemblyVariants' per-nut sampler ranges).
        for axis_range, bias, attr in (
            (_DEFAULT_X_RANGE, bias_x, "x_range"),
            (_DEFAULT_Y_RANGE, bias_y, "y_range"),
        ):
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
        # Unlike several of the other task variants, no disjoint sub-sampler is needed here:
        # StackedObjectRandomSampler only special-cases the first two objects (pre-stacking
        # them); any objects beyond that are sampled independently within the same region, so
        # a distractor can simply be added to the same sampler.
        self.placement_initializer.add_objects(self.distractor_objects)

    # ----------------------- Sim references / reset -----------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        self._distractor_body_ids = [
            self.sim.model.body_name2id(obj.root_body) for obj in self.distractor_objects
        ]
        self._camera_orbit_ref = capture_camera_reference(self.sim, "robot0_robotview")
        self.cubeC_body_id = (
            self.sim.model.body_name2id(self.cubeC.root_body) if self.cubeC is not None else None
        )

    def _reset_internal(self) -> None:
        super()._reset_internal()
        apply_camera_orbit(self.sim, self._camera_orbit_ref, self.variation_config.get("camera_orbit"))
        paraphrase = self.variation_config.get("paraphrase_language", False)
        self.language_instruction = (
            str(self.rng.choice(PROMPT_PARAPHRASES)) if paraphrase else CANONICAL_PROMPT
        )

    def get_ep_meta(self) -> dict[str, Any]:
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.language_instruction
        return ep_meta

    # ----------------------- Difficulty override -----------------------

    def _check_success(self) -> bool:
        """3-cube tower success: cubeA resting on cubeB AND cubeB resting on cubeC (same
        contact + not-grasping check as StackVariants' stack_3_cubes). See
        VARIATIONS["stack_3_cubes"]. The wrapper-level "secondary check"
        (FrankaRobosuiteCubesRestackVariantsLowLevel.task_completed) still separately verifies
        the whole tower is resting on the table, not held in the air."""
        if not self.variation_config.get("stack_3_cubes") or self.cubeC is None:
            return super()._check_success()

        grasping_a = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        grasping_b = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        a_on_b = self.check_contact(self.cubeA, self.cubeB) and not grasping_a
        b_on_c = self.check_contact(self.cubeB, self.cubeC) and not grasping_b
        return a_on_b and b_on_c

    def reward(self, action=None) -> float:
        """Simplified 2-stage reward for the 3-cube tower, matching StackVariants'
        stack_3_cubes: 0.5 once cubeB is resting on cubeC, 1.0 once cubeA is also resting on
        cubeB."""
        if not self.variation_config.get("stack_3_cubes") or self.cubeC is None:
            return super().reward(action)

        grasping_a = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        grasping_b = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        a_on_b = self.check_contact(self.cubeA, self.cubeB) and not grasping_a
        b_on_c = self.check_contact(self.cubeB, self.cubeC) and not grasping_b

        if a_on_b and b_on_c:
            return 1.0
        if b_on_c:
            return 0.5
        return 0.0


class FrankaRobosuiteCubesRestackVariantsLowLevel(RobosuiteBaseEnv):
    """FrankaRobosuiteCubesRestackLowLevel, backed by RestackVariants instead of a plain
    Stack + post-hoc cube/sampler override.

    Identical control/observation/reward interface to FrankaRobosuiteCubesRestackLowLevel
    (robosuite_cubes_restack.py), including its "secondary check" reward/success gating
    (success requires the cubes stacked AND resting on the table, not held mid-air); the
    only difference is the extra `variation`/`variation_overrides` kwargs forwarded to
    RestackVariants. `variation="baseline"` reproduces the original cube_restack task
    exactly. Unlike the original wrapper, this doesn't need to override cubeA/cubeB/
    placement_initializer after construction -- RestackVariants already builds them
    correctly inside its own _load_model, which capx's original wrapper's post-reset
    cubeA/cubeB re-creation (robosuite_cubes_restack.py's reset()) has no effect beyond
    (it reassigns the attribute after the sim is already compiled, so it never actually
    changes the running scene); that step is intentionally not replicated here since it's
    a no-op.
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

        self.cube_A_length = 0.02
        self.cube_B_length = 0.02

        common_kwargs: dict[str, Any] = dict(
            variation=variation,
            variation_overrides=variation_overrides,
            robots=["Panda"],
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
            cube_A_length=self.cube_A_length,
            cube_B_length=self.cube_B_length,
        )

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = RestackVariants(
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
                self.robosuite_env = RestackVariants(
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
            self.robosuite_env = RestackVariants(
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

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

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

        pose_dict = {
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

        cubeC_body_id = getattr(self.robosuite_env, "cubeC_body_id", None)
        if cubeC_body_id is not None:
            # cubeC has no registered observable sensor (upstream Stack._setup_observables is
            # hardcoded to cubeA/cubeB only), so read its pose directly from sim data instead
            # of the observation dict -- same world-frame source _get_observations() itself
            # reads cubeA_pos/cubeB_pos from, just not routed through an observable.
            cubeC_world = vtf.SE3(
                wxyz_xyz=np.concatenate(
                    [
                        self.robosuite_env.sim.data.xquat[cubeC_body_id],
                        self.robosuite_env.sim.data.xpos[cubeC_body_id],
                    ]
                )
            )
            cubeC_robot_base = base_transform @ cubeC_world
            pose_dict["tertiary"] = [
                float(x)
                for x in np.concatenate(
                    [cubeC_robot_base.translation(), cubeC_robot_base.rotation().wxyz]
                )
            ]

        return pose_dict

    def compute_reward(self) -> float:
        """Dense stacking reward, zeroed if the tower is held elevated rather than resting on
        the table (verbatim from FrankaRobosuiteCubesRestackLowLevel.compute_reward for the
        2-cube case). For stack_3_cubes, "resting" means the base (cubeC) isn't elevated --
        cubeA/cubeB being elevated is expected once they're stacked on top of it, so checking
        all three the same way the 2-cube case checks both would always fail."""
        if self.robosuite_env.reward(action=None) > 0.4:
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            if "tertiary" in pose_dict:
                if pose_dict["tertiary"][2] > 0.04:
                    return 0.0
                else:
                    return self.robosuite_env.reward(action=None)
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return 0.0
            else:
                return self.robosuite_env.reward(action=None)
        else:
            return self.robosuite_env.reward(action=None)

    def task_completed(self) -> bool:
        """Verbatim from FrankaRobosuiteCubesRestackLowLevel.task_completed for the 2-cube
        case; see compute_reward()'s docstring for why stack_3_cubes checks only the base
        (cubeC)."""
        if self.robosuite_env._check_success():
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            if "tertiary" in pose_dict:
                return pose_dict["tertiary"][2] <= 0.04
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return False
            else:
                return True
        else:
            return False

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


__all__ = [
    "RestackVariants",
    "FrankaRobosuiteCubesRestackVariantsLowLevel",
    "VARIATIONS",
]
