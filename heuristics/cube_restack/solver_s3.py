"""Deterministic S3 solver for ``cube_restack``.

S3 exposes perception and low-level control primitives, not privileged object
poses or Cartesian ``goto_pose``.  This program therefore implements the full
hand-written pipeline:

    RGB-D -> colour mask / 3-D OBB -> Contact-GraspNet -> IK -> joint motion

The task-specific state machine first moves the green cube off the red cube,
then places the red cube on the green cube.  No LLM or VLM is queried.

The CaP-X executor injects the S3 API functions used below into ``__main__``.
Keeping execution behind ``main`` also makes the geometry helpers importable by
lightweight tests without starting a simulator or model server.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation
from heuristics.common import *  # noqa: F401,F403 -- inlined by static_solver_server


CAMERA_NAME = "robot0_robotview"

# The competition task is known in advance.  Both Robosuite cubes have a
# 4-cm side.  Perceived OBB extents are still computed and logged, while this
# calibrated dimension prevents partial-view OBBs from changing release height.
CUBE_SIDE_M = 0.040
APPROACH_DISTANCE_M = 0.090
LIFT_DISTANCE_M = 0.120
TRANSFER_CLEARANCE_M = 0.100
PARK_DISTANCE_M = 0.150
RELEASE_CLEARANCE_M = 0.001

MIN_MASK_PIXELS = 80
MIN_TOP_DOWN_COSINE = 0.55
TOP_DOWN_QUATERNION_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

_artifact_dir_value = os.environ.get("CAPX_HEURISTIC_ARTIFACT_DIR")
ARTIFACT_DIR = Path(_artifact_dir_value) if _artifact_dir_value else None
TRACE: list[dict[str, Any]] = []



record = make_recorder("cube_restack", TRACE, ARTIFACT_DIR)



def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small dependency-free binary erosion used to reject depth-edge bleed."""

    original = np.asarray(mask, dtype=bool)
    result = original.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(result)
        for row_offset in range(3):
            for col_offset in range(3):
                eroded &= padded[
                    row_offset : row_offset + result.shape[0],
                    col_offset : col_offset + result.shape[1],
                ]
        if int(eroded.sum()) < MIN_MASK_PIXELS:
            break
        result = eroded
    return result


def _colour_dominance(rgb_pixels: np.ndarray, colour: str) -> float:
    if len(rgb_pixels) == 0:
        return -1.0
    pixels = np.asarray(rgb_pixels, dtype=np.float64)
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    if colour == "green":
        values = green - np.maximum(red, blue)
    elif colour == "red":
        values = red - np.maximum(green, blue)
    else:
        values = np.zeros(len(pixels), dtype=np.float64)
    return float(np.median(values) / 255.0)


def colour_component_detections(rgb: np.ndarray, colour: str) -> list[dict[str, Any]]:
    """Segment task-known red/green cubes without a learned detector.

    Robosuite renders the two cubes with fixed saturated material colours.  A
    channel-dominance threshold followed by connected components is therefore
    a deterministic S3 BBox/mask estimator and needs no gated model weights.
    """

    image = np.asarray(rgb, dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if colour == "green":
        target = green
        dominance = green - np.maximum(red, blue)
    elif colour == "red":
        target = red
        dominance = red - np.maximum(green, blue)
    else:
        raise ValueError(f"Unsupported cube colour: {colour}")

    raw_mask = (target > 45) & (dominance > 18)
    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndimage.binary_closing(raw_mask, structure=structure, iterations=2)
    cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=1)
    labels, component_count = ndimage.label(cleaned)

    detections: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels < MIN_MASK_PIXELS:
            continue
        rows, cols = np.nonzero(component)
        component_dominance = float(np.median(dominance[component]) / 255.0)
        detections.append(
            {
                "mask": component,
                "box": [
                    float(cols.min()),
                    float(rows.min()),
                    float(cols.max() + 1),
                    float(rows.max() + 1),
                ],
                "score": float(np.clip(component_dominance, 0.0, 1.0)),
                "label": f"{colour} cube (RGB heuristic)",
            }
        )
    return detections


def select_detection(
    results: list[dict[str, Any]], rgb: np.ndarray, colour: str
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    """Select a component using detector confidence plus cube colour purity."""

    candidates: list[tuple[float, dict[str, Any], np.ndarray, dict[str, float]]] = []
    image_shape = np.asarray(rgb).shape[:2]
    for result in results:
        mask = np.asarray(result.get("mask"), dtype=bool).squeeze()
        if mask.shape != image_shape:
            continue
        pixels = int(mask.sum())
        if pixels < MIN_MASK_PIXELS:
            continue
        confidence = float(result.get("score", 0.0))
        dominance = _colour_dominance(np.asarray(rgb)[mask], colour)
        # Confidence remains the main term; colour purity disambiguates the two
        # adjacent cubes and suppresses table/shadow components.
        rank = confidence + 0.60 * dominance
        metrics = {
            "detector_score": confidence,
            "colour_dominance": dominance,
            "rank": rank,
            "mask_pixels": float(pixels),
        }
        candidates.append((rank, result, mask, metrics))

    if not candidates:
        raise RuntimeError(f"RGB heuristic produced no usable {colour}-cube mask")
    _, result, mask, metrics = max(candidates, key=lambda item: item[0])
    return result, mask, metrics


def deproject_mask(
    depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-frame points and the corresponding valid image mask."""

    depth_image = _depth_2d(depth)
    binary_mask = np.asarray(mask, dtype=bool)
    valid = (
        binary_mask
        & np.isfinite(depth_image)
        & (depth_image > 0.05)
        & (depth_image < 3.0)
    )
    rows, cols = np.nonzero(valid)
    if len(rows) < MIN_MASK_PIXELS:
        raise RuntimeError(f"Only {len(rows)} valid RGB-D pixels remain in cube mask")

    z = depth_image[rows, cols]
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    x = (cols - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (rows - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    return np.column_stack([x, y, z]), valid



def estimate_cube_geometry(
    points_camera: np.ndarray, camera_to_base: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Estimate cube center/top from visible surfaces and compute an S3 OBB."""

    # This API call is part of the S3 contract.  The OBB is useful for XY and
    # extent diagnostics, but a partial view can underestimate a cube axis.
    # Computing the OBB in the robot-base frame makes its axes/extents easier
    # to interpret in the human trace and keeps all following geometry in one
    # frame.
    points_base = transform_points(camera_to_base, points_camera)
    obb = get_oriented_bounding_box_from_3d_points(points_base)
    obb_center_base = np.asarray(obb["center"], dtype=np.float64)

    # The upper quantile robustly identifies the horizontal top face.  Its
    # trimmed XY bounds are less biased than a centroid containing side faces.
    top_z = float(np.quantile(points_base[:, 2], 0.90))
    top_band = points_base[points_base[:, 2] >= top_z - 0.004]
    if len(top_band) >= 20:
        low_xy = np.quantile(top_band[:, :2], 0.05, axis=0)
        high_xy = np.quantile(top_band[:, :2], 0.95, axis=0)
        center_xy = 0.5 * (low_xy + high_xy)
    else:
        center_xy = obb_center_base[:2]

    center_base = np.array(
        [center_xy[0], center_xy[1], top_z - 0.5 * CUBE_SIDE_M], dtype=np.float64
    )
    return {
        "center": center_base,
        "top_z": top_z,
        "obb_center": obb_center_base,
        "obb_extent": np.asarray(obb["extent"], dtype=np.float64),
        "points_base": points_base,
    }



def save_detection_overlay(
    phase: str,
    colour: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    detection: dict[str, Any],
    geometry: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    image = np.asarray(rgb, dtype=np.uint8).copy()
    tint = np.array([255, 60, 60] if colour == "red" else [40, 255, 80], dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    image[mask_bool] = np.clip(0.55 * image[mask_bool] + 0.45 * tint, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    box = detection.get("box")
    if box is not None and len(box) == 4:
        draw.rectangle(tuple(float(value) for value in box), outline="yellow", width=3)
    center = np.asarray(geometry["center"])
    label = (
        f"{phase}: {colour} cube  score={metrics['detector_score']:.3f}\n"
        f"center={np.round(center, 4).tolist()}  top_z={geometry['top_z']:.4f}"
    )
    draw.rectangle((4, 4, min(canvas.width - 4, 505), 42), fill=(0, 0, 0))
    draw.text((8, 7), label, fill="white")
    canvas.save(ARTIFACT_DIR / f"{_safe_name(phase)}_{colour}_perception.png")


def perceive_cube(colour: str, phase: str) -> dict[str, Any]:
    observation = get_observation()
    camera = observation[CAMERA_NAME]
    rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
    depth = _depth_2d(camera["images"]["depth"])
    results = colour_component_detections(rgb, colour)
    detection, mask, metrics = select_detection(results, rgb, colour)
    interior_mask = erode_mask(mask, iterations=1)
    points_camera, _ = deproject_mask(depth, camera["intrinsics"], interior_mask)
    geometry = estimate_cube_geometry(points_camera, camera["pose_mat"])

    result = {
        "colour": colour,
        "observation": observation,
        "camera": camera,
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "detection": detection,
        "metrics": metrics,
        **geometry,
    }
    record(
        "perception",
        phase=phase,
        colour=colour,
        center=result["center"],
        top_z=result["top_z"],
        obb_extent=result["obb_extent"],
        detector="rgb_colour_component",
        detector_score=metrics["detector_score"],
        mask_pixels=int(mask.sum()),
    )
    save_detection_overlay(phase, colour, rgb, mask, detection, result, metrics)
    return result


def estimate_table_height(observation: dict[str, Any], fallback: float) -> float:
    """Estimate the horizontal table plane as the dominant workspace Z mode."""

    camera = observation[CAMERA_NAME]
    depth = _depth_2d(camera["images"]["depth"])
    all_mask = np.ones(depth.shape, dtype=bool)
    try:
        points_camera, _ = deproject_mask(depth, camera["intrinsics"], all_mask)
        points_base = transform_points(camera["pose_mat"], points_camera)
        workspace = points_base[
            (points_base[:, 0] > 0.20)
            & (points_base[:, 0] < 0.85)
            & (np.abs(points_base[:, 1]) < 0.45)
            & (points_base[:, 2] > -0.30)
            & (points_base[:, 2] < 0.10)
        ]
        if len(workspace) < 1000:
            raise RuntimeError("too few table candidates")
        edges = np.arange(-0.30, 0.101, 0.001)
        counts, edges = np.histogram(workspace[:, 2], bins=edges)
        peak = int(np.argmax(counts))
        midpoint = 0.5 * (edges[peak] + edges[peak + 1])
        support = workspace[np.abs(workspace[:, 2] - midpoint) < 0.002, 2]
        if len(support) < 500:
            raise RuntimeError("table mode has too little support")
        height = float(np.median(support))
    except Exception as exc:
        height = float(fallback)
        record("table_fallback", error=repr(exc), height=height)
    record("table_height", height=height)
    return height



def rank_grasp_candidates(
    grasp_poses_camera: np.ndarray,
    grasp_scores: np.ndarray,
    camera_to_base: np.ndarray,
    object_center: np.ndarray,
) -> list[dict[str, Any]]:
    """Transform and rank reachable, mostly top-down Contact-GraspNet poses."""

    poses = np.asarray(grasp_poses_camera, dtype=np.float64)
    scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
        return []

    candidates: list[dict[str, Any]] = []
    camera_transform = np.asarray(camera_to_base, dtype=np.float64)
    center = np.asarray(object_center, dtype=np.float64)
    for index, (pose_camera, score) in enumerate(zip(poses, scores, strict=True)):
        pose_base = camera_transform @ pose_camera
        position = pose_base[:3, 3]
        approach_axis = pose_base[:3, 2]
        downness = float(-approach_axis[2])
        center_distance = float(np.linalg.norm(position - center))
        finite = np.isfinite(pose_base).all() and np.isfinite(score)
        reachable = 0.18 < position[0] < 0.85 and abs(position[1]) < 0.45
        if not finite or not reachable or downness < MIN_TOP_DOWN_COSINE:
            continue
        rank = float(score) + 0.35 * downness - 1.5 * max(center_distance - 0.025, 0.0)
        candidates.append(
            {
                "index": index,
                "pose": pose_base,
                "position": position,
                "quaternion": matrix_to_quaternion_wxyz(pose_base[:3, :3]),
                "approach_axis": approach_axis,
                "score": float(score),
                "downness": downness,
                "center_distance": center_distance,
                "rank": rank,
                "source": "contact_graspnet",
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["rank"], reverse=True)


def analytic_top_down_grasp(object_center: np.ndarray) -> dict[str, Any]:
    """Deterministic fallback if Contact-GraspNet has no usable candidate."""

    rotation = Rotation.from_quat([0.0, 1.0, 0.0, 0.0]).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(object_center, dtype=np.float64)
    return {
        "index": -1,
        "pose": pose,
        "position": pose[:3, 3],
        "quaternion": TOP_DOWN_QUATERNION_WXYZ.copy(),
        "approach_axis": pose[:3, 2],
        "score": 0.0,
        "downness": 1.0,
        "center_distance": 0.0,
        "rank": 0.0,
        "source": "analytic_fallback",
    }


def plan_cube_grasp(perception: dict[str, Any]) -> dict[str, Any]:
    try:
        poses, scores = plan_grasp(
            depth=perception["depth"],
            intrinsics=perception["camera"]["intrinsics"],
            segmentation=perception["mask"].astype(np.int32),
        )
        candidates = rank_grasp_candidates(
            poses,
            scores,
            perception["camera"]["pose_mat"],
            perception["center"],
        )
    except Exception as exc:
        record("graspnet_error", colour=perception["colour"], error=repr(exc))
        candidates = []

    grasp = candidates[0] if candidates else analytic_top_down_grasp(perception["center"])
    record(
        "grasp_selected",
        colour=perception["colour"],
        source=grasp["source"],
        candidate_index=grasp["index"],
        score=grasp["score"],
        downness=grasp["downness"],
        center_distance=grasp["center_distance"],
        position=grasp["position"],
        quaternion=grasp["quaternion"],
    )
    return grasp


def move_pose(label: str, position: np.ndarray, quaternion: np.ndarray) -> None:
    target_position = np.asarray(position, dtype=np.float64).reshape(3)
    target_quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    record("motion", label=label, position=target_position, quaternion=target_quaternion)
    joints = np.asarray(solve_ik(target_position, target_quaternion), dtype=np.float64).reshape(7)
    if not np.isfinite(joints).all():
        raise RuntimeError(f"IK returned non-finite joints for {label}: {joints}")
    move_to_joints(joints)


def pick_cube(perception: dict[str, Any], grasp: dict[str, Any]) -> np.ndarray:
    position = np.asarray(grasp["position"], dtype=np.float64)
    quaternion = np.asarray(grasp["quaternion"], dtype=np.float64)
    approach_axis = np.asarray(grasp["approach_axis"], dtype=np.float64)
    pregrasp = position - APPROACH_DISTANCE_M * approach_axis

    open_gripper()
    move_pose(f"pregrasp_{perception['colour']}", pregrasp, quaternion)
    move_pose(f"grasp_{perception['colour']}", position, quaternion)
    close_gripper()
    move_pose(
        f"lift_{perception['colour']}",
        position + np.array([0.0, 0.0, LIFT_DISTANCE_M]),
        quaternion,
    )
    # Preserve the perceived center-to-grasp offset during transfer.  This
    # compensates for Contact-GraspNet contacts that are not exactly centered.
    return np.asarray(perception["center"], dtype=np.float64) - position


def place_cube(
    colour: str,
    desired_center: np.ndarray,
    center_minus_grasp: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    desired_grasp_position = (
        np.asarray(desired_center, dtype=np.float64)
        - np.asarray(center_minus_grasp, dtype=np.float64)
    )
    preplace = desired_grasp_position + np.array([0.0, 0.0, TRANSFER_CLEARANCE_M])
    move_pose(f"preplace_{colour}", preplace, quaternion)
    move_pose(f"place_{colour}", desired_grasp_position, quaternion)
    open_gripper()
    move_pose(f"retract_{colour}", preplace, quaternion)


def choose_park_center(stack_center: np.ndarray, table_height: float) -> np.ndarray:
    stack = np.asarray(stack_center, dtype=np.float64)
    offsets = np.array(
        [
            [-PARK_DISTANCE_M, 0.0],
            [PARK_DISTANCE_M, 0.0],
            [0.0, -PARK_DISTANCE_M],
            [0.0, PARK_DISTANCE_M],
        ]
    )
    candidates = stack[:2] + offsets
    valid = (
        (candidates[:, 0] > 0.28)
        & (candidates[:, 0] < 0.72)
        & (np.abs(candidates[:, 1]) < 0.24)
    )
    if not np.any(valid):
        raise RuntimeError(f"No collision-free park location around {stack[:2]}")
    candidates = candidates[valid]
    # Prefer central, low-|y| targets for Franka reachability.
    costs = np.abs(candidates[:, 0] - 0.50) + 0.35 * np.abs(candidates[:, 1])
    xy = candidates[int(np.argmin(costs))]
    return np.array(
        [xy[0], xy[1], table_height + 0.5 * CUBE_SIDE_M + RELEASE_CLEARANCE_M],
        dtype=np.float64,
    )


def main() -> None:
    global RESULT

    # Makes the small-noise OBB helper deterministic across identical trials.
    np.random.seed(0)
    record("pipeline_start", mode="S3", llm_calls=0, cube_side_m=CUBE_SIDE_M)

    # State 1: the green cube initially blocks access to the red cube.
    green_initial = perceive_cube("green", "initial_stack")
    table_height = estimate_table_height(
        green_initial["observation"],
        fallback=float(green_initial["center"][2] - 1.5 * CUBE_SIDE_M),
    )
    park_center = choose_park_center(green_initial["center"], table_height)
    record("park_target", center=park_center)

    green_grasp = plan_cube_grasp(green_initial)
    green_offset = pick_cube(green_initial, green_grasp)
    place_cube(
        "green",
        park_center,
        green_offset,
        np.asarray(green_grasp["quaternion"]),
    )

    # State 2: refresh perception after physics settling; the red cube is now
    # completely visible and the parked green cube defines the stack target.
    green_parked = perceive_cube("green", "green_parked")
    red_exposed = perceive_cube("red", "red_exposed")
    red_grasp = plan_cube_grasp(red_exposed)
    red_offset = pick_cube(red_exposed, red_grasp)

    desired_red_center = np.array(
        [
            green_parked["center"][0],
            green_parked["center"][1],
            green_parked["top_z"] + 0.5 * CUBE_SIDE_M + RELEASE_CLEARANCE_M,
        ],
        dtype=np.float64,
    )
    record("stack_target", red_center=desired_red_center)
    place_cube(
        "red",
        desired_red_center,
        red_offset,
        np.asarray(red_grasp["quaternion"]),
    )

    # Perception-based postcondition diagnostics are deliberately non-fatal:
    # the simulator's task_completed flag remains the authoritative score.
    diagnostics: dict[str, Any] = {"commanded_red_center": desired_red_center}
    try:
        final_red = perceive_cube("red", "final_stack")
        final_green = perceive_cube("green", "final_stack")
        xy_error = float(np.linalg.norm(final_red["center"][:2] - final_green["center"][:2]))
        height_delta = float(final_red["center"][2] - final_green["center"][2])
        diagnostics.update(
            {
                "final_red_center": final_red["center"],
                "final_green_center": final_green["center"],
                "xy_error_m": xy_error,
                "height_delta_m": height_delta,
                "expected_height_delta_m": CUBE_SIDE_M,
            }
        )
        record(
            "final_diagnostics",
            xy_error_m=xy_error,
            height_delta_m=height_delta,
            expected_height_delta_m=CUBE_SIDE_M,
        )
    except Exception as exc:
        diagnostics["verification_error"] = repr(exc)
        record("verification_warning", error=repr(exc))

    RESULT = _jsonable(
        {
            "mode": "S3",
            "llm_calls": 0,
            "green_grasp_source": green_grasp["source"],
            "red_grasp_source": red_grasp["source"],
            "table_height_m": table_height,
            "park_center": park_center,
            **diagnostics,
        }
    )
    record("pipeline_complete", result=RESULT)


if __name__ == "__main__":
    main()
