"""Deterministic S3 solver for ``nut_assembly``.

The task is known in advance, so this policy uses task geometry instead of an
LLM / VLM:

    RGB-D -> task-known spatial ROIs -> 3-D boxes / nut hole -> GraspNet
          -> handle-to-nut rigid transform -> fixed insertion trajectory

Only the reduced S3 API functions injected by CaP-X are used.  In particular,
the program never reads simulator object poses and never calls a language
model.  The helper functions are kept importable so lightweight tests can
exercise the geometry without launching MuJoCo or a model server.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation


CAMERA_NAME = "robot0_robotview"

# Geometry from Robosuite's square-nut and square-peg assets.  These are task
# priors, not privileged state.  The nut is a 20-mm-thick 87.5-mm square ring;
# its handle center is 54 mm along the nut's +X axis.  The fixed square peg is
# axis-aligned and 32 mm wide.
NUT_THICKNESS_M = 0.020
NUT_OUTER_SIDE_M = 0.0875
NUT_HANDLE_CENTER_OFFSET_M = 0.054
NUT_HANDLE_X_RANGE_M = (0.027, 0.082)
NUT_HANDLE_HALF_WIDTH_M = 0.019
PEG_WIDTH_M = 0.032

# The source and target have non-overlapping task-defined workspace regions in
# robot-base coordinates.  Bounds are deliberately wider than the environment
# randomization ranges and are checked against observed 3-D geometry.
SOURCE_X_RANGE_M = (0.30, 0.66)
SOURCE_Y_RANGE_M = (0.03, 0.31)
PEG_X_RANGE_M = (0.68, 0.90)
PEG_Y_RANGE_M = (0.00, 0.22)

MIN_COMPONENT_PIXELS = 80
MIN_HANDLE_PIXELS = 35
# Reject steep side grasps even when GraspNet scores them highly: insertion
# needs the fingers and wrist to remain clear of the tall peg.  GraspNet is
# still the primary proposal source; the analytic task-known handle grasp is a
# deterministic fallback when its candidates do not satisfy this safety gate.
MIN_TOP_DOWN_COSINE = 0.95
MAX_HANDLE_DISTANCE_M = 0.018

APPROACH_DISTANCE_M = 0.080
# The peg top is roughly 120 mm above the source nut center.  A 120-mm lift
# therefore leaves the carried ring level with the peg and lets a joint-space
# transfer sweep the handle into it.  Lift high enough for the entire nut to
# clear the peg before any lateral motion.
LIFT_DISTANCE_M = 0.190
PREINSERT_CLEARANCE_M = 0.055
ENGAGE_CLEARANCE_M = 0.010
INSERT_DEPTH_M = 0.020
RETRACT_DISTANCE_M = 0.100

_artifact_dir_value = os.environ.get("CAPX_HEURISTIC_ARTIFACT_DIR")
ARTIFACT_DIR = Path(_artifact_dir_value) if _artifact_dir_value else None
TRACE: list[dict[str, Any]] = []


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def record(event: str, **values: Any) -> None:
    entry = {"event": event, **{key: _jsonable(value) for key, value in values.items()}}
    TRACE.append(entry)
    compact = ", ".join(f"{key}={value}" for key, value in entry.items() if key != "event")
    print(f"[nut_assembly:s3] {event}" + (f": {compact}" if compact else ""))
    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "trajectory_trace.json").write_text(
            json.dumps(TRACE, indent=2) + "\n", encoding="utf-8"
        )


def _depth_2d(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got {array.shape}")
    return array


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([points_array, np.ones(len(points_array), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def deproject_depth_image(
    depth: np.ndarray, intrinsics: np.ndarray, camera_to_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera/base point images and their validity mask."""

    depth_image = _depth_2d(depth)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    rows, cols = np.indices(depth_image.shape)
    valid = (
        np.isfinite(depth_image)
        & (depth_image > 0.05)
        & (depth_image < 3.0)
    )
    z = depth_image
    x = (cols - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (rows - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    points_camera = np.stack([x, y, z], axis=-1)
    points_base = np.full(points_camera.shape, np.nan, dtype=np.float64)
    points_base[valid] = transform_points(camera_to_base, points_camera[valid])
    return points_camera, points_base, valid


def estimate_table_height(points_base: np.ndarray, valid: np.ndarray) -> float:
    """Estimate the table plane as the dominant millimetre-scale Z mode."""

    points = np.asarray(points_base, dtype=np.float64)
    mask = (
        np.asarray(valid, dtype=bool)
        & (points[:, :, 0] > 0.20)
        & (points[:, :, 0] < 0.92)
        & (np.abs(points[:, :, 1]) < 0.38)
        & (points[:, :, 2] > -0.30)
        & (points[:, :, 2] < 0.12)
    )
    z = points[:, :, 2][mask]
    if len(z) < 1000:
        raise RuntimeError(f"Only {len(z)} table/workspace points are visible")
    edges = np.arange(-0.30, 0.121, 0.001)
    counts, edges = np.histogram(z, bins=edges)
    peak = int(np.argmax(counts))
    midpoint = 0.5 * (edges[peak] + edges[peak + 1])
    support = z[np.abs(z - midpoint) < 0.002]
    if len(support) < 500:
        raise RuntimeError(f"Table mode has only {len(support)} supporting points")
    return float(np.median(support))


def _clean_component(mask: np.ndarray, min_pixels: int = MIN_COMPONENT_PIXELS) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    binary = ndimage.binary_closing(binary, structure=structure, iterations=2)
    binary = ndimage.binary_opening(binary, structure=structure, iterations=1)
    labels, count = ndimage.label(binary)
    components = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels >= min_pixels:
            components.append((pixels, component))
    if not components:
        raise RuntimeError("No task-sized connected depth component was found")
    return max(components, key=lambda item: item[0])[1]


def _mask_box(mask: np.ndarray) -> list[int]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        return [0, 0, 0, 0]
    return [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)]


def find_hole_center(mask: np.ndarray, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the square nut's hollow center and return its pixel mask."""

    component = np.asarray(mask, dtype=bool)
    # Seal tiny raster/depth gaps in the four-bar ring before filling it.
    sealed = ndimage.binary_closing(component, structure=np.ones((3, 3), dtype=bool), iterations=3)
    holes = ndimage.binary_fill_holes(sealed) & ~sealed
    labels, count = ndimage.label(holes)
    candidates: list[tuple[int, np.ndarray]] = []
    for component_id in range(1, count + 1):
        hole = labels == component_id
        pixels = int(hole.sum())
        if pixels < 20:
            continue
        world_points = np.asarray(points_base)[hole]
        finite = np.isfinite(world_points).all(axis=1)
        if int(finite.sum()) < 10:
            continue
        candidates.append((pixels, hole))
    if not candidates:
        raise RuntimeError("The nut ring was found, but its hollow center was not closed")
    _, hole_mask = max(candidates, key=lambda item: item[0])
    hole_points = np.asarray(points_base)[hole_mask]
    hole_points = hole_points[np.isfinite(hole_points).all(axis=1)]
    center_xy = np.median(hole_points[:, :2], axis=0)
    return center_xy, hole_mask


def estimate_handle_axis(points_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Estimate the directed nut +X axis (hole center toward the handle)."""

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if len(points) < 20:
        raise RuntimeError("Too few nut points to estimate handle direction")
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis /= np.linalg.norm(axis)

    # The handle creates both more mass and a longer extent on the positive
    # side of the square ring.  Either cue disambiguates PCA's sign.
    centroid_direction = np.mean(centered, axis=0)
    projection = centered @ axis
    positive_extent = float(np.quantile(projection, 0.99))
    negative_extent = float(-np.quantile(projection, 0.01))
    sign_score = float(np.dot(centroid_direction, axis))
    if abs(sign_score) > 0.002:
        if sign_score < 0:
            axis = -axis
    elif positive_extent < negative_extent:
        axis = -axis
    return axis


def _task_obb(points: np.ndarray) -> dict[str, np.ndarray]:
    """Call the injected S3 OBB primitive and normalize its values."""

    obb = get_oriented_bounding_box_from_3d_points(np.asarray(points, dtype=np.float64))
    return {
        "center": np.asarray(obb["center"], dtype=np.float64),
        "extent": np.asarray(obb["extent"], dtype=np.float64),
        "R": np.asarray(obb["R"], dtype=np.float64),
    }


def detect_nut(
    points_base: np.ndarray, valid: np.ndarray, table_height: float
) -> dict[str, Any]:
    x = points_base[:, :, 0]
    y = points_base[:, :, 1]
    z = points_base[:, :, 2]
    raw_mask = (
        valid
        & (x > SOURCE_X_RANGE_M[0])
        & (x < SOURCE_X_RANGE_M[1])
        & (y > SOURCE_Y_RANGE_M[0])
        & (y < SOURCE_Y_RANGE_M[1])
        & (z > table_height + 0.004)
        & (z < table_height + 0.055)
    )
    mask = _clean_component(raw_mask)
    nut_points = points_base[mask]
    nut_points = nut_points[np.isfinite(nut_points).all(axis=1)]
    obb = _task_obb(nut_points)

    center_method = "hollow_center"
    try:
        center_xy, hole_mask = find_hole_center(mask, points_base)
    except RuntimeError as exc:
        # A partial rasterized ring can occasionally leave the hole open.  The
        # task-known overall silhouette center is 17.75 mm toward the handle
        # from the actual hole center, so an OBB-based fallback is available.
        provisional_center = obb["center"][:2]
        provisional_axis = estimate_handle_axis(nut_points[:, :2], provisional_center)
        center_xy = provisional_center - 0.01775 * provisional_axis
        hole_mask = np.zeros(mask.shape, dtype=bool)
        center_method = f"obb_fallback:{exc}"

    handle_axis = estimate_handle_axis(nut_points[:, :2], center_xy)
    perpendicular_axis = np.array([-handle_axis[1], handle_axis[0]], dtype=np.float64)
    delta = points_base[:, :, :2] - center_xy
    handle_u = delta[:, :, 0] * handle_axis[0] + delta[:, :, 1] * handle_axis[1]
    handle_v = delta[:, :, 0] * perpendicular_axis[0] + delta[:, :, 1] * perpendicular_axis[1]
    handle_mask = (
        mask
        & (handle_u > NUT_HANDLE_X_RANGE_M[0])
        & (handle_u < NUT_HANDLE_X_RANGE_M[1])
        & (np.abs(handle_v) < NUT_HANDLE_HALF_WIDTH_M)
    )
    if int(handle_mask.sum()) >= MIN_HANDLE_PIXELS:
        eroded = ndimage.binary_erosion(handle_mask, structure=np.ones((3, 3), dtype=bool))
        if int(eroded.sum()) >= MIN_HANDLE_PIXELS:
            handle_mask = eroded
    if int(handle_mask.sum()) < MIN_HANDLE_PIXELS:
        raise RuntimeError(f"Only {int(handle_mask.sum())} handle pixels were isolated")

    top_z = float(np.quantile(nut_points[:, 2], 0.90))
    center = np.array(
        [center_xy[0], center_xy[1], table_height + 0.5 * NUT_THICKNESS_M],
        dtype=np.float64,
    )
    handle_center = center.copy()
    handle_center[:2] += NUT_HANDLE_CENTER_OFFSET_M * handle_axis
    return {
        "mask": mask,
        "handle_mask": handle_mask,
        "hole_mask": hole_mask,
        "box": _mask_box(mask),
        "center": center,
        "handle_center": handle_center,
        "handle_axis": handle_axis,
        "top_z": top_z,
        "center_method": center_method,
        "obb": obb,
    }


def detect_square_peg(
    points_base: np.ndarray, valid: np.ndarray, table_height: float
) -> dict[str, Any]:
    x = points_base[:, :, 0]
    y = points_base[:, :, 1]
    z = points_base[:, :, 2]
    raw_mask = (
        valid
        & (x > PEG_X_RANGE_M[0])
        & (x < PEG_X_RANGE_M[1])
        & (y > PEG_Y_RANGE_M[0])
        & (y < PEG_Y_RANGE_M[1])
        & (z > table_height + 0.080)
        & (z < table_height + 0.205)
    )
    mask = _clean_component(raw_mask, min_pixels=30)
    peg_points = points_base[mask]
    peg_points = peg_points[np.isfinite(peg_points).all(axis=1)]
    top_z = float(np.quantile(peg_points[:, 2], 0.95))
    top_points = peg_points[peg_points[:, 2] >= top_z - 0.004]
    if len(top_points) < 20:
        top_points = peg_points
    low_xy = np.quantile(top_points[:, :2], 0.05, axis=0)
    high_xy = np.quantile(top_points[:, :2], 0.95, axis=0)
    center_xy = 0.5 * (low_xy + high_xy)
    obb = _task_obb(peg_points)

    # The arena's square peg is axis-aligned with the robot base.  A square's
    # PCA/OBB yaw is intrinsically unstable, so snap the measured OBB axis to
    # this known symmetry class.  Rotations by 90 degrees are equivalent for
    # insertion and are considered later when choosing a reachable handle side.
    measured_axis = obb["R"][:2, int(np.argmax(obb["extent"][:2]))]
    measured_yaw = float(math.atan2(measured_axis[1], measured_axis[0]))
    snapped_yaw = round(measured_yaw / (0.5 * math.pi)) * (0.5 * math.pi)
    axis = np.array([math.cos(snapped_yaw), math.sin(snapped_yaw)], dtype=np.float64)
    return {
        "mask": mask,
        "box": _mask_box(mask),
        "center": np.array([center_xy[0], center_xy[1], top_z], dtype=np.float64),
        "top_z": top_z,
        "axis": axis,
        "measured_yaw": measured_yaw,
        "snapped_yaw": snapped_yaw,
        "obb": obb,
    }


def project_base_point(
    point_base: np.ndarray, intrinsics: np.ndarray, camera_to_base: np.ndarray
) -> tuple[float, float] | None:
    point = np.asarray(point_base, dtype=np.float64).reshape(3)
    point_camera = transform_points(np.linalg.inv(camera_to_base), point[None, :])[0]
    if not np.isfinite(point_camera).all() or point_camera[2] <= 0:
        return None
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    col = camera_matrix[0, 0] * point_camera[0] / point_camera[2] + camera_matrix[0, 2]
    row = camera_matrix[1, 1] * point_camera[1] / point_camera[2] + camera_matrix[1, 2]
    return float(col), float(row)


def _draw_world_axis(
    draw: ImageDraw.ImageDraw,
    origin: np.ndarray,
    axis_xy: np.ndarray,
    z: float,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
    color: str,
    length: float = 0.065,
) -> None:
    start = project_base_point(np.array([origin[0], origin[1], z]), intrinsics, camera_to_base)
    end = project_base_point(
        np.array([origin[0] + length * axis_xy[0], origin[1] + length * axis_xy[1], z]),
        intrinsics,
        camera_to_base,
    )
    if start is not None and end is not None:
        draw.line([start, end], fill=color, width=4)
        radius = 4
        draw.ellipse(
            [end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius],
            fill=color,
        )


def save_initial_overlay(
    rgb: np.ndarray,
    camera: dict[str, Any],
    nut: dict[str, Any],
    peg: dict[str, Any],
    table_height: float,
) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8).copy()
    overlays = (
        (nut["mask"], np.array([30, 220, 255], dtype=np.float64)),
        (nut["handle_mask"], np.array([255, 145, 20], dtype=np.float64)),
        (peg["mask"], np.array([235, 50, 235], dtype=np.float64)),
    )
    for mask, tint in overlays:
        mask_bool = np.asarray(mask, dtype=bool)
        image[mask_bool] = np.clip(0.50 * image[mask_bool] + 0.50 * tint, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(tuple(nut["box"]), outline="cyan", width=3)
    draw.rectangle(tuple(peg["box"]), outline="magenta", width=3)
    intrinsics = camera["intrinsics"]
    camera_to_base = camera["pose_mat"]
    _draw_world_axis(
        draw,
        nut["center"][:2],
        nut["handle_axis"],
        nut["top_z"],
        intrinsics,
        camera_to_base,
        "orange",
    )
    _draw_world_axis(
        draw,
        peg["center"][:2],
        peg["axis"],
        peg["top_z"],
        intrinsics,
        camera_to_base,
        "white",
        length=0.050,
    )
    for point, color in ((nut["center"], "yellow"), (peg["center"], "lime")):
        pixel = project_base_point(point, intrinsics, camera_to_base)
        if pixel is not None:
            radius = 6
            draw.ellipse(
                [pixel[0] - radius, pixel[1] - radius, pixel[0] + radius, pixel[1] + radius],
                outline=color,
                width=3,
            )
    label = (
        f"cyan=nut  orange=handle  magenta=square peg\n"
        f"nut center={np.round(nut['center'], 4).tolist()}  "
        f"peg top={np.round(peg['center'], 4).tolist()}  table z={table_height:.4f}"
    )
    draw.rectangle((4, 4, min(canvas.width - 4, 720), 46), fill=(0, 0, 0))
    draw.text((8, 7), label, fill="white")
    canvas.save(ARTIFACT_DIR / "initial_perception.png")


def perceive_initial_scene() -> dict[str, Any]:
    observation = get_observation()
    camera = observation[CAMERA_NAME]
    rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
    depth = _depth_2d(camera["images"]["depth"])
    points_camera, points_base, valid = deproject_depth_image(
        depth, camera["intrinsics"], camera["pose_mat"]
    )
    table_height = estimate_table_height(points_base, valid)
    nut = detect_nut(points_base, valid, table_height)
    peg = detect_square_peg(points_base, valid, table_height)
    record(
        "perception",
        phase="initial",
        detector="task_roi_rgbd",
        table_height=table_height,
        nut_center=nut["center"],
        nut_handle_center=nut["handle_center"],
        nut_handle_axis=nut["handle_axis"],
        nut_bbox_px=nut["box"],
        nut_mask_pixels=int(nut["mask"].sum()),
        handle_mask_pixels=int(nut["handle_mask"].sum()),
        nut_obb_extent=nut["obb"]["extent"],
        nut_center_method=nut["center_method"],
        peg_top_center=peg["center"],
        peg_bbox_px=peg["box"],
        peg_mask_pixels=int(peg["mask"].sum()),
        peg_obb_extent=peg["obb"]["extent"],
        peg_measured_yaw=peg["measured_yaw"],
        peg_snapped_yaw=peg["snapped_yaw"],
    )
    save_initial_overlay(rgb, camera, nut, peg, table_height)
    return {
        "observation": observation,
        "camera": camera,
        "rgb": rgb,
        "depth": depth,
        "points_camera": points_camera,
        "points_base": points_base,
        "valid": valid,
        "table_height": table_height,
        "nut": nut,
        "peg": peg,
    }


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def nut_rotation_from_handle_axis(handle_axis: np.ndarray) -> np.ndarray:
    axis_x = np.asarray(handle_axis, dtype=np.float64).reshape(2)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float64)
    return np.array(
        [
            [axis_x[0], axis_y[0], 0.0],
            [axis_x[1], axis_y[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def analytic_handle_grasp(nut: dict[str, Any]) -> dict[str, Any]:
    """Top-down grasp with gripper X pointing from handle toward the hole."""

    handle_axis = np.asarray(nut["handle_axis"], dtype=np.float64)
    gripper_x = np.array([-handle_axis[0], -handle_axis[1], 0.0], dtype=np.float64)
    gripper_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gripper_y = np.cross(gripper_z, gripper_x)
    rotation = np.column_stack([gripper_x, gripper_y, gripper_z])
    position = np.asarray(nut["handle_center"], dtype=np.float64).copy()
    return {
        "index": -1,
        "pose": pose_matrix(position, rotation),
        "position": position,
        "rotation": rotation,
        "quaternion": matrix_to_quaternion_wxyz(rotation),
        "approach_axis": gripper_z,
        "score": 0.0,
        "downness": 1.0,
        "axis_alignment": 1.0,
        "handle_distance": 0.0,
        "rank": 0.0,
        "source": "analytic_fallback",
    }


def rank_grasp_candidates(
    grasp_poses_camera: np.ndarray,
    grasp_scores: np.ndarray,
    camera_to_base: np.ndarray,
    nut_center: np.ndarray,
    expected_handle_center: np.ndarray,
    handle_axis: np.ndarray,
) -> list[dict[str, Any]]:
    """Transform, orient, and rank top-down handle grasps."""

    poses = np.asarray(grasp_poses_camera, dtype=np.float64)
    scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
        return []
    camera_transform = np.asarray(camera_to_base, dtype=np.float64)
    nut_position = np.asarray(nut_center, dtype=np.float64)
    expected = np.asarray(expected_handle_center, dtype=np.float64)
    outward = np.asarray(handle_axis, dtype=np.float64).reshape(2)
    candidates: list[dict[str, Any]] = []

    for index, (pose_camera, score) in enumerate(zip(poses, scores, strict=True)):
        pose_base = camera_transform @ pose_camera
        if not np.isfinite(pose_base).all() or not np.isfinite(score):
            continue
        position = pose_base[:3, 3]
        rotation = pose_base[:3, :3].copy()
        approach_axis = rotation[:, 2]
        downness = float(-approach_axis[2])
        reachable = 0.18 < position[0] < 0.86 and abs(position[1]) < 0.42
        handle_distance = float(np.linalg.norm(position - expected))
        gripper_x_xy = rotation[:2, 0]
        norm_xy = float(np.linalg.norm(gripper_x_xy))
        axis_alignment = (
            float(abs(np.dot(gripper_x_xy / norm_xy, outward))) if norm_xy > 1e-6 else 0.0
        )
        if (
            not reachable
            or downness < MIN_TOP_DOWN_COSINE
            or handle_distance > MAX_HANDLE_DISTANCE_M
            or axis_alignment < 0.45
        ):
            continue

        # The gripper is symmetric under 180 degrees about its approach axis.
        # Normalize that ambiguity so +X always points from handle to nut hole,
        # matching the rigid-transform convention used for insertion.
        direction_to_center = nut_position - position
        if float(np.dot(rotation[:, 0], direction_to_center)) < 0.0:
            rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
            pose_base[:3, :3] = rotation
        rank = (
            float(score)
            + 0.30 * downness
            + 0.25 * axis_alignment
            - 5.0 * handle_distance
        )
        candidates.append(
            {
                "index": index,
                "pose": pose_base,
                "position": position.copy(),
                "rotation": rotation,
                "quaternion": matrix_to_quaternion_wxyz(rotation),
                "approach_axis": rotation[:, 2],
                "score": float(score),
                "downness": downness,
                "axis_alignment": axis_alignment,
                "handle_distance": handle_distance,
                "rank": rank,
                "source": "contact_graspnet",
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["rank"], reverse=True)


def plan_handle_grasp(scene: dict[str, Any]) -> dict[str, Any]:
    nut = scene["nut"]
    try:
        poses, scores = plan_grasp(
            depth=scene["depth"],
            intrinsics=scene["camera"]["intrinsics"],
            segmentation=nut["handle_mask"].astype(np.int32),
        )
        candidates = rank_grasp_candidates(
            poses,
            scores,
            scene["camera"]["pose_mat"],
            nut["center"],
            nut["handle_center"],
            nut["handle_axis"],
        )
    except Exception as exc:
        record("graspnet_error", error=repr(exc))
        candidates = []
    grasp = candidates[0] if candidates else analytic_handle_grasp(nut)
    record(
        "grasp_selected",
        source=grasp["source"],
        candidate_index=grasp["index"],
        score=grasp["score"],
        downness=grasp["downness"],
        axis_alignment=grasp["axis_alignment"],
        handle_distance=grasp["handle_distance"],
        position=grasp["position"],
        quaternion=grasp["quaternion"],
    )
    return grasp


def save_grasp_overlay(scene: dict[str, Any], grasp: dict[str, Any]) -> None:
    """Visualize the accepted grasp point and orientation on the real RGB frame."""

    if ARTIFACT_DIR is None:
        return
    image = np.asarray(scene["rgb"], dtype=np.uint8).copy()
    handle_mask = np.asarray(scene["nut"]["handle_mask"], dtype=bool)
    tint = np.array([255, 145, 20], dtype=np.float64)
    image[handle_mask] = np.clip(
        0.45 * image[handle_mask] + 0.55 * tint, 0, 255
    ).astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    camera = scene["camera"]
    position = np.asarray(grasp["position"], dtype=np.float64)
    rotation = np.asarray(grasp["rotation"], dtype=np.float64)
    origin = project_base_point(position, camera["intrinsics"], camera["pose_mat"])
    x_end = project_base_point(
        position + 0.050 * rotation[:, 0], camera["intrinsics"], camera["pose_mat"]
    )
    y_end = project_base_point(
        position + 0.035 * rotation[:, 1], camera["intrinsics"], camera["pose_mat"]
    )
    if origin is not None:
        draw.ellipse(
            [origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7],
            outline="red",
            width=4,
        )
        if x_end is not None:
            draw.line([origin, x_end], fill="yellow", width=4)
        if y_end is not None:
            draw.line([origin, y_end], fill="lime", width=4)
    label = (
        f"orange=handle mask  red=grasp  yellow=gripper X  lime=closing axis\n"
        f"source={grasp['source']}  score={grasp['score']:.3f}  "
        f"downness={grasp['downness']:.3f}  axis alignment={grasp['axis_alignment']:.3f}"
    )
    draw.rectangle((4, 4, min(canvas.width - 4, 730), 46), fill=(0, 0, 0))
    draw.text((8, 7), label, fill="white")
    canvas.save(ARTIFACT_DIR / "grasp_selection.png")


def choose_target_handle_axis(
    peg_axis: np.ndarray,
    peg_center: np.ndarray,
    source_handle_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Choose a reachable square-symmetric yaw with little in-gripper rotation.

    All four 90-degree rotations fit the square ring over the square peg.  A
    large wrist yaw during transport can twist the short handle inside the
    fingers, so prefer the equivalent yaw nearest the observed source yaw.
    """

    axis = np.asarray(peg_axis, dtype=np.float64).reshape(2)
    axis /= np.linalg.norm(axis)
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float64)
    candidates = [axis, -axis, perpendicular, -perpendicular]
    peg_xy = np.asarray(peg_center, dtype=np.float64)[:2]

    source = None
    if source_handle_axis is not None:
        source = np.asarray(source_handle_axis, dtype=np.float64).reshape(2)
        source /= np.linalg.norm(source)

    def cost(candidate: np.ndarray) -> float:
        handle_xy = peg_xy + NUT_HANDLE_CENTER_OFFSET_M * candidate
        # +X places the TCP close to the Franka's fully-extended boundary.
        # Keep the prior conservative bound and use either +/-Y when that yaw
        # is closer than the comfortably reachable -X alternative.
        invalid = not (0.25 < handle_xy[0] < 0.83 and abs(handle_xy[1]) < 0.35)
        yaw_change = (
            0.0
            if source is None
            else math.acos(float(np.clip(np.dot(source, candidate), -1.0, 1.0)))
        )
        return (
            (100.0 if invalid else 0.0)
            + yaw_change
            + 0.20 * abs(handle_xy[0] - 0.55)
            + 0.05 * abs(handle_xy[1])
        )

    return min(candidates, key=cost).copy()


def compute_nut_to_grasp_transform(nut: dict[str, Any], grasp: dict[str, Any]) -> np.ndarray:
    nut_rotation = nut_rotation_from_handle_axis(nut["handle_axis"])
    base_to_nut = pose_matrix(nut["center"], nut_rotation)
    base_to_grasp = pose_matrix(grasp["position"], grasp["rotation"])
    return np.linalg.inv(base_to_nut) @ base_to_grasp


def desired_grasp_pose(
    desired_nut_center: np.ndarray,
    desired_handle_axis: np.ndarray,
    nut_to_grasp: np.ndarray,
) -> np.ndarray:
    desired_nut = pose_matrix(
        desired_nut_center, nut_rotation_from_handle_axis(desired_handle_axis)
    )
    return desired_nut @ np.asarray(nut_to_grasp, dtype=np.float64)


def move_pose(label: str, pose: np.ndarray) -> None:
    transform = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    position = transform[:3, 3]
    quaternion = matrix_to_quaternion_wxyz(transform[:3, :3])
    record("motion", label=label, position=position, quaternion=quaternion)
    joints = np.asarray(solve_ik(position, quaternion), dtype=np.float64).reshape(7)
    if not np.isfinite(joints).all():
        raise RuntimeError(f"IK returned non-finite joints for {label}: {joints}")
    move_to_joints(joints)


def save_trajectory_schematic(
    scene: dict[str, Any],
    grasp: dict[str, Any],
    target_axis: np.ndarray,
    target_centers: list[tuple[str, np.ndarray]],
) -> None:
    if ARTIFACT_DIR is None:
        return
    width, height = 760, 520
    canvas = Image.new("RGB", (width, height), (244, 244, 238))
    draw = ImageDraw.Draw(canvas)
    x_min, x_max = 0.20, 0.92
    y_min, y_max = -0.34, 0.34

    def pixel(xy: np.ndarray) -> tuple[float, float]:
        x_value, y_value = float(xy[0]), float(xy[1])
        col = 55 + (x_value - x_min) / (x_max - x_min) * (width - 100)
        row = height - 55 - (y_value - y_min) / (y_max - y_min) * (height - 105)
        return col, row

    draw.rectangle((55, 45, width - 45, height - 55), outline=(90, 90, 90), width=2)
    draw.text((55, 16), "nut_assembly S3: base-frame top view (metres)", fill=(10, 10, 10))
    nut = scene["nut"]
    peg = scene["peg"]
    nut_px = pixel(nut["center"][:2])
    peg_px = pixel(peg["center"][:2])
    handle_px = pixel(nut["handle_center"][:2])
    grasp_px = pixel(grasp["position"][:2])
    draw.ellipse((nut_px[0] - 17, nut_px[1] - 17, nut_px[0] + 17, nut_px[1] + 17), outline=(0, 120, 160), width=4)
    draw.line([nut_px, handle_px], fill=(230, 120, 0), width=5)
    draw.ellipse((grasp_px[0] - 6, grasp_px[1] - 6, grasp_px[0] + 6, grasp_px[1] + 6), fill=(210, 35, 35))
    draw.rectangle((peg_px[0] - 11, peg_px[1] - 11, peg_px[0] + 11, peg_px[1] + 11), outline=(160, 0, 160), width=4)
    target_handle_px = pixel(peg["center"][:2] + NUT_HANDLE_CENTER_OFFSET_M * target_axis)
    draw.line([peg_px, target_handle_px], fill=(0, 145, 50), width=5)
    draw.line([grasp_px, target_handle_px], fill=(40, 80, 220), width=3)
    draw.polygon(
        [
            (target_handle_px[0], target_handle_px[1]),
            (target_handle_px[0] - 10, target_handle_px[1] - 5),
            (target_handle_px[0] - 8, target_handle_px[1] + 7),
        ],
        fill=(40, 80, 220),
    )
    labels = [
        (nut_px, "detected nut hole", (0, 100, 145)),
        (grasp_px, f"{grasp['source']} grasp", (190, 20, 20)),
        (peg_px, "detected square peg", (130, 0, 130)),
        (target_handle_px, "chosen insertion yaw", (0, 115, 35)),
    ]
    for location, label, color in labels:
        draw.text((location[0] + 9, location[1] + 8), label, fill=color)
    y_text = 72
    for label, center in target_centers:
        draw.text(
            (70, y_text),
            f"{label:>9}: nut center {np.round(center, 4).tolist()}",
            fill=(20, 20, 20),
        )
        y_text += 18
    draw.text((70, height - 40), "blue=transfer  green=handle direction  red=selected grasp", fill=(20, 20, 20))
    canvas.save(ARTIFACT_DIR / "trajectory_schematic.png")


def save_final_overlay(
    rgb: np.ndarray,
    camera: dict[str, Any],
    points_base: np.ndarray,
    valid: np.ndarray,
    table_height: float,
    peg_center: np.ndarray,
    target_axis: np.ndarray,
) -> dict[str, Any]:
    peg_xy = np.asarray(peg_center, dtype=np.float64)[:2]
    delta = points_base[:, :, :2] - peg_xy
    distance = np.linalg.norm(delta, axis=2)
    z = points_base[:, :, 2]
    low_nut_mask = (
        valid
        & (distance < 0.14)
        & (z > table_height + 0.004)
        & (z < table_height + 0.045)
    )
    expected_handle_xy = peg_xy + NUT_HANDLE_CENTER_OFFSET_M * target_axis
    handle_delta = points_base[:, :, :2] - expected_handle_xy
    handle_support = low_nut_mask & (np.linalg.norm(handle_delta, axis=2) < 0.030)
    support_points = points_base[handle_support]
    support_points = support_points[np.isfinite(support_points).all(axis=1)]
    estimated_handle_xy = (
        np.median(support_points[:, :2], axis=0) if len(support_points) else np.array([np.nan, np.nan])
    )
    handle_error = (
        float(np.linalg.norm(estimated_handle_xy - expected_handle_xy))
        if np.isfinite(estimated_handle_xy).all()
        else None
    )

    if ARTIFACT_DIR is not None:
        image = np.asarray(rgb, dtype=np.uint8).copy()
        tint = np.array([40, 225, 90], dtype=np.float64)
        image[low_nut_mask] = np.clip(
            0.55 * image[low_nut_mask] + 0.45 * tint, 0, 255
        ).astype(np.uint8)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        _draw_world_axis(
            draw,
            peg_xy,
            target_axis,
            table_height + NUT_THICKNESS_M,
            camera["intrinsics"],
            camera["pose_mat"],
            "yellow",
        )
        center_pixel = project_base_point(
            np.array([peg_xy[0], peg_xy[1], table_height + NUT_THICKNESS_M]),
            camera["intrinsics"],
            camera["pose_mat"],
        )
        if center_pixel is not None:
            draw.ellipse(
                [center_pixel[0] - 7, center_pixel[1] - 7, center_pixel[0] + 7, center_pixel[1] + 7],
                outline="red",
                width=3,
            )
        label = (
            f"green=low nut points near peg  yellow=commanded handle axis\n"
            f"handle support={len(support_points)} px  error={handle_error}"
        )
        draw.rectangle((4, 4, min(canvas.width - 4, 690), 46), fill=(0, 0, 0))
        draw.text((8, 7), label, fill="white")
        canvas.save(ARTIFACT_DIR / "final_verification.png")
    return {
        "low_nut_pixels": int(low_nut_mask.sum()),
        "handle_support_pixels": int(len(support_points)),
        "estimated_handle_xy": estimated_handle_xy,
        "expected_handle_xy": expected_handle_xy,
        "handle_xy_error_m": handle_error,
    }


def main() -> None:
    global RESULT

    np.random.seed(0)
    record(
        "pipeline_start",
        mode="S3",
        llm_calls=0,
        segmentation="task_known_rgbd_roi",
        nut_handle_offset_m=NUT_HANDLE_CENTER_OFFSET_M,
    )
    scene = perceive_initial_scene()
    grasp = plan_handle_grasp(scene)
    save_grasp_overlay(scene, grasp)
    nut_to_grasp = compute_nut_to_grasp_transform(scene["nut"], grasp)
    target_axis = choose_target_handle_axis(
        scene["peg"]["axis"],
        scene["peg"]["center"],
        scene["nut"]["handle_axis"],
    )
    peg_top_z = float(scene["peg"]["top_z"])
    target_yaw_change = math.acos(
        float(
            np.clip(
                np.dot(scene["nut"]["handle_axis"], target_axis),
                -1.0,
                1.0,
            )
        )
    )
    target_centers = [
        (
            "preinsert",
            np.array(
                [scene["peg"]["center"][0], scene["peg"]["center"][1], peg_top_z + PREINSERT_CLEARANCE_M]
            ),
        ),
        (
            "engage",
            np.array(
                [scene["peg"]["center"][0], scene["peg"]["center"][1], peg_top_z + ENGAGE_CLEARANCE_M]
            ),
        ),
        (
            "insert",
            np.array(
                [scene["peg"]["center"][0], scene["peg"]["center"][1], peg_top_z - INSERT_DEPTH_M]
            ),
        ),
    ]
    record(
        "rigid_transform",
        nut_to_grasp=nut_to_grasp,
        target_handle_axis=target_axis,
        target_yaw_change_deg=math.degrees(target_yaw_change),
        target_centers={label: center for label, center in target_centers},
    )
    save_trajectory_schematic(scene, grasp, target_axis, target_centers)

    if os.environ.get("CAPX_HEURISTIC_PERCEPTION_ONLY", "0") == "1":
        RESULT = _jsonable(
            {
                "mode": "S3_perception_only",
                "llm_calls": 0,
                "nut_center": scene["nut"]["center"],
                "peg_top_center": scene["peg"]["center"],
                "grasp_source": grasp["source"],
            }
        )
        record("pipeline_complete", result=RESULT)
        return

    open_gripper()
    grasp_pose = pose_matrix(grasp["position"], grasp["rotation"])
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] -= APPROACH_DISTANCE_M * np.asarray(grasp["approach_axis"])
    move_pose("pregrasp_handle", pregrasp_pose)
    move_pose("grasp_handle", grasp_pose)
    close_gripper()
    lift_pose = grasp_pose.copy()
    lift_pose[:3, 3] += np.array([0.0, 0.0, LIFT_DISTANCE_M])
    move_pose("lift_nut", lift_pose)

    insertion_poses: list[tuple[str, np.ndarray]] = []
    for label, desired_center in target_centers:
        target_grasp = desired_grasp_pose(desired_center, target_axis, nut_to_grasp)
        insertion_poses.append((label, target_grasp))
        move_pose(label, target_grasp)

    open_gripper()
    retract_pose = insertion_poses[-1][1].copy()
    retract_pose[:3, 3] += np.array([0.0, 0.0, RETRACT_DISTANCE_M])
    move_pose("retract", retract_pose)

    # Move back over the now-empty source region so the overhead camera can
    # verify the placed nut without the arm occluding the peg.
    camera_clear_pose = pregrasp_pose.copy()
    move_pose("camera_clear", camera_clear_pose)

    diagnostics: dict[str, Any] = {}
    try:
        final_observation = get_observation()
        final_camera = final_observation[CAMERA_NAME]
        final_rgb = np.asarray(final_camera["images"]["rgb"], dtype=np.uint8)
        _, final_points_base, final_valid = deproject_depth_image(
            final_camera["images"]["depth"],
            final_camera["intrinsics"],
            final_camera["pose_mat"],
        )
        diagnostics = save_final_overlay(
            final_rgb,
            final_camera,
            final_points_base,
            final_valid,
            scene["table_height"],
            scene["peg"]["center"],
            target_axis,
        )
        record("final_diagnostics", **diagnostics)
    except Exception as exc:
        diagnostics = {"verification_error": repr(exc)}
        record("verification_warning", error=repr(exc))

    RESULT = _jsonable(
        {
            "mode": "S3",
            "llm_calls": 0,
            "segmentation": "task_known_rgbd_roi",
            "grasp_source": grasp["source"],
            "nut_center_initial": scene["nut"]["center"],
            "peg_top_center": scene["peg"]["center"],
            "target_handle_axis": target_axis,
            "nut_to_grasp": nut_to_grasp,
            **diagnostics,
        }
    )
    record("pipeline_complete", result=RESULT)


if __name__ == "__main__":
    main()