"""Deterministic S3 solver for ``cube_stack``.

Two-cube pick-and-place: perceive both cubes (smaller cubeA on top of larger
cubeB), grasp the smaller with Contact-GraspNet, place it on the larger, release.
No LLM/VLM query, no privileged state.

Robustness strategy across the variant set:
- baseline: cubeA red (side 0.04 m), cubeB green (side 0.05 m). Size delta (1 cm)
  is a stronger cue than color, so target selection is size-first.
- color_shift_*: both cubes retinted to the same color -- size ordering still works.
- object_position_shift / robot_position_shift: workspace bounds are wide enough
  to cover the biased sampling range.
- distractor_object: an extra cylinder is placed in the workspace. We pick the
  two workspace-foreground clusters whose OBB extent looks most cube-shaped
  (near-1 aspect ratio); the cylinder distractor becomes the discarded third.
- target_object_swap: cubeA replaced by a cylinder. Since the swap only changes
  cubeA's *shape* (its OBB extent stays ~0.04 in the two horizontal dims), the
  size-first ranking still selects it as the smaller of the two objects.
- lighting_*: perception uses depth-based foreground extraction, so brightness
  doesn't affect detection quality.

The CaP-X executor injects the S3 API functions (get_observation, plan_grasp,
solve_ik, move_to_joints, open_gripper, close_gripper,
get_oriented_bounding_box_from_3d_points) into ``__main__``.
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


CAMERA_NAME = "robot0_robotview"

# Robosuite's Stack task: cubeA half-side 0.02 (full side 4 cm), cubeB half-side
# 0.025 (full side 5 cm). See capx/third_party/robosuite/.../stack.py .
CUBE_A_SIDE_M = 0.040
CUBE_B_SIDE_M = 0.050
APPROACH_DISTANCE_M = 0.090
LIFT_CLEARANCE_M = 0.180
RELEASE_CLEARANCE_M = 0.002

MIN_MASK_PIXELS = 60
MIN_TOP_DOWN_COSINE = 0.55
TOP_DOWN_QUATERNION_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

WORKSPACE_X_RANGE = (0.22, 0.85)
WORKSPACE_Y_RANGE = (-0.40, 0.40)

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
    print(f"[cube_stack:s3] {event}" + (f": {compact}" if compact else ""))
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


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
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


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def deproject_depth_image(
    depth: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    depth_image = _depth_2d(depth)
    valid = np.isfinite(depth_image) & (depth_image > 0.05) & (depth_image < 3.0)
    height, width = depth_image.shape
    rows, cols = np.mgrid[0:height, 0:width]
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    z = depth_image
    x = (cols - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (rows - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def estimate_table_height(points_base: np.ndarray) -> float:
    workspace = points_base[
        (points_base[:, 0] > WORKSPACE_X_RANGE[0])
        & (points_base[:, 0] < WORKSPACE_X_RANGE[1])
        & (points_base[:, 1] > WORKSPACE_Y_RANGE[0])
        & (points_base[:, 1] < WORKSPACE_Y_RANGE[1])
        & (points_base[:, 2] > -0.30)
        & (points_base[:, 2] < 0.10)
    ]
    if len(workspace) < 1000:
        raise RuntimeError(f"too few workspace points to estimate table: {len(workspace)}")
    edges = np.arange(-0.30, 0.101, 0.001)
    counts, edges = np.histogram(workspace[:, 2], bins=edges)
    peak = int(np.argmax(counts))
    midpoint = 0.5 * (edges[peak] + edges[peak + 1])
    support = workspace[np.abs(workspace[:, 2] - midpoint) < 0.003, 2]
    if len(support) < 500:
        raise RuntimeError("table mode has too little support")
    return float(np.median(support))


def workspace_foreground_mask(
    points_base: np.ndarray,
    valid: np.ndarray,
    table_height: float,
) -> np.ndarray:
    height, width = valid.shape
    points = points_base.reshape(height, width, 3)
    on_table = (
        valid
        & (points[:, :, 0] > WORKSPACE_X_RANGE[0])
        & (points[:, :, 0] < WORKSPACE_X_RANGE[1])
        & (points[:, :, 1] > WORKSPACE_Y_RANGE[0])
        & (points[:, :, 1] < WORKSPACE_Y_RANGE[1])
        & (points[:, :, 2] > table_height + 0.008)
        & (points[:, :, 2] < table_height + 0.20)
    )
    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndimage.binary_closing(on_table, structure=structure, iterations=1)
    cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=1)
    return cleaned


def component_detections(
    mask: np.ndarray, points_base: np.ndarray
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask)
    detections: list[dict[str, Any]] = []
    height, width = mask.shape
    for component_id in range(1, count + 1):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels < MIN_MASK_PIXELS:
            continue
        rows, cols = np.nonzero(component)
        cluster_points = points_base.reshape(height, width, 3)[component]
        center_xy = cluster_points[:, :2].mean(axis=0)
        top_z = float(np.quantile(cluster_points[:, 2], 0.90))
        detections.append(
            {
                "mask": component,
                "box": [float(cols.min()), float(rows.min()),
                        float(cols.max() + 1), float(rows.max() + 1)],
                "pixels": pixels,
                "center_xy": center_xy,
                "top_z": top_z,
            }
        )
    return detections


def estimate_object_geometry(
    points_camera: np.ndarray,
    mask: np.ndarray,
    camera_to_base: np.ndarray,
    nominal_side_m: float,
) -> dict[str, Any]:
    valid_mask = mask & np.isfinite(points_camera[:, :, 2])
    cluster_camera = points_camera[valid_mask]
    if len(cluster_camera) < MIN_MASK_PIXELS:
        raise RuntimeError(f"only {len(cluster_camera)} valid RGB-D pixels in mask")

    points_base = transform_points(camera_to_base, cluster_camera)
    obb = get_oriented_bounding_box_from_3d_points(points_base)
    obb_center_base = np.asarray(obb["center"], dtype=np.float64)
    obb_extent = np.asarray(obb["extent"], dtype=np.float64)

    top_z = float(np.quantile(points_base[:, 2], 0.90))
    top_band = points_base[points_base[:, 2] >= top_z - 0.004]
    if len(top_band) >= 20:
        low_xy = np.quantile(top_band[:, :2], 0.05, axis=0)
        high_xy = np.quantile(top_band[:, :2], 0.95, axis=0)
        center_xy = 0.5 * (low_xy + high_xy)
    else:
        center_xy = obb_center_base[:2]

    center_base = np.array(
        [center_xy[0], center_xy[1], top_z - 0.5 * nominal_side_m],
        dtype=np.float64,
    )
    return {
        "center": center_base,
        "top_z": top_z,
        "obb_center": obb_center_base,
        "obb_extent": obb_extent,
        "points_base": points_base,
    }


def select_cube_pair(
    detections: list[dict[str, Any]],
    points_camera: np.ndarray,
    camera_to_base: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """From workspace-foreground blobs, pick (target = smaller, base = larger).

    We compute a preliminary OBB extent per blob to sort by size. Blobs whose
    horizontal aspect ratio is very non-square (max/min > 1.8) are treated as
    "unlikely to be a cube" and used only as fallbacks. Distractor cylinders
    still show up as square-ish from above, but their diameter (0.03-0.05 m)
    and detection area tend to be smaller than a 5-cm cube; a 3rd-largest blob
    is always dropped, which handles distractor cases naturally.
    """

    if len(detections) < 2:
        raise RuntimeError(
            f"cube_stack needs >=2 workspace blobs, got {len(detections)}"
        )

    scored: list[dict[str, Any]] = []
    for det in detections:
        interior = erode_mask(det["mask"], iterations=1)
        try:
            geom = estimate_object_geometry(
                points_camera, interior, camera_to_base, CUBE_A_SIDE_M
            )
        except RuntimeError:
            continue
        extent_xy = np.sort(geom["obb_extent"][:2])
        # Robust footprint: geometric mean of the two horizontal extents.
        footprint = float(np.sqrt(extent_xy[0] * extent_xy[1]))
        aspect = float(extent_xy[1] / max(extent_xy[0], 1e-6))
        scored.append(
            {
                **det,
                "interior": interior,
                "geometry": geom,
                "footprint": footprint,
                "aspect": aspect,
            }
        )

    if len(scored) < 2:
        raise RuntimeError(
            f"only {len(scored)} usable workspace blobs after geometry fit"
        )

    # Prefer square-ish (aspect <= 1.6) blobs, and among them the two largest.
    squareish = [item for item in scored if item["aspect"] <= 1.6]
    pool = squareish if len(squareish) >= 2 else scored
    pool.sort(key=lambda item: item["footprint"], reverse=True)
    # Base = largest, target = second-largest. Anything past top-2 is dropped
    # (this is the distractor-object rejection channel).
    base = pool[0]
    target = pool[1]
    record(
        "cube_pair",
        base_footprint=base["footprint"],
        base_aspect=base["aspect"],
        base_center_xy=base["geometry"]["center"][:2],
        target_footprint=target["footprint"],
        target_aspect=target["aspect"],
        target_center_xy=target["geometry"]["center"][:2],
        rejected=len(scored) - 2,
    )
    return target, base


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def save_detection_overlay(
    phase: str,
    rgb: np.ndarray,
    detections: list[dict[str, Any]],
) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8).copy()
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    palette = [("target", "orange"), ("base", "cyan")]
    for det, (label, color) in zip(detections, palette, strict=False):
        box = det.get("box")
        if box is not None and len(box) == 4:
            draw.rectangle(tuple(float(v) for v in box), outline=color, width=3)
        center = np.asarray(det["geometry"]["center"])
        draw.text(
            (float(box[0]) + 4, float(box[1]) + 4),
            f"{label}\nfootprint={det['footprint']:.3f}\ncenter={np.round(center, 3).tolist()}",
            fill=color,
        )
    canvas.save(ARTIFACT_DIR / f"{_safe_name(phase)}_pair_perception.png")


def perceive_pair(phase: str) -> dict[str, Any]:
    observation = get_observation()
    camera = observation[CAMERA_NAME]
    rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
    depth = _depth_2d(camera["images"]["depth"])

    points_camera, valid = deproject_depth_image(depth, camera["intrinsics"])
    height, width = valid.shape
    points_base = transform_points(
        camera["pose_mat"], points_camera.reshape(-1, 3)
    ).reshape(height, width, 3)
    flat_points = points_base.reshape(-1, 3)
    finite = np.isfinite(flat_points).all(axis=1)
    if int(finite.sum()) < 1000:
        raise RuntimeError("insufficient finite depth points to estimate table")

    table_height = estimate_table_height(flat_points[finite])
    fg_mask = workspace_foreground_mask(points_base, valid, table_height)
    detections = component_detections(fg_mask, points_base)
    target, base = select_cube_pair(detections, points_camera, camera["pose_mat"])
    save_detection_overlay(phase, rgb, [target, base])

    return {
        "phase": phase,
        "observation": observation,
        "camera": camera,
        "rgb": rgb,
        "depth": depth,
        "table_height": table_height,
        "target": target,
        "base": base,
        "detection_count": len(detections),
    }


def perceive_single(phase: str) -> dict[str, Any]:
    """Diagnostic-only single-blob perception for the final verification pass.

    When cubeA is stacked on cubeB, the pair may collapse into a single blob;
    detection_count may drop to 1 and select_cube_pair would raise. This helper
    returns whichever blob is topmost.
    """

    observation = get_observation()
    camera = observation[CAMERA_NAME]
    rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
    depth = _depth_2d(camera["images"]["depth"])
    points_camera, valid = deproject_depth_image(depth, camera["intrinsics"])
    height, width = valid.shape
    points_base = transform_points(
        camera["pose_mat"], points_camera.reshape(-1, 3)
    ).reshape(height, width, 3)
    flat_points = points_base.reshape(-1, 3)
    finite = np.isfinite(flat_points).all(axis=1)
    if int(finite.sum()) < 1000:
        raise RuntimeError("insufficient finite depth points in final check")
    table_height = estimate_table_height(flat_points[finite])
    fg_mask = workspace_foreground_mask(points_base, valid, table_height)
    detections = component_detections(fg_mask, points_base)
    if not detections:
        raise RuntimeError("no foreground blobs in final check")
    detections.sort(key=lambda item: item["top_z"], reverse=True)
    top = detections[0]
    return {
        "phase": phase,
        "top_center_xy": top["center_xy"].tolist(),
        "top_z": top["top_z"],
        "table_height": table_height,
        "blob_count": len(detections),
    }


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def rank_grasp_candidates(
    grasp_poses_camera: np.ndarray,
    grasp_scores: np.ndarray,
    camera_to_base: np.ndarray,
    object_center: np.ndarray,
) -> list[dict[str, Any]]:
    poses = np.asarray(grasp_poses_camera, dtype=np.float64)
    scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
        return []

    camera_transform = np.asarray(camera_to_base, dtype=np.float64)
    center = np.asarray(object_center, dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for index, (pose_camera, score) in enumerate(zip(poses, scores, strict=True)):
        pose_base = camera_transform @ pose_camera
        position = pose_base[:3, 3]
        approach_axis = pose_base[:3, 2]
        downness = float(-approach_axis[2])
        center_distance = float(np.linalg.norm(position - center))
        finite = np.isfinite(pose_base).all() and np.isfinite(score)
        reachable = (
            WORKSPACE_X_RANGE[0] < position[0] < WORKSPACE_X_RANGE[1]
            and WORKSPACE_Y_RANGE[0] < position[1] < WORKSPACE_Y_RANGE[1]
        )
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


def plan_object_grasp(
    perception: dict[str, Any], detection: dict[str, Any]
) -> dict[str, Any]:
    try:
        poses, scores = plan_grasp(
            depth=perception["depth"],
            intrinsics=perception["camera"]["intrinsics"],
            segmentation=detection["mask"].astype(np.int32),
        )
        candidates = rank_grasp_candidates(
            poses,
            scores,
            perception["camera"]["pose_mat"],
            detection["geometry"]["center"],
        )
    except Exception as exc:
        record("graspnet_error", error=repr(exc))
        candidates = []

    grasp = candidates[0] if candidates else analytic_top_down_grasp(
        detection["geometry"]["center"]
    )
    record(
        "grasp_selected",
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


def pick_object(
    detection: dict[str, Any], grasp: dict[str, Any]
) -> np.ndarray:
    """Approach, grasp, lift. Returns center_minus_grasp offset (base frame)."""

    grasp_pos = np.asarray(grasp["position"], dtype=np.float64)
    quaternion = np.asarray(grasp["quaternion"], dtype=np.float64)
    approach_axis = np.asarray(grasp["approach_axis"], dtype=np.float64)
    pregrasp = grasp_pos - APPROACH_DISTANCE_M * approach_axis

    open_gripper()
    move_pose("pregrasp_target", pregrasp, quaternion)
    move_pose("grasp_target", grasp_pos, quaternion)
    close_gripper()
    move_pose(
        "lift_target",
        grasp_pos + np.array([0.0, 0.0, LIFT_CLEARANCE_M]),
        quaternion,
    )
    object_center = np.asarray(detection["geometry"]["center"], dtype=np.float64)
    return object_center - grasp_pos


def place_on_base(
    base_detection: dict[str, Any],
    target_side_m: float,
    center_minus_grasp: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    base_top_z = float(base_detection["geometry"]["top_z"])
    base_center_xy = np.asarray(
        base_detection["geometry"]["center"][:2], dtype=np.float64
    )
    desired_target_center = np.array(
        [
            base_center_xy[0],
            base_center_xy[1],
            base_top_z + 0.5 * target_side_m + RELEASE_CLEARANCE_M,
        ],
        dtype=np.float64,
    )
    desired_grasp_position = desired_target_center - np.asarray(
        center_minus_grasp, dtype=np.float64
    )
    preplace = desired_grasp_position + np.array([0.0, 0.0, 0.10])
    move_pose("preplace_target", preplace, quaternion)
    move_pose("place_target", desired_grasp_position, quaternion)
    open_gripper()
    move_pose("retract_target", preplace, quaternion)
    record("place_target_command", desired_center=desired_target_center)


def main() -> None:
    global RESULT

    np.random.seed(0)
    record("pipeline_start", mode="S3", llm_calls=0)

    perception = perceive_pair("initial")
    target = perception["target"]
    base = perception["base"]
    target_side = CUBE_A_SIDE_M

    grasp = plan_object_grasp(perception, target)
    center_minus_grasp = pick_object(target, grasp)
    place_on_base(
        base,
        target_side,
        center_minus_grasp,
        np.asarray(grasp["quaternion"], dtype=np.float64),
    )

    diagnostics: dict[str, Any] = {
        "target_footprint_m": target["footprint"],
        "base_footprint_m": base["footprint"],
        "base_center_xy": base["geometry"]["center"][:2].tolist(),
    }
    try:
        final = perceive_single("final")
        stack_height = float(final["top_z"] - perception["table_height"])
        expected_min = base["geometry"]["top_z"] - perception["table_height"] + 0.5 * target_side
        diagnostics.update(
            {
                "final_top_center_xy": final["top_center_xy"],
                "final_top_z": final["top_z"],
                "stack_height_m": stack_height,
                "expected_min_stack_height_m": expected_min,
                "final_blob_count": final["blob_count"],
            }
        )
        record(
            "final_diagnostics",
            stack_height_m=stack_height,
            expected_min_stack_height_m=expected_min,
            final_blob_count=final["blob_count"],
        )
    except Exception as exc:
        diagnostics["verification_error"] = repr(exc)
        record("verification_warning", error=repr(exc))

    RESULT = _jsonable(
        {
            "mode": "S3",
            "llm_calls": 0,
            "grasp_source": grasp["source"],
            "table_height_m": perception["table_height"],
            **diagnostics,
        }
    )
    record("pipeline_complete", result=RESULT)


if __name__ == "__main__":
    main()
