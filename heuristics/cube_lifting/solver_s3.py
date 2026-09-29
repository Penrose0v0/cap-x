"""Deterministic S3 solver for ``cube_lifting``.

Single-cube variant of the cube_restack solver: perceive one object on the
table, grasp it with Contact-GraspNet (analytic fallback), lift it clear.
No LLM/VLM query, no privileged state.

Robustness strategy across the variant set:
- baseline: red cube (dominant red channel)
- color_shift_mild / strong: cube retinted; red-dominance alone would fail on
  color_shift_strong (green tint), so we fall back to a workspace-foreground
  detector (isolate points on the table that are not the table plane).
- target_object_swap: cube replaced by a cylinder; shape-agnostic foreground
  detection still finds it, and Contact-GraspNet handles the pose.
- distractor_object: an additional cylinder present; pick whichever workspace
  cluster is closest to the workspace center (the target keeps its default
  sampling range, distractors are placed off to the side).
- lighting_*: perception thresholds are dominance-based (color-channel deltas)
  rather than absolute intensity, so they degrade gracefully.

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

# Robosuite's Lift cube is 4-cm on each side (see robosuite/environments/manipulation/lift.py
# `BoxObject(size=[0.020, 0.020, 0.020])` -> full extent 4 cm). Cylinder swap has similar
# footprint. Height threshold for success is 0.04 m above table (upstream Lift._check_success),
# and 0.12 m for the `lift_higher` difficulty knob preset.
DEFAULT_OBJECT_SIDE_M = 0.040
APPROACH_DISTANCE_M = 0.090
LIFT_DISTANCE_M = 0.180  # comfortably above both 0.04 m and 0.12 m thresholds
RELEASE_CLEARANCE_M = 0.001

MIN_MASK_PIXELS = 60
MIN_TOP_DOWN_COSINE = 0.55
TOP_DOWN_QUATERNION_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

# Workspace bounds in robot-base frame (xy). Wide enough to cover the default lift sampling
# range plus every variant's position bias.
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
    print(f"[cube_lifting:s3] {event}" + (f": {compact}" if compact else ""))
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


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def deproject_depth_image(
    depth: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_camera [H*W, 3], validity mask [H, W])."""

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
    """Estimate the table plane as the dominant Z mode in the workspace region."""

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
    """Pixel mask of everything sitting on the table inside the workspace.

    Shape-agnostic: works whether the target is a cube, cylinder, etc. Rejects
    the table itself (Z near table_height) and the robot arm (Z well above the
    table but outside a workspace-XY cluster window; safe here because the
    solver only calls this before the arm moves near the target).
    """

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


def select_target_detection(
    detections: list[dict[str, Any]], rgb: np.ndarray
) -> dict[str, Any]:
    """Pick which foreground blob is the lift target.

    Ranking: (red-channel dominance for the baseline / mild color-shift cases,
    where it's a strong signal) + (proximity to workspace center, to prefer the
    task cube over distractors placed near the workspace edge) + (pixel count,
    to prefer solid objects over noise).
    """

    if not detections:
        raise RuntimeError("no workspace foreground detections")

    image = np.asarray(rgb, dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    red_dominance = (red - np.maximum(green, blue)).astype(np.float64) / 255.0
    workspace_center_xy = np.array(
        [0.5 * (WORKSPACE_X_RANGE[0] + WORKSPACE_X_RANGE[1]), 0.0], dtype=np.float64
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for det in detections:
        mask = det["mask"]
        red_score = float(np.clip(np.median(red_dominance[mask]), -0.5, 0.5))
        center_dist = float(np.linalg.norm(det["center_xy"] - workspace_center_xy))
        pixel_bonus = min(det["pixels"] / 2000.0, 1.0)
        # Prioritize centrality (targets stay near default sampling center), then
        # solidity, then red-dominance as a mild tiebreaker.
        rank = -1.5 * center_dist + 0.35 * pixel_bonus + 0.20 * red_score
        det["rank_terms"] = {
            "red_dominance": red_score,
            "center_distance_m": center_dist,
            "pixel_bonus": pixel_bonus,
            "rank": rank,
        }
        scored.append((rank, det))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def estimate_object_geometry(
    points_camera: np.ndarray, mask: np.ndarray, camera_to_base: np.ndarray
) -> dict[str, Any]:
    """Get a robust center + top-Z + OBB for the target from its mask points."""

    valid_mask = mask & np.isfinite(points_camera[:, :, 2])
    cluster_camera = points_camera[valid_mask]
    if len(cluster_camera) < MIN_MASK_PIXELS:
        raise RuntimeError(f"only {len(cluster_camera)} valid RGB-D pixels in target mask")

    points_base = transform_points(camera_to_base, cluster_camera)
    obb = get_oriented_bounding_box_from_3d_points(points_base)
    obb_center_base = np.asarray(obb["center"], dtype=np.float64)

    top_z = float(np.quantile(points_base[:, 2], 0.90))
    top_band = points_base[points_base[:, 2] >= top_z - 0.004]
    if len(top_band) >= 20:
        low_xy = np.quantile(top_band[:, :2], 0.05, axis=0)
        high_xy = np.quantile(top_band[:, :2], 0.95, axis=0)
        center_xy = 0.5 * (low_xy + high_xy)
    else:
        center_xy = obb_center_base[:2]

    # Object body center (roughly the geometric middle rather than the top).
    center_base = np.array(
        [center_xy[0], center_xy[1], top_z - 0.5 * DEFAULT_OBJECT_SIDE_M],
        dtype=np.float64,
    )
    return {
        "center": center_base,
        "top_z": top_z,
        "obb_center": obb_center_base,
        "obb_extent": np.asarray(obb["extent"], dtype=np.float64),
        "points_base": points_base,
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def save_detection_overlay(
    phase: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    detection: dict[str, Any],
    geometry: dict[str, Any],
) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8).copy()
    tint = np.array([255, 200, 60], dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    image[mask_bool] = np.clip(0.55 * image[mask_bool] + 0.45 * tint, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    box = detection.get("box")
    if box is not None and len(box) == 4:
        draw.rectangle(tuple(float(v) for v in box), outline="yellow", width=3)
    center = np.asarray(geometry["center"])
    label = (
        f"{phase}: target  rank={detection.get('rank_terms', {}).get('rank', 0.0):.3f}\n"
        f"center={np.round(center, 4).tolist()}  top_z={geometry['top_z']:.4f}"
    )
    draw.rectangle((4, 4, min(canvas.width - 4, 505), 42), fill=(0, 0, 0))
    draw.text((8, 7), label, fill="white")
    canvas.save(ARTIFACT_DIR / f"{_safe_name(phase)}_target_perception.png")


def perceive_target(phase: str) -> dict[str, Any]:
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
    target = select_target_detection(detections, rgb)

    interior_mask = erode_mask(target["mask"], iterations=1)
    geometry = estimate_object_geometry(points_camera, interior_mask, camera["pose_mat"])

    result = {
        "phase": phase,
        "observation": observation,
        "camera": camera,
        "rgb": rgb,
        "depth": depth,
        "table_height": table_height,
        "mask": target["mask"],
        "detection": target,
        **geometry,
    }
    record(
        "perception",
        phase=phase,
        center=result["center"],
        top_z=result["top_z"],
        table_height=table_height,
        obb_extent=result["obb_extent"],
        mask_pixels=int(target["mask"].sum()),
        rank_terms=target.get("rank_terms"),
        detection_count=len(detections),
    )
    save_detection_overlay(phase, rgb, target["mask"], target, result)
    return result


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


def plan_target_grasp(perception: dict[str, Any]) -> dict[str, Any]:
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
        record("graspnet_error", error=repr(exc))
        candidates = []

    grasp = candidates[0] if candidates else analytic_top_down_grasp(perception["center"])
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


def lift_target(perception: dict[str, Any], grasp: dict[str, Any]) -> None:
    position = np.asarray(grasp["position"], dtype=np.float64)
    quaternion = np.asarray(grasp["quaternion"], dtype=np.float64)
    approach_axis = np.asarray(grasp["approach_axis"], dtype=np.float64)
    pregrasp = position - APPROACH_DISTANCE_M * approach_axis

    open_gripper()
    move_pose("pregrasp", pregrasp, quaternion)
    move_pose("grasp", position, quaternion)
    close_gripper()
    move_pose(
        "lift",
        position + np.array([0.0, 0.0, LIFT_DISTANCE_M]),
        quaternion,
    )


def main() -> None:
    global RESULT

    np.random.seed(0)
    record("pipeline_start", mode="S3", llm_calls=0)

    perception = perceive_target("initial")
    grasp = plan_target_grasp(perception)
    lift_target(perception, grasp)

    # Non-fatal post-condition diagnostics.
    diagnostics: dict[str, Any] = {}
    try:
        final = perceive_target("final")
        height_delta = float(final["center"][2] - perception["table_height"])
        diagnostics.update(
            {
                "final_center": final["center"],
                "final_top_z": final["top_z"],
                "height_delta_m": height_delta,
                "expected_min_delta_m": 0.04,
            }
        )
        record(
            "final_diagnostics",
            height_delta_m=height_delta,
            expected_min_delta_m=0.04,
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
            "initial_center": perception["center"],
            **diagnostics,
        }
    )
    record("pipeline_complete", result=RESULT)


if __name__ == "__main__":
    main()
