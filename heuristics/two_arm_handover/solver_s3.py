"""Deterministic visual S3 heuristic for two-arm hammer handover."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


CAMERA_NAME = "agentview"
ARM0_GRASP_WXYZ = np.array([0.0, 0.70710678, 0.70710678, 0.0])
ARM0_HANDOVER_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
ARM1_GRASP_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
HANDOVER_X = 0.81
HANDOVER_Z = 0.19
TRACE: list[dict[str, Any]] = []


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def record(event: str, **values: Any) -> None:
    entry = {"event": event, **{key: _plain(value) for key, value in values.items()}}
    TRACE.append(entry)
    print(f"[two_arm_handover:s3] {json.dumps(entry)}", flush=True)


def deproject_mask(depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z > 0.05) & (z < 4.0)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    x = (cols - matrix[0, 2]) * z / matrix[0, 0]
    y = (rows - matrix[1, 2]) * z / matrix[1, 1]
    return np.column_stack([x, y, z])


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def detect_hammer(rgb: np.ndarray) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for query in ("hammer", "mallet", "hand tool with a handle"):
        try:
            for detection in detect_object_owlvit(rgb, query):  # noqa: F821
                item = dict(detection)
                item["query"] = query
                candidates.append(item)
        except Exception as exc:
            record("detection_error", query=query, error=repr(exc))
    if not candidates:
        raise RuntimeError("OWL-ViT found no hammer-like object")
    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = candidates[0]
    record("detection", query=best["query"], score=best.get("score"), box=best.get("box"))
    return best


def hammer_geometry() -> dict[str, Any]:
    observation = get_observation()  # noqa: F821
    camera = observation[CAMERA_NAME]
    rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
    detection = detect_hammer(rgb)
    masks = segment_sam2(rgb, box=detection["box"])  # noqa: F821
    masks = [result for result in masks if "mask" in result]
    if not masks:
        raise RuntimeError("SAM2 returned no hammer mask")
    result = max(masks, key=lambda item: float(item.get("score", 0.0)))
    mask = np.asarray(result["mask"], dtype=bool)
    points_camera = deproject_mask(camera["images"]["depth"], camera["intrinsics"], mask)
    if len(points_camera) < 50:
        raise RuntimeError(f"hammer mask has only {len(points_camera)} depth points")
    points = transform_points(camera["pose_mat"], points_camera)
    points = points[np.isfinite(points).all(axis=1)]

    low = np.quantile(points, 0.03, axis=0)
    high = np.quantile(points, 0.97, axis=0)
    center = 0.5 * (low + high)
    length_y = float(high[1] - low[1])
    # The task fixes the hammer axis along Y: head at -Y, handle at +Y.
    arm0_grasp_y = float(low[1] + min(0.06, 0.28 * length_y))
    handle_tip_y = float(high[1])
    geometry = {
        "center": center,
        "low": low,
        "high": high,
        "length_y": length_y,
        "arm0_grasp": np.array([center[0], arm0_grasp_y, center[2]]),
        "handle_from_grip": max(0.10, handle_tip_y - arm0_grasp_y),
    }
    record("geometry", **geometry)
    return geometry


def move_arm0(position: np.ndarray, quaternion: np.ndarray) -> None:
    joints = np.asarray(solve_ik_arm0(position, quaternion), dtype=np.float64).reshape(7)  # noqa: F821
    move_to_joints_arm0(joints)  # noqa: F821


def move_arm1(position: np.ndarray, quaternion: np.ndarray) -> None:
    joints = np.asarray(solve_ik_arm1(position, quaternion), dtype=np.float64).reshape(7)  # noqa: F821
    move_to_joints_arm1(joints)  # noqa: F821


def main() -> None:
    record("pipeline_start", mode="S3", llm_calls=0, privileged_state=False)
    geometry = hammer_geometry()
    grasp = np.asarray(geometry["arm0_grasp"], dtype=np.float64)

    open_gripper_arm0()  # noqa: F821
    move_arm0(grasp + np.array([0.0, 0.0, 0.10]), ARM0_GRASP_WXYZ)
    move_arm0(grasp, ARM0_GRASP_WXYZ)
    close_gripper_arm0()  # noqa: F821

    lift = np.array([grasp[0], grasp[1], HANDOVER_Z])
    move_arm0(lift, ARM0_GRASP_WXYZ)
    handover = np.array([HANDOVER_X, grasp[1], HANDOVER_Z])
    move_arm0(handover, ARM0_GRASP_WXYZ)
    move_arm0(handover, ARM0_HANDOVER_WXYZ)

    arm1_x = HANDOVER_X + max(0.10, 0.60 * float(geometry["handle_from_grip"]))
    arm1_grasp = np.array([arm1_x, grasp[1], HANDOVER_Z])
    open_gripper_arm1()  # noqa: F821
    move_arm1(arm1_grasp + np.array([0.0, 0.0, 0.10]), ARM1_GRASP_WXYZ)
    move_arm1(arm1_grasp, ARM1_GRASP_WXYZ)
    close_gripper_arm1()  # noqa: F821
    open_gripper_arm0()  # noqa: F821
    record("handover_complete", arm1_grasp=arm1_grasp)


main()
