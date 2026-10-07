"""Hard-coded S3 heuristic for robosuite ``two_arm_lift``: depth-only pot pose + known geometry.

No semantic segmentation. The pot always spawns within a few cm of a known spot on the
table and only its yaw is randomised, and the PotWithHandles geometry is fixed: the two
graspable handle bars sit 0.16 m from the pot centre along the pot's long axis, 0.13 m
above the table. So the solver only needs the pot's yaw, which it reads from the depth
image: points in the handle-height band that lie outside the pot body form two clusters,
one per handle; their midpoint is the pot centre and the line through them is the axis.

Both arms then do a top-down grasp on the bars with the finger-close axis aligned to the
handle axis, and lift together.

API surface (FrankaControlApiReducedBimanual): get_observation, solve_ik_arm0/arm1 (both
take arm0-base-frame poses), move_to_joints_both, open/close_gripper_arm0/arm1.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from heuristics.common import *  # noqa: F401,F403 -- inlined by static_solver_server


CAMERA_NAME = "robot0_robotview"

# Pot spawn centre in arm0's base frame (spawn jitter is only +-0.03 m).
NOMINAL_POT_XY = np.array([0.55, 0.0], dtype=np.float64)
# PotWithHandles defaults: body half-size 0.07, handle_length 0.09 -> bar at 0.16 from centre,
# bar centre at body_half_z + (body_half_z - handle_radius) = 0.13 above the table.
HANDLE_OFFSET_M = 0.16
BAR_HEIGHT_ABOVE_TABLE_M = 0.13
BODY_RADIUS_M = 0.11
HANDLE_RADIUS_RANGE_M = (0.115, 0.20)
OUTER_BAR_MIN_RADIUS_M = 0.145
HANDLE_Z_BAND_M = (0.10, 0.16)

Z_APPROACH_M = 0.15
LIFT_HEIGHT_M = 0.20
MIN_HANDLE_POINTS = 200
TRACE: list[dict[str, Any]] = []



record = make_recorder("two_arm_lift", TRACE, json_lines=True)



def deproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    rows, cols = np.mgrid[0:height, 0:width]
    K = np.asarray(intrinsics, dtype=np.float64)
    x = (cols - K[0, 2]) * depth / K[0, 0]
    y = (rows - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1)



def perceive_pot() -> tuple[float, np.ndarray, np.ndarray]:
    """Return (table_z, pot_center_xy, handle_axis_xy) in arm0's base frame."""

    observation = get_observation()  # noqa: F821
    camera = observation[CAMERA_NAME]
    depth = depth_2d(camera["images"]["depth"])
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    points = transform_points(camera["pose_mat"], deproject(depth, camera["intrinsics"])[valid])
    points = points[np.isfinite(points).all(axis=1)]

    near = points[np.linalg.norm(points[:, :2] - NOMINAL_POT_XY, axis=1) < 0.45]
    lower_half = near[near[:, 2] < np.quantile(near[:, 2], 0.5)]
    table_z = float(np.median(lower_half[:, 2]))

    in_band = (near[:, 2] > table_z + HANDLE_Z_BAND_M[0]) & (near[:, 2] < table_z + HANDLE_Z_BAND_M[1])
    body = near[in_band & (np.linalg.norm(near[:, :2] - NOMINAL_POT_XY, axis=1) < BODY_RADIUS_M)]
    if len(body) < MIN_HANDLE_POINTS:
        raise RuntimeError(f"only {len(body)} pot body points near the nominal spawn")
    center = np.median(body[:, :2], axis=0)

    # The far handle is often partly hidden behind the pot body from the oblique camera, so
    # cluster medians are biased. Instead: the along-axis position comes from the outermost
    # points of both handles (the outer bars' far edges are the least occluded part), and the
    # lateral position from the two ends of the better-visible handle's outer bar.
    handle_count = [0, 0]
    for _ in range(4):
        radius = np.linalg.norm(near[:, :2] - center, axis=1)
        handles = near[in_band & (radius > HANDLE_RADIUS_RANGE_M[0]) & (radius < HANDLE_RADIUS_RANGE_M[1]), :2]
        if len(handles) < MIN_HANDLE_POINTS:
            raise RuntimeError(f"only {len(handles)} handle points")
        offsets = handles - center
        _, eigvecs = np.linalg.eigh(np.cov(offsets.T))
        axis = eigvecs[:, -1]
        perp_axis = np.array([-axis[1], axis[0]])
        projection = offsets @ axis
        lo, hi = np.quantile(projection, 0.02), np.quantile(projection, 0.98)
        center = center + axis * 0.5 * (lo + hi)

        offsets = handles - center
        projection = offsets @ axis
        handle_count = [int((projection > 0).sum()), int((projection < 0).sum())]
        sign = 1.0 if handle_count[0] >= handle_count[1] else -1.0
        outer_bar = offsets[sign * projection > OUTER_BAR_MIN_RADIUS_M]
        if len(outer_bar) >= 30:
            perp = outer_bar @ perp_axis
            center = center + perp_axis * 0.5 * (np.quantile(perp, 0.02) + np.quantile(perp, 0.98))

    record(
        "pot_pose",
        table_z=table_z,
        center=center,
        axis=axis,
        span=hi - lo,
        body_points=len(body),
        handle_points=handle_count,
    )
    return table_z, center, axis


def top_down_quat_for_axis(handle_axis_xy: np.ndarray) -> np.ndarray:
    """Top-down grip with the finger-close axis (gripper +Y) parallel to the handle axis."""

    d = np.asarray(handle_axis_xy, dtype=np.float64).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-12)
    fy = np.array([d[0], d[1], 0.0], dtype=np.float64)
    fz = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    fx = np.cross(fy, fz)
    fx = fx / (np.linalg.norm(fx) + 1e-12)
    fy = np.cross(fz, fx)
    return rotation_matrix_to_quat_wxyz(np.column_stack([fx, fy, fz]))


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def move_both(pos0: np.ndarray, quat0: np.ndarray, pos1: np.ndarray, quat1: np.ndarray) -> None:
    j0 = np.asarray(solve_ik_arm0(pos0, quat0), dtype=np.float64).reshape(7)  # noqa: F821
    j1 = np.asarray(solve_ik_arm1(pos1, quat1), dtype=np.float64).reshape(7)  # noqa: F821
    if not (np.isfinite(j0).all() and np.isfinite(j1).all()):
        raise RuntimeError(f"IK non-finite for pos0={pos0}, pos1={pos1}")
    move_to_joints_both(j0, j1)  # noqa: F821


def main() -> None:
    record("pipeline_start", mode="S3", llm_calls=0, privileged_state=False, strategy="depth_handle_clusters")
    table_z, center, axis = perceive_pot()

    bar_z = table_z + BAR_HEIGHT_ABOVE_TABLE_M
    handle_plus = np.array([*(center + HANDLE_OFFSET_M * axis), bar_z], dtype=np.float64)
    handle_minus = np.array([*(center - HANDLE_OFFSET_M * axis), bar_z], dtype=np.float64)
    quat_plus = top_down_quat_for_axis(axis)
    quat_minus = top_down_quat_for_axis(-axis)
    record("grasp_targets", handle_plus=handle_plus, handle_minus=handle_minus, quat_plus=quat_plus, quat_minus=quat_minus)

    lift = np.array([0.0, 0.0, Z_APPROACH_M])
    open_gripper_arm0()  # noqa: F821
    open_gripper_arm1()  # noqa: F821
    move_both(handle_plus + lift, quat_plus, handle_minus + lift, quat_minus)
    record("approach_done")
    move_both(handle_plus, quat_plus, handle_minus, quat_minus)
    record("grip_pose_done")
    close_gripper_arm0()  # noqa: F821
    close_gripper_arm1()  # noqa: F821

    lift = np.array([0.0, 0.0, LIFT_HEIGHT_M])
    move_both(handle_plus + lift, quat_plus, handle_minus + lift, quat_minus)
    record("lift_complete", lift_m=LIFT_HEIGHT_M)


main()
