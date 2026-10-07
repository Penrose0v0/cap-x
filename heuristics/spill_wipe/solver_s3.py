"""Hard-coded S3 heuristic for robosuite ``spill_wipe``: wipe the whole dirt region.

No perception of the spill at all. The WipeArena samples its dirt markers inside a
fixed rectangle (table_half_size * coverage_factor around the arena origin), and the
robot base is always placed at the same offset from the table, so the marker region is
a known rectangle in the robot base frame. The solver rasters that entire rectangle
twice (cross-hatched) so the wiping pad covers it regardless of pad orientation.

Only the depth image is used, to measure the table height; if that fails a fixed
fallback is used.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from heuristics.common import *  # noqa: F401,F403 -- inlined by static_solver_server


CAMERA_NAME = "robot0_robotview"
DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

# Marker region in the robot base frame: arena origin is 0.46 m ahead of the base,
# markers span +-0.15 m in x and +-0.24 m in y (table_half_size * coverage_factor).
REGION_LOW_XY = np.array([0.28, -0.27], dtype=np.float64)
REGION_HIGH_XY = np.array([0.64, 0.27], dtype=np.float64)
TABLE_Z_FALLBACK_M = -0.017

APPROACH_HEIGHT_M = 0.055
# Commanding the pad slightly above the table keeps it flat; pressing into the table tilts
# the wrist and lifts the pad plane out of the marker-detection band.
WIPE_HOVER_M = 0.006
# The wiping pad's 0.12 m long side lies along base y, so strips along x sweep a 0.12 m swath.
STRIP_SPACING_M = 0.045
NUM_PASSES = 2
WAYPOINT_STEP_M = 0.03
TRACE: list[dict[str, Any]] = []



record = make_recorder("spill_wipe", TRACE, json_lines=True)



def deproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    rows, cols = np.mgrid[0:height, 0:width]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    x = (cols - matrix[0, 2]) * depth / matrix[0, 0]
    y = (rows - matrix[1, 2]) * depth / matrix[1, 1]
    return np.stack([x, y, depth], axis=-1)



def measure_table_z() -> float:
    try:
        observation = get_observation()  # noqa: F821
        camera = observation[CAMERA_NAME]
        depth = depth_2d(camera["images"]["depth"])
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
        points = transform_points(camera["pose_mat"], deproject(depth, camera["intrinsics"])[valid])
        in_region = (
            (points[:, 0] >= REGION_LOW_XY[0])
            & (points[:, 0] <= REGION_HIGH_XY[0])
            & (points[:, 1] >= REGION_LOW_XY[1])
            & (points[:, 1] <= REGION_HIGH_XY[1])
        )
        region_z = points[in_region, 2]
        region_z = region_z[np.isfinite(region_z)]
        if len(region_z) < 100:
            raise RuntimeError(f"only {len(region_z)} depth points in the table region")
        table_z = float(np.median(region_z))
        if abs(table_z - TABLE_Z_FALLBACK_M) > 0.08:
            raise RuntimeError(f"implausible table z {table_z:.3f}")
        record("table_z", table_z=table_z, points=int(len(region_z)), source="depth")
        return table_z
    except Exception as exc:
        record("table_z", table_z=TABLE_Z_FALLBACK_M, source="fallback", error=repr(exc))
        return TABLE_Z_FALLBACK_M


def move(position: np.ndarray) -> None:
    target = np.asarray(position, dtype=np.float64).reshape(3)
    joints = np.asarray(solve_ik(target, DOWN_WXYZ), dtype=np.float64).reshape(7)  # noqa: F821
    if not np.isfinite(joints).all():
        raise RuntimeError(f"IK returned non-finite joints for {target}")
    move_to_joints(joints)  # noqa: F821


def inclusive_axis(start: float, stop: float, max_step: float) -> np.ndarray:
    count = max(1, int(np.ceil(abs(stop - start) / max_step)))
    return np.linspace(start, stop, count + 1)


def raster_pass(pass_index: int, wipe_z: float) -> int:
    offset = STRIP_SPACING_M * pass_index / NUM_PASSES
    strips = inclusive_axis(
        float(REGION_LOW_XY[1]) + offset,
        float(REGION_HIGH_XY[1]) - STRIP_SPACING_M + offset,
        STRIP_SPACING_M,
    )
    xs_forward = inclusive_axis(float(REGION_LOW_XY[0]), float(REGION_HIGH_XY[0]), WAYPOINT_STEP_M)
    moves = 0
    for row, y_value in enumerate(strips):
        xs = xs_forward if row % 2 == 0 else xs_forward[::-1]
        for x_value in xs:
            move(np.array([x_value, y_value, wipe_z]))
            moves += 1
    record("raster_pass_complete", pass_index=pass_index, strips=len(strips), points_per_strip=len(xs_forward), moves=moves)
    return moves


def main() -> None:
    record("pipeline_start", mode="S3", llm_calls=0, privileged_state=False, strategy="full_region_raster")
    table_z = measure_table_z()
    wipe_z = table_z + WIPE_HOVER_M
    record("wipe_region", low_xy=REGION_LOW_XY, high_xy=REGION_HIGH_XY, table_z=table_z, wipe_z=wipe_z)

    move(np.array([REGION_LOW_XY[0], REGION_LOW_XY[1], wipe_z + APPROACH_HEIGHT_M]))
    move(np.array([REGION_LOW_XY[0], REGION_LOW_XY[1], wipe_z]))

    total = 0
    for pass_index in range(NUM_PASSES):
        total += raster_pass(pass_index, wipe_z)

    center = 0.5 * (REGION_LOW_XY + REGION_HIGH_XY)
    move(np.array([center[0], center[1], wipe_z + APPROACH_HEIGHT_M]))
    record("raster_complete", total_moves=total)


main()
