"""Camera-orbit perturbation shared by the 7 `*_variants.py` task envs.

Moves the observation camera on a sphere around the point where its original optical axis
meets the table top, the same way LIBERO's `relocate_camera` pipeline does: the three knobs
are *deltas* from the original camera, so (0, 0, 0) reproduces it exactly.

  - radius:        metres along the view axis; + backs the camera away from the table
  - azimuth_deg:   rotation around world z, + is counter-clockwise seen from above
  - elevation_deg: + raises the camera (steeper, more top-down view)

The camera keeps looking at the same table point, so the scene stays framed; only the
viewpoint changes. This perturbs what the *policy sees* (perspective, occlusions, pixel
coordinates of everything) without touching the scene or the task, which is what
geometry/depth-based hard-coded solvers are blind to under the other presets.

The pose is written into the compiled model (`sim.model.cam_pos/cam_quat`) on every reset,
in the camera's parent-body frame, so it works both for arena cameras (`agentview`, parent =
world) and robot-mounted ones (`robot0_robotview`, parent = the robot base body). capx reads
camera extrinsics/intrinsics back from the sim at observation time, so depth deprojection
stays consistent automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite.utils.transform_utils as T


CAMERA_ORBIT_PRESETS: dict[str, dict[str, Any]] = {
    "camera_shift_mild": {"camera_orbit": {"radius": 0.10, "azimuth_deg": 10.0, "elevation_deg": 5.0}},
    "camera_shift_strong": {"camera_orbit": {"radius": 0.25, "azimuth_deg": 25.0, "elevation_deg": 12.0}},
}

TABLE_TOP_SITE = "table_top"


def capture_camera_reference(sim: Any, camera_name: str) -> dict[str, Any]:
    """Record the camera's original local pose. Call from `_setup_references` (i.e. after every
    model rebuild) so repeated resets orbit from the true original, never from a moved pose."""
    cam_id = sim.model.camera_name2id(camera_name)
    return {
        "name": camera_name,
        "cam_id": cam_id,
        "pos": np.array(sim.model.cam_pos[cam_id], dtype=np.float64),
        "quat": np.array(sim.model.cam_quat[cam_id], dtype=np.float64),
    }


def _spherical(offset: np.ndarray) -> tuple[float, float, float]:
    r = float(np.linalg.norm(offset))
    az = float(np.degrees(np.arctan2(offset[1], offset[0])))
    el = float(np.degrees(np.arcsin(np.clip(offset[2] / r, -1.0, 1.0))))
    return r, az, el


def _orbit_offset(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _orbit_rotation(az0: float, el0: float, az1: float, el1: float) -> np.ndarray:
    """Rigid rotation taking the spherical direction (az0, el0) to (az1, el1).

    The direction at (az, el) is Rz(az) @ Ry(-el) @ x_hat, so un-rotating the original to the
    canonical direction and rotating to the new one moves the camera frame rigidly. Applied to
    the camera's own rotation this keeps it looking at the pivot while preserving its original
    roll, is exactly the identity at zero delta, and has no singularity for a straight-down
    camera (where a look-at reconstruction would).
    """
    return _rot_z(az1) @ _rot_y(-el1) @ _rot_y(el0) @ _rot_z(-az0)


def apply_camera_orbit(sim: Any, reference: dict[str, Any], orbit: dict[str, Any] | None) -> None:
    """Apply the orbit deltas in `orbit` (radius / azimuth_deg / elevation_deg) to the camera
    recorded by `capture_camera_reference`. A None/empty `orbit` restores the original pose."""
    cam_id, name = reference["cam_id"], reference["name"]
    sim.model.cam_pos[cam_id] = reference["pos"]
    sim.model.cam_quat[cam_id] = reference["quat"]
    sim.forward()
    if not orbit:
        return

    pos0 = np.array(sim.data.get_camera_xpos(name), dtype=np.float64)
    rot0 = np.array(sim.data.get_camera_xmat(name), dtype=np.float64).reshape(3, 3)
    forward = -rot0[:, 2]

    table_z = float(sim.data.site_xpos[sim.model.site_name2id(TABLE_TOP_SITE)][2])
    t = (table_z - pos0[2]) / forward[2] if abs(forward[2]) > 1e-6 else -1.0
    if t <= 0:
        t = 1.0
    pivot = pos0 + t * forward

    r0, az0, el0 = _spherical(pos0 - pivot)
    r1 = r0 + float(orbit.get("radius", 0.0))
    az1 = az0 + float(orbit.get("azimuth_deg", 0.0))
    el1 = el0 + float(orbit.get("elevation_deg", 0.0))
    pos = pivot + _orbit_offset(r1, az1, el1)
    rot = _orbit_rotation(az0, el0, az1, el1) @ rot0

    body_id = int(sim.model.cam_bodyid[cam_id])
    body_pos = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
    body_rot = np.array(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    sim.model.cam_pos[cam_id] = body_rot.T @ (pos - body_pos)
    sim.model.cam_quat[cam_id] = T.convert_quat(T.mat2quat(body_rot.T @ rot), to="wxyz")
    sim.forward()
