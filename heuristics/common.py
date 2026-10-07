"""Helpers shared by the hard-coded S3 solvers in heuristics/<task>/solver_s3.py.

Solvers are served to the sandbox as one self-contained code block: the static-solver stub
(capx/serving/static_solver_server.py) replaces each solver's
``from heuristics.common import *`` line with this file's contents, so nothing here may rely
on being a real module (no relative imports, no ``from __future__``). Only the exact,
behaviour-neutral duplicates live here; task-specific perception stays in each solver.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def make_recorder(
    tag: str,
    trace: list,
    artifact_dir: "Path | None" = None,
    json_lines: bool = False,
) -> Callable[..., None]:
    """Return a ``record(event, **values)`` that appends to ``trace`` and logs one line.

    json_lines=True prints ``[tag:s3] {json}``; otherwise ``[tag:s3] event: k=v, ...`` and,
    when artifact_dir is set, rewrites artifact_dir/trajectory_trace.json after each event.
    """

    def record(event: str, **values: Any) -> None:
        entry = {"event": event, **{key: to_jsonable(value) for key, value in values.items()}}
        trace.append(entry)
        if json_lines:
            print(f"[{tag}:s3] {json.dumps(entry)}", flush=True)
            return
        compact = ", ".join(f"{key}={value}" for key, value in entry.items() if key != "event")
        print(f"[{tag}:s3] {event}" + (f": {compact}" if compact else ""))
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "trajectory_trace.json").write_text(
                json.dumps(trace, indent=2) + "\n", encoding="utf-8"
            )

    return record


def depth_2d(depth: np.ndarray) -> np.ndarray:
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


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def component_detections(
    mask: np.ndarray, points_base: np.ndarray, min_mask_pixels: int
) -> list:
    """Connected components of ``mask`` with their pixel box, XY centre and top-surface z."""
    labels, count = ndimage.label(mask)
    detections = []
    height, width = mask.shape
    for component_id in range(1, count + 1):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels < min_mask_pixels:
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


# Names the solvers used before these helpers were shared.
_jsonable = _plain = to_jsonable
_depth_2d = depth_2d
_safe_name = safe_name
