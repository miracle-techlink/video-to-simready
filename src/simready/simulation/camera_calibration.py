"""Exact pinhole metadata for authored render cameras, not real-camera fitting."""

import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

from simready.core.artifacts import digest, object_digest
from simready.geometry.cameras import look_at_c2w
from simready.geometry.transforms_validation import validate_sim3


def rendered_camera_calibration(
    image: str | Path,
    *,
    ply_sha256: str | None = None,
    source_to_world: list[list[float]] | None = None,
    eye: list[float],
    target: list[float],
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    evidence: str,
) -> dict[str, Any]:
    """Bind an undistorted, Z-up render camera to its exact image and point file.

    Assumes zero lens shift and vertical aperture = horizontal aperture * H/W,
    matching scene_video. Pixel centers use integer coordinates (principal point
    (W-1)/2, (H-1)/2). External images require separately established calibration.
    When ``ply_sha256`` and ``source_to_world`` are omitted, the calibration is
    shared across benchmark members; neither a PLY hash nor a transform hash is
    recorded, and ``calibration_kind`` becomes ``shared_render_camera``.
    """
    if ply_sha256 is None and source_to_world is None:
        pass  # shared benchmark camera, not bound to any member
    elif ply_sha256 is None or source_to_world is None:
        raise ValueError(
            "ply_sha256 and source_to_world must be provided together or both omitted"
        )
    if ply_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", ply_sha256):
        raise ValueError("A source PLY SHA256 is required")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("A calibration evidence description is required")
    for value in (focal_length_mm, horizontal_aperture_mm):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Focal length and aperture must be finite and positive")
    with Image.open(image) as raster:
        width, height = raster.size
    focal_pixels = width * focal_length_mm / horizontal_aperture_mm
    result = {
        "schema_version": 1,
        "camera_convention": "opencv_c2w",
        "distortion": "none",
        "image_size": [width, height],
        "intrinsics": [
            [focal_pixels, 0, (width - 1) / 2],
            [0, focal_pixels, (height - 1) / 2],
            [0, 0, 1],
        ],
        "camera_to_world": look_at_c2w(eye, target).tolist(),
        "image_sha256": digest(image),
        "evidence": evidence,
        "real_camera_calibrated": False,
    }
    if ply_sha256 is not None:
        transform = validate_sim3(source_to_world)
        result.update(
            ply_sha256=ply_sha256,
            source_to_world_sha256=object_digest(transform.tolist()),
            calibration_kind="authored_render_camera",
        )
    else:
        result["calibration_kind"] = "shared_render_camera"
    return result
