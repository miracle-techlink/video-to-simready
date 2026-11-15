"""Build PR1-contract inputs for the two-view GPU regression run.

Run with the regression venv python from the inputs directory:
    cd <run>/inputs && python make_inputs.py

Generates: camera evidence files, view_set.json, render_request.json with
real sha256 digests computed after every referenced file is on disk.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

INPUTS = Path(__file__).resolve().parent
REPO = Path("/root/gpufree-data/liuyue/video-to-simready-gpu-regression")
OBSERVATION_CODE = REPO / "src/simready/simulation/benchmark_observation.py"

WIDTH, HEIGHT = 640, 480
FX = 554.0
CUP_CENTER = np.array([0.05, 0.05, 0.05])  # from scene-spec.json bounds


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opencv_c2w(eye, target, world_up=(0.0, 0.0, 1.0)) -> list:
    """OpenCV camera_to_world: columns x=right, y=down, z=forward, t=eye."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z = z / np.linalg.norm(z)
    up = np.asarray(world_up, dtype=float)
    y = up - np.dot(up, z) * z
    y = -y / np.linalg.norm(y)  # camera y points down
    x = np.cross(y, z)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    # rigidity self-check before freezing
    r = c2w[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12), r @ r.T
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12), np.linalg.det(r)
    return c2w.tolist()


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    # USD prim identifiers cannot contain '-': Sdf.Path(".../view-00")
    # silently yields emptyPath, so the worker's camera Define fails.
    views_spec = [
        ("view_00", (0.05, -0.29, 0.24)),
        ("view_01", (0.30, 0.16, 0.30)),
    ]

    for view_id, eye in views_spec:
        evidence = {
            "kind": "camera_evidence",
            "view_id": view_id,
            "origin": "synthetic-two-view-regression",
            "note": (
                "Pose authored for the 2026-11-15 GPU regression acceptance; "
                "not derived from real camera estimation."
            ),
            "eye_m": list(eye),
            "target_m": CUP_CENTER.tolist(),
        }
        write_json(INPUTS / f"camera_evidence_{view_id}.json", evidence)

    views = []
    for view_id, eye in views_spec:
        views.append(
            {
                "view_id": view_id,
                "resolution": {"width": WIDTH, "height": HEIGHT},
                "intrinsics": [[FX, 0.0, WIDTH / 2.0], [0.0, FX, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
                "camera_to_world": opencv_c2w(eye, CUP_CENTER),
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {
                    "path": f"camera_evidence_{view_id}.json",
                    "sha256": sha256(INPUTS / f"camera_evidence_{view_id}.json"),
                },
            }
        )

    view_set = {
        "kind": "view_set",
        "schema_version": 1,
        "view_set_id": "gpu-regression-twoview-20261115",
        "coordinate_frame": "scene.usda world frame (metersPerUnit=1, upAxis=Z)",
        "unit": "m",
        "views": views,
    }
    write_json(INPUTS / "view_set.json", view_set)

    render_request = {
        "kind": "render_request",
        "schema_version": 1,
        "request_id": "gpu-regression-twoview-20261115-001",
        "scene": {"path": "scene.usda", "sha256": sha256(INPUTS / "scene.usda")},
        "reconstruction": {"path": "gaussian.ply", "sha256": sha256(INPUTS / "gaussian.ply")},
        "view_set": {"path": "view_set.json", "sha256": sha256(INPUTS / "view_set.json")},
        "renderer": {
            "name": "isaac-sim-native",
            "version": "6.2.0-rc.14",
            "code_sha256": sha256(OBSERVATION_CODE),
            "environment": {"image": "native-standalone", "python": "3.12.3"},
        },
        "image_settings": {"width": WIDTH, "height": HEIGHT, "mode": "rgbd"},
        "lighting": {"policy": "frozen", "exposure": 0.0, "seed": 7},
        "seed": 42,
        "budget": {"max_wall_seconds": 900.0},
    }
    write_json(INPUTS / "render_request.json", render_request)
    print("inputs written to", INPUTS)
    for name in ("scene.usda", "gaussian.ply", "view_set.json", "render_request.json"):
        print(f"  {name}: {sha256(INPUTS / name)}")


if __name__ == "__main__":
    main()
