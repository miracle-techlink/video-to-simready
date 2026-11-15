"""Versioned benchmark input contracts shared by planning, rendering and scoring.

These pure validators freeze dataset / view-set / render-request /
evaluation-request manifests before any execution. File references are resolved
against a manifest root, hash-checked, and path escapes are rejected. A split is
a *claimed* split unless access evidence covering every fitted component
(including camera estimation and texture fitting) is provided; validators
reject manifests that misrepresent a claimed split as audited.
"""

import re
from pathlib import Path

from simready.core.artifacts import digest

SCHEMA_VERSION = 1

SPLITS = {"fit", "development", "sealed"}
FITTING_COMPONENTS = {"reconstruction", "camera_estimation", "texture_fitting"}
RENDER_MODES = {"rgb", "rgbd"}
CANDIDATE_SOURCE_MODES = {"external", "auto"}

_HASH = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """Raised when a benchmark input manifest violates its frozen contract."""


def _error(message):
    raise ContractError(message)


def _fields(value, allowed, required, label):
    if not isinstance(value, dict):
        _error(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        _error(f"Unknown {label} fields: {sorted(unknown)}")
    missing = set(required) - set(value)
    if missing:
        _error(f"Missing {label} fields: {sorted(missing)}")


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        _error(f"{label} must be a nonempty string")
    return value


def _bool(value, label):
    if type(value) is not bool:
        _error(f"{label} must be a boolean")
    return value


def _sha256(value, label="sha256"):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        _error(f"{label} must be a lowercase SHA256 digest")
    return value


def _finite_number(value, label, minimum=None, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value != value
        or value in (float("inf"), float("-inf"))
    ):
        _error(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        _error(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        _error(f"{label} must be <= {maximum}")
    return value


def _vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        _error(f"{label} must be a list of {length} finite numbers")
    for index, item in enumerate(value):
        _finite_number(item, f"{label}[{index}]")
    return list(value)


def _matrix(value, shape, label):
    if not isinstance(value, list) or len(value) != shape[0]:
        _error(f"{label} must be a {shape[0]}x{shape[1]} list of finite numbers")
    for row_index, row in enumerate(value):
        _vector(row, shape[1], f"{label}[{row_index}]")
    return [list(row) for row in value]


def _versioned(value, kind, fields_allowed, fields_required):
    _fields(value, fields_allowed | {"schema_version", "kind"},
            fields_required | {"schema_version", "kind"}, kind)
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        _error(f"{kind} requires schema_version {SCHEMA_VERSION}")
    if value["kind"] != kind:
        _error(f"{kind} kind mismatch: {value['kind']!r}")


def _file_ref(value, root, label, *, verify_hash=True):
    """Resolve a manifest-relative reference; escape and hash drift are fatal."""
    _fields(value, {"path", "sha256"}, {"path", "sha256"}, f"{label} file reference")
    _string(value["path"], f"{label} path")
    if Path(value["path"]).is_absolute():
        _error(f"{label} path must be relative to the manifest root: {value['path']!r}")
    _sha256(value["sha256"], f"{label} sha256")
    root = Path(root).resolve()
    resolved = (root / value["path"]).resolve()
    if not resolved.is_relative_to(root):
        _error(f"{label} path escapes the manifest root: {value['path']!r}")
    if not resolved.is_file():
        _error(f"Missing {label} file: {value['path']!r}")
    if verify_hash and digest(resolved) != value["sha256"]:
        _error(f"Hash mismatch for {label}: {value['path']!r}")
    return {"path": value["path"], "sha256": value["sha256"]}


def _string_list(value, label):
    if not isinstance(value, list) or not value:
        _error(f"{label} must be a nonempty list of strings")
    return [_string(item, f"{label}[]") for item in value]


def _check_split_access(split, label):
    """A claimed split only becomes audited with restricted inputs and evidence."""
    _fields(split, {"status", "access_evidence", "fitted_components"},
            {"status", "access_evidence", "fitted_components"}, label)
    if split["status"] not in {"claimed", "audited"}:
        _error(f"{label} status must be 'claimed' or 'audited'")
    _string_list(split["fitted_components"], f"{label} fitted_components")
    unknown = set(split["fitted_components"]) - FITTING_COMPONENTS
    if unknown:
        _error(f"{label} unknown fitted components: {sorted(unknown)}")
    evidence = split["access_evidence"]
    if evidence is not None:
        _fields(evidence, {"restricted_inputs", "access_log"},
                {"restricted_inputs", "access_log"}, f"{label} access_evidence")
        _string_list(evidence["restricted_inputs"], f"{label} restricted_inputs")
        _string(evidence["access_log"], f"{label} access_log")
    if split["status"] == "audited" and evidence is None:
        _error(
            f"{label} cannot be reported as an audited split without restricted "
            "inputs and actual access evidence; use 'claimed' instead"
        )


def validate_dataset(manifest, root):
    """Validate a frozen dataset manifest: identities, splits and hashed media."""
    _versioned(manifest, "dataset",
               {"dataset_id", "version", "source", "license", "scenes",
                "split_access", "capture_evidence"},
               {"dataset_id", "version", "source", "license", "scenes",
                "split_access"})
    _string(manifest["dataset_id"], "dataset_id")
    _string(manifest["version"], "dataset version")
    _fields(manifest["source"], {"type", "reference"}, {"type", "reference"},
            "dataset source")
    _string(manifest["source"]["type"], "dataset source type")
    _string(manifest["source"]["reference"], "dataset source reference")
    _string(manifest["license"], "dataset license")
    root = Path(root)
    capture_evidence = manifest.get("capture_evidence")
    if capture_evidence is not None:
        _file_ref(capture_evidence, root, "capture evidence")
    scene_ids, capture_ids, frame_ids = set(), set(), set()
    frame_split = {}
    scenes = manifest["scenes"]
    if not isinstance(scenes, list) or not scenes:
        _error("dataset scenes must be a nonempty list")
    for scene in scenes:
        _fields(scene, {"scene_id", "captures"}, {"scene_id", "captures"}, "scene")
        _string(scene["scene_id"], "scene_id")
        if scene["scene_id"] in scene_ids:
            _error(f"Duplicate scene id: {scene['scene_id']!r}")
        scene_ids.add(scene["scene_id"])
        if not isinstance(scene["captures"], list) or not scene["captures"]:
            _error("scene captures must be a nonempty list")
        for capture in scene["captures"]:
            _fields(capture, {"capture_id", "frames"},
                    {"capture_id", "frames"}, "capture")
            _string(capture["capture_id"], "capture_id")
            if capture["capture_id"] in capture_ids:
                _error(f"Duplicate capture id: {capture['capture_id']!r}")
            capture_ids.add(capture["capture_id"])
            if not isinstance(capture["frames"], list) or not capture["frames"]:
                _error("capture frames must be a nonempty list")
            for frame in capture["frames"]:
                _fields(frame,
                        {"frame_id", "split", "image", "depth",
                         "derived_from_capture"},
                        {"frame_id", "split", "image"}, "frame")
                _string(frame["frame_id"], "frame_id")
                if frame["frame_id"] in frame_ids:
                    _error(f"Duplicate frame id: {frame['frame_id']!r}")
                frame_ids.add(frame["frame_id"])
                if frame["split"] not in SPLITS:
                    _error(f"Unknown frame split: {frame['split']!r}")
                frame_split[frame["frame_id"]] = frame["split"]
                _file_ref(frame["image"], root, "frame image")
                if frame.get("depth") is not None:
                    _file_ref(frame["depth"], root, "frame depth")
                derived = frame.get("derived_from_capture")
                if derived is not None:
                    # A fit-only extraction is a derived product; it must name
                    # its original capture instead of replacing the video hash.
                    _string(derived, "frame derived_from_capture")
                    if derived not in capture_ids:
                        _error(
                            f"Frame {frame['frame_id']!r} derived from unknown "
                            f"capture {derived!r}"
                        )
    counts = {name: sum(1 for split in frame_split.values() if split == name)
              for name in SPLITS}
    if not counts["fit"]:
        _error("dataset needs a nonempty fit split")
    if not counts["development"] and not counts["sealed"]:
        _error("dataset needs a development or sealed evaluation split")
    access = manifest["split_access"]
    _fields(access, SPLITS, SPLITS, "split_access")
    for name in SPLITS:
        _check_split_access(access[name], f"split_access.{name}")
    return manifest


def validate_view_set(manifest, root):
    """Validate the shared target views every model must render."""
    _versioned(manifest, "view_set",
               {"view_set_id", "coordinate_frame", "unit", "views"},
               {"view_set_id", "coordinate_frame", "unit", "views"})
    _string(manifest["view_set_id"], "view_set_id")
    _string(manifest["coordinate_frame"], "coordinate_frame")
    if manifest["unit"] not in {"m", "cm", "mm", "arbitrary"}:
        _error(f"Unknown view set unit: {manifest['unit']!r}")
    root = Path(root)
    views = manifest["views"]
    if not isinstance(views, list) or not views:
        _error("view_set views must be a nonempty list")
    seen = set()
    for view in views:
        _fields(view,
                {"view_id", "resolution", "intrinsics", "camera_to_world",
                 "distortion", "preprocessing", "camera_evidence"},
                {"view_id", "resolution", "intrinsics", "camera_to_world",
                 "distortion", "preprocessing", "camera_evidence"},
                "view")
        _string(view["view_id"], "view_id")
        if view["view_id"] in seen:
            _error(f"Duplicate view id: {view['view_id']!r}")
        seen.add(view["view_id"])
        _fields(view["resolution"], {"width", "height"}, {"width", "height"},
                "view resolution")
        for axis in ("width", "height"):
            value = view["resolution"][axis]
            if type(value) is not int or value <= 0 or value > 16384:
                _error(f"view resolution {axis} must be an integer in 1..16384")
        _matrix(view["intrinsics"], (3, 3), "view intrinsics")
        _matrix(view["camera_to_world"], (4, 4), "view camera_to_world")
        _fields(view["distortion"], {"model", "coefficients"},
                {"model", "coefficients"}, "view distortion")
        if view["distortion"]["model"] not in {"none", "opencv"}:
            _error(f"Unknown distortion model: {view['distortion']['model']!r}")
        _vector(view["distortion"]["coefficients"],
                len(view["distortion"]["coefficients"]),
                "view distortion coefficients")
        if view["distortion"]["model"] == "none" and view["distortion"]["coefficients"]:
            _error("Distortion 'none' must not carry coefficients")
        if view["distortion"]["model"] == "opencv" and not (
            4 <= len(view["distortion"]["coefficients"]) <= 14
        ):
            _error("OpenCV distortion needs 4..14 coefficients")
        _string(view["preprocessing"], "view preprocessing")
        _file_ref(view["camera_evidence"], root, "view camera evidence")
    return manifest


def validate_render_request(manifest, root):
    """Freeze renderer identity, views, image settings, lighting, seed and budget."""
    _versioned(manifest, "render_request",
               {"request_id", "scene", "reconstruction", "view_set", "renderer",
                "image_settings", "lighting", "seed", "budget"},
               {"request_id", "scene", "reconstruction", "view_set", "renderer",
                "image_settings", "lighting", "seed", "budget"})
    _string(manifest["request_id"], "request_id")
    root = Path(root)
    _file_ref(manifest["scene"], root, "scene")
    _file_ref(manifest["reconstruction"], root, "reconstruction")
    _file_ref(manifest["view_set"], root, "view_set manifest")
    renderer = manifest["renderer"]
    _fields(renderer,
            {"name", "version", "code_sha256", "environment"},
            {"name", "version", "code_sha256", "environment"}, "renderer")
    _string(renderer["name"], "renderer name")
    _string(renderer["version"], "renderer version")
    _sha256(renderer["code_sha256"], "renderer code_sha256")
    _fields(renderer["environment"], {"image", "python"},
            {"image", "python"}, "renderer environment")
    _string(renderer["environment"]["image"], "renderer environment image")
    _string(renderer["environment"]["python"], "renderer environment python")
    settings = manifest["image_settings"]
    _fields(settings, {"width", "height", "mode", "background"},
            {"width", "height", "mode"}, "image_settings")
    for axis in ("width", "height"):
        value = settings[axis]
        if type(value) is not int or value <= 0 or value > 16384:
            _error(f"image_settings {axis} must be an integer in 1..16384")
    if settings["mode"] not in RENDER_MODES:
        _error(f"Unknown image_settings mode: {settings['mode']!r}")
    if settings.get("background") is not None:
        _vector(settings["background"], 3, "image_settings background")
    lighting = manifest["lighting"]
    _fields(lighting, {"policy", "exposure", "seed"},
            {"policy", "exposure"}, "lighting")
    if lighting["policy"] not in {"frozen", "scene_default"}:
        _error(f"Unknown lighting policy: {lighting['policy']!r}")
    _finite_number(lighting["exposure"], "lighting exposure")
    if lighting["policy"] == "frozen" and lighting.get("seed") is None:
        _error("Frozen lighting requires an explicit seed")
    if lighting.get("seed") is not None and (
        type(lighting["seed"]) is not int or lighting["seed"] < 0
    ):
        _error("lighting seed must be a nonnegative integer")
    if type(manifest["seed"]) is not int or manifest["seed"] < 0:
        _error("render_request seed must be a nonnegative integer")
    budget = manifest["budget"]
    _fields(budget, {"max_wall_seconds", "max_memory_bytes", "max_gpu_seconds"},
            {"max_wall_seconds"}, "budget")
    _finite_number(budget["max_wall_seconds"], "budget max_wall_seconds",
                   minimum=0)
    for name in ("max_memory_bytes", "max_gpu_seconds"):
        if budget.get(name) is not None:
            _finite_number(budget[name], f"budget {name}", minimum=0)
    return manifest


def validate_evaluation_request(manifest, root):
    """Freeze the scoring protocol; thresholds stay separate from render config."""
    _versioned(manifest, "evaluation_request",
               {"request_id", "protocol", "candidate_source", "scorer",
                "deadline", "render_request"},
               {"request_id", "protocol", "candidate_source", "scorer",
                "deadline", "render_request"})
    _string(manifest["request_id"], "request_id")
    root = Path(root)
    _file_ref(manifest["render_request"], root, "render_request manifest")
    protocol = manifest["protocol"]
    _fields(protocol, {"name", "version", "sha256"},
            {"name", "version", "sha256"}, "protocol")
    _string(protocol["name"], "protocol name")
    _string(protocol["version"], "protocol version")
    _sha256(protocol["sha256"], "protocol sha256")
    candidate = manifest["candidate_source"]
    _fields(candidate, {"mode", "reference"},
            {"mode"}, "candidate_source")
    if candidate["mode"] not in CANDIDATE_SOURCE_MODES:
        _error(f"Unknown candidate source mode: {candidate['mode']!r}")
    if candidate["mode"] == "external":
        if candidate.get("reference") is None:
            _error("External candidate source requires a frozen reference")
        _file_ref(candidate["reference"], root, "candidate reference")
    elif candidate.get("reference") is not None:
        _error("Auto candidate source must not bind a pre-existing reference")
    scorer = manifest["scorer"]
    _fields(scorer, {"name", "version", "code_sha256"},
            {"name", "version", "code_sha256"}, "scorer")
    _string(scorer["name"], "scorer name")
    _string(scorer["version"], "scorer version")
    _sha256(scorer["code_sha256"], "scorer code_sha256")
    _string(manifest["deadline"], "deadline")
    return manifest


_VALIDATORS = {
    "dataset": validate_dataset,
    "view_set": validate_view_set,
    "render_request": validate_render_request,
    "evaluation_request": validate_evaluation_request,
}


def validate_manifest(manifest, root):
    """Dispatch on the manifest kind; unknown kinds are rejected."""
    if not isinstance(manifest, dict) or manifest.get("kind") not in _VALIDATORS:
        _error(f"Unknown benchmark manifest kind: {manifest.get('kind')!r}"
               if isinstance(manifest, dict) else "Manifest must be an object")
    return _VALIDATORS[manifest["kind"]](manifest, root)
