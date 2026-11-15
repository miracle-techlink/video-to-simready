"""Real Grounding DINO + SAM inference tests (PR 6a).

Skipped unless ``--run-real-model`` is passed AND the pinned weights are
staged locally. No downloads, no network access, no package installation.

Unlock conditions (all required to leave skip state):
  1. ``pip install 'simready-pipeline[segmentation]'`` (torch + transformers
     >= 4.51). Availability is probed at runtime, not import time.
  2. Weights staged under ``~/.cache/simready_models/`` (see
     ``docs/PR6_ASSETS.md`` for the download recipe):
       - ``grounding-dino-tiny/model.safetensors``  sha256 pinned
       - ``sam-vit-base/model.safetensors``         sha256 pinned
     The tests copy these into a temporary HF cache layout so that
     ``grounded_segment_image(local_files_only=True)`` can load them
     without network access.
  3. Optional: ``tests/fixtures/data/grounded_real_image.jpg`` (a real photo
     containing at least one of the queried categories). If absent, a
     deterministic generated RGB stand-in is used, and detection assertions
     that require a real object skip independently.
  4. Optional: ``tests/fixtures/data/grounded_real_reference_mask.png``
     (binary 0/255 reference mask for the highest-scoring instance) to
     activate the IoU >= ``REFERENCE_MASK_IOU_THRESHOLD`` assertion.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import digest
from simready.ingestion.segmentation.grounded import (
    DETECTOR_ID,
    DETECTOR_REVISION,
    MASK_MODEL_ID,
    MASK_MODEL_REVISION,
    grounded_segment_image,
)

pytestmark = pytest.mark.real_model

# ---------------------------------------------------------------------------
# Parameterization (all thresholds and paths are explicit constants).
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "data"
REAL_IMAGE_PATH = FIXTURE_DIR / "grounded_real_image.jpg"
REFERENCE_MASK_PATH = FIXTURE_DIR / "grounded_real_reference_mask.png"

REFERENCE_MASK_IOU_THRESHOLD = float(
    os.environ.get("SIMREADY_GROUNDED_IOU_THRESHOLD", "0.5")
)
MIN_EXPECTED_INSTANCES = 1
MASK_VALUES_BINARY = {0, 255}
GENERATED_IMAGE_SIZE = (640, 480)  # width, height for the fallback fixture

EXPECTED_MODEL_ID = "grounding_dino_tiny+sam_vit_base"
EXPECTED_INSTANCE_METHOD = "text_grounded_box_prompted_sam"

# Agent G's staged weight directories (docs/PR6_ASSETS.md).
SIMREADY_CACHE = Path.home() / ".cache" / "simready_models"
WEIGHT_SOURCES = {
    DETECTOR_ID: (
        SIMREADY_CACHE / "grounding-dino-tiny",
        DETECTOR_REVISION,
        {
            "config.json": "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a",
            "model.safetensors": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
            "preprocessor_config.json": "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",
            "processor_config.json": "f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59",
            "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
            "tokenizer_config.json": "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",
            "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        },
    ),
    MASK_MODEL_ID: (
        SIMREADY_CACHE / "sam-vit-base",
        MASK_MODEL_REVISION,
        {
            "config.json": "5ebd0d8643b486f3a716bf17c2a15531eb818b2b96ff0c6c5dcc88fa015161af",
            "model.safetensors": "892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579",
            "preprocessor_config.json": "225545a743c654e3c495ec6f545a0eaba57c8ba3fbbd8483b3cb1c0fc58db517",
            "processor_config.json": "f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59",
        },
    ),
}

# Kept small for CPU runtime; phrases must survive the _query_mapping validator.
TEST_QUERIES = {"table": "table", "chair": "chair"}


# ---------------------------------------------------------------------------
# Skip helpers — every skip reason is explicit and greppable.
# ---------------------------------------------------------------------------


def _transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )
        from transformers.utils.hub import cached_file  # noqa: F401
    except ImportError:
        return False
    return True


def _require_segmentation_stack() -> None:
    if not _transformers_available():
        pytest.skip(
            "transformers/torch not installed; install simready-pipeline[segmentation]"
        )


def _verify_staged_weights() -> dict[str, Path]:
    """Verify Agent G's staged weights exist and match pinned sha256s."""
    missing = []
    for model_id, (directory, revision, expected_hashes) in WEIGHT_SOURCES.items():
        if not directory.exists():
            missing.append(f"{model_id}: directory {directory} not found")
            continue
        for filename, expected_sha in expected_hashes.items():
            path = directory / filename
            if not path.exists():
                missing.append(f"{model_id}: {filename} missing")
                continue
            actual = digest(path)
            if actual != expected_hashes[filename]:
                missing.append(
                    f"{model_id}: {filename} sha256 mismatch "
                    f"(expected {expected_hashes[filename]}, got {actual})"
                )
    if missing:
        pytest.skip("staged weights incomplete or corrupted: " + "; ".join(missing))
    return {model_id: directory for model_id, (directory, _, _) in WEIGHT_SOURCES.items()}


@pytest.fixture(scope="module")
def hf_cache(tmp_path_factory):
    """Build a temporary HF cache from the staged weights.

    ``grounded_segment_image`` uses ``transformers`` hub functions with
    ``local_files_only=True``; it expects the standard HF cache layout.
    This fixture copies the verified staged files into that layout.
    """
    _require_segmentation_stack()
    weight_dirs = _verify_staged_weights()
    cache_root = tmp_path_factory.mktemp("hf_cache")
    for model_id, (directory, revision, expected_hashes) in WEIGHT_SOURCES.items():
        # HF cache layout: models--<org>--<name>/snapshots/<revision>/
        safe_name = f"models--{model_id.replace('/', '--')}"
        snapshot_dir = cache_root / safe_name / "snapshots" / revision
        snapshot_dir.mkdir(parents=True)
        for filename in expected_hashes:
            shutil.copy2(directory / filename, snapshot_dir / filename)
    return cache_root


@pytest.fixture(scope="module")
def test_image(tmp_path_factory) -> Path:
    """Real fixture image if present, else a deterministic generated RGB stand-in."""
    if REAL_IMAGE_PATH.exists():
        return REAL_IMAGE_PATH
    rng = np.random.default_rng(20261115)
    width, height = GENERATED_IMAGE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (xx / width * 180 + 40),
            (yy / height * 160 + 30),
            ((xx + yy) / (width + height) * 200 + 20),
        ],
        axis=-1,
    )
    noise = rng.integers(-12, 13, size=base.shape)
    array = np.clip(base + noise, 0, 255).astype(np.uint8)
    path = tmp_path_factory.mktemp("grounded_real") / "generated_input.png"
    Image.fromarray(array, "RGB").save(path)
    return path


@pytest.fixture(scope="module")
def manifest(hf_cache, test_image, tmp_path_factory):
    """Single shared inference run per module (model loading is expensive)."""
    # Point transformers at our temporary cache.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(hf_cache))
    output_dir = tmp_path_factory.mktemp("grounded_out") / "run"
    result = grounded_segment_image(
        test_image,
        output_dir,
        queries=TEST_QUERIES,
        device="cpu",
        box_threshold=0.15,  # below default so a weak fixture can still detect
        text_threshold=0.15,
        min_area_px=64,
        local_files_only=True,
    )
    monkeypatch.undo()
    return result, output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_manifest_identity_and_provenance(manifest):
    data, _ = manifest
    assert data["model"]["model_id"] == EXPECTED_MODEL_ID
    assert data["provenance"]["instance_method"] == EXPECTED_INSTANCE_METHOD
    assert data["provenance"]["source_type"] == "model_inference"
    component_ids = {c["model_id"] for c in data["model"]["components"]}
    assert component_ids == {DETECTOR_ID, MASK_MODEL_ID}
    for component in data["model"]["components"]:
        expected_revision = WEIGHT_SOURCES[component["model_id"]][1]
        assert component["revision"] == expected_revision, (
            f"checkpoint drift: {component['model_id']} at "
            f"{component['revision']}, expected {expected_revision}"
        )
        checkpoint_sha256 = component["checkpoint_sha256"]
        assert isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64, (
            f"checkpoint_sha256 malformed for {component['model_id']}"
        )


def test_at_least_one_instance_detected(manifest, test_image):
    data, _ = manifest
    if not data["instances"] and test_image != REAL_IMAGE_PATH:
        pytest.skip(
            "generated fallback fixture produced no detections; provide "
            f"{REAL_IMAGE_PATH} for a detection guarantee"
        )
    assert len(data["instances"]) >= MIN_EXPECTED_INSTANCES, (
        "real model detected zero instances on a real fixture image"
    )


def test_instance_schema_and_score_fields(manifest):
    data, _ = manifest
    for instance in data["instances"]:
        for key in (
            "asset_id",
            "category",
            "score",
            "sam_predicted_iou",
            "mask",
            "bbox_xyxy",
            "area_px",
        ):
            assert key in instance, f"instance missing key {key!r}"
        assert 0.0 <= instance["score"] <= 1.0
        assert 0.0 <= instance["sam_predicted_iou"] <= 1.0
        assert instance["instance_method"] == EXPECTED_INSTANCE_METHOD


def test_masks_are_binary_grayscale_and_match_image_size(manifest, test_image):
    data, output_dir = manifest
    source = Image.open(test_image)
    for instance in data["instances"]:
        mask_ref = instance["mask"]
        mask_path = output_dir / mask_ref["path"]
        assert mask_path.exists(), f"mask artifact missing: {mask_path}"
        mask = np.asarray(Image.open(mask_path))
        assert mask.ndim == 2, "mask must be single-channel grayscale"
        assert mask.shape == (source.height, source.width), (
            f"mask shape {mask.shape} != image size "
            f"({source.height}, {source.width})"
        )
        unique = set(np.unique(mask).tolist())
        assert unique <= MASK_VALUES_BINARY, (
            f"mask is not binary 0/255; observed values {sorted(unique)}"
        )
        assert 255 in unique or instance["area_px"] == 0, (
            "instance with nonzero area has an all-zero mask"
        )


def test_reference_mask_iou(manifest):
    """IoU against a reference mask, when one is provided for the top instance."""
    data, output_dir = manifest
    if not REFERENCE_MASK_PATH.exists():
        pytest.skip(f"reference mask not present: {REFERENCE_MASK_PATH}")
    if not data["instances"]:
        pytest.skip("no instances detected; cannot compare to reference mask")
    top = max(data["instances"], key=lambda row: row["score"])
    predicted = np.asarray(Image.open(output_dir / top["mask"]["path"])) > 0
    reference = np.asarray(Image.open(REFERENCE_MASK_PATH)) > 0
    assert predicted.shape == reference.shape, (
        "reference mask dimensions must match the input image"
    )
    intersection = np.logical_and(predicted, reference).sum()
    union = np.logical_or(predicted, reference).sum()
    iou = intersection / union if union else 0.0
    assert iou >= REFERENCE_MASK_IOU_THRESHOLD, (
        f"IoU {iou:.3f} below threshold {REFERENCE_MASK_IOU_THRESHOLD} "
        f"for instance {top['asset_id']}"
    )


def test_local_files_only_means_no_network_fallback(manifest):
    """Inference under local_files_only=True must record zero LLM/API usage."""
    data, _ = manifest
    assert data["usage"]["llm_calls"] == 0
    assert data["timing"]["model_load_s"] > 0
