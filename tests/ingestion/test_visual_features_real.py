"""Real open_clip encoding tests (opt-in, ``--run-real-model``).

Skipped unless BOTH the pinned ViT-B-16/datacomp_xl checkpoint and a real
fixture image are staged locally. No downloads and no network access happen
here. These tests verify encoder mechanics (dimensions, finiteness, text
normalization, cache replay honesty) — never material accuracy.

Unlock conditions:
  1. ``pip install 'simready-pipeline[visual-features]'`` (torch +
     open-clip-torch).
  2. Local open_clip checkpoint at
     ``~/.cache/simready_models/clip/ViT-B-16-datacomp_xl/open_clip_pytorch_model.bin``
     (LAION CLIP-ViT-B-16-DataComp.XL-s13B-b90K, MIT) or the path in
     ``SIMREADY_VISUAL_WEIGHTS``.
  3. A real image at ``tests/fixtures/data/grounded_real_image.jpg`` (shared
     with the grounded segmentation tests). Without a REAL image these tests
     stay skipped — synthetic pixels are not accepted here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import read_json
from simready.ingestion.visual_features import (
    DEFAULT_BACKBONE,
    DEFAULT_PRETRAINED,
    encode_regions,
)
from tests.paths import REPOSITORY_ROOT

pytestmark = pytest.mark.real_model

FIXTURE_IMAGE = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "data" / "grounded_real_image.jpg"
)
DEFAULT_WEIGHTS = (
    Path.home()
    / ".cache"
    / "simready_models"
    / "clip"
    / "ViT-B-16-datacomp_xl"
    / "open_clip_pytorch_model.bin"
)
PROMPTS = ["wood", "metal", "plastic", "glass", "fabric", "concrete"]


def _weights() -> Path:
    path = Path(os.environ.get("SIMREADY_VISUAL_WEIGHTS", DEFAULT_WEIGHTS))
    if not path.is_file():
        pytest.skip(f"open_clip checkpoint not staged: {path}")
    return path


@pytest.fixture
def real_request(tmp_path):
    weights = _weights()
    if not FIXTURE_IMAGE.is_file():
        pytest.skip(f"real fixture image missing: {FIXTURE_IMAGE}")
    with Image.open(FIXTURE_IMAGE) as image:
        width, height = image.convert("RGB").size
    regions = [
        {"region_id": "upper_half", "bbox_xyxy": [0, 0, width, height // 2]},
        {"region_id": "lower_half", "bbox_xyxy": [0, height // 2, width, height]},
    ]
    from simready.core.artifacts import digest

    request = {
        "schema_version": 1,
        "image": {"path": str(FIXTURE_IMAGE), "sha256": digest(FIXTURE_IMAGE)},
        "regions": regions,
        "model": {
            "backbone": DEFAULT_BACKBONE,
            "pretrained": DEFAULT_PRETRAINED,
            "weights": {"path": str(weights), "sha256": digest(weights)},
            "expected_feature_dim": 512,
        },
        "preprocess": {"mode": "open_clip_default", "input_size": 224},
        "texts": {"prompts": PROMPTS, "prefix": "a photo of ", "suffix": " material"},
        "seed": 0,
        "device": "cpu",
        "cpu_threads": 2,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path


def test_real_clip_encoding_dimensions_and_provenance(real_request, tmp_path):
    receipt = read_json(encode_regions(real_request, tmp_path / "fresh"))
    assert receipt["new_visual_inference_performed"] is True
    assert receipt["cache_hit"] is False
    assert receipt["model"]["backbone"] == DEFAULT_BACKBONE
    assert receipt["model"]["feature_dim"] == 512
    assert receipt["model"]["upstream_backbone_match"] is True
    with np.load(
        tmp_path / "fresh" / "region-features.npz", allow_pickle=False
    ) as data:
        assert data["image_features"].shape == (2, 512)
        assert np.isfinite(data["image_features"]).all()
        assert data["text_features"].shape == (len(PROMPTS), 512)
        assert np.allclose(
            np.linalg.norm(data["text_features"], axis=1), 1.0, atol=1e-3
        )
        # Two genuinely different regions must not collapse to one feature.
        assert not np.allclose(data["image_features"][0], data["image_features"][1])


def test_real_clip_cache_replay_is_honest(real_request, tmp_path):
    cache = tmp_path / "cache"
    first = read_json(encode_regions(real_request, tmp_path / "a", cache_dir=cache))
    assert first["new_visual_inference_performed"] is True
    second = read_json(encode_regions(real_request, tmp_path / "b", cache_dir=cache))
    assert second["new_visual_inference_performed"] is False
    assert second["cache_hit"] is True
    assert (tmp_path / "a" / "region-features.npz").read_bytes() == (
        tmp_path / "b" / "region-features.npz"
    ).read_bytes()
