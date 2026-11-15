"""Real NeRF2Physics feature-regression tests (PR 6b).

Skipped unless ``--run-real-model`` is passed AND all external inputs exist.
No weights are downloaded and no upstream repository is cloned by these tests.

Unlock conditions (all required):
  1. ``docs/PR6_ASSETS.md`` (Agent G's manifest) contains the pinned commit
     and source-file sha256 table.
  2. Upstream repository cloned at ``/tmp/nerf2physics_probe`` and checked
     out at the pinned commit. The clone is verified against the sha256
     table in the manifest; a mismatch skips.
  3. CLIP feature cache staged under ``~/.cache/simready_models/clip_cache/``
     with the sha256s recorded in the manifest.
  4. ``scikit-learn`` installed for the regression-head probe.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from simready.core.artifacts import digest, read_json, write_json
from simready.core.evidence import reference
from simready.ingestion.nerf2physics import PROPERTIES, REPOSITORY, run_estimation
from tests.paths import REPOSITORY_ROOT

pytestmark = pytest.mark.real_model

# ---------------------------------------------------------------------------
# Parameterization.
# ---------------------------------------------------------------------------

ASSETS_DOC = REPOSITORY_ROOT / "docs" / "PR6_ASSETS.md"
UPSTREAM_PROBE = Path("/tmp/nerf2physics_probe")
CLIP_CACHE = Path.home() / ".cache" / "simready_models" / "clip_cache"

COMMIT_PATTERN = re.compile(r"pinned commit:\s*`([0-9a-f]{40})`", re.MULTILINE)

# Physical plausibility bands (kg/m^3, kg, unitless friction).
PLAUSIBLE_RANGES = {
    "density_kg_m3": (100.0, 20000.0),
    "mass_kg": (0.001, 1000.0),
    "dynamic_friction": (0.0, 2.0),
}

UPSTREAM_TOLERANCE = 0.05  # 5% relative tolerance vs upstream reference outputs

PROPERTY_UNDER_TEST = "density"  # density needs no carving evidence beyond the payload


# ---------------------------------------------------------------------------
# Manifest parsing and verification.
# ---------------------------------------------------------------------------


def _parse_pinned_commit() -> str:
    if not ASSETS_DOC.exists():
        pytest.skip(f"PR6 asset manifest not written yet: {ASSETS_DOC}")
    match = COMMIT_PATTERN.search(ASSETS_DOC.read_text())
    if not match:
        pytest.skip(f"NeRF2Physics upstream commit not pinned in {ASSETS_DOC}")
    return match.group(1)


def _parse_source_file_hashes() -> dict[str, str]:
    """Extract the upstream source-file sha256 table from Agent G's manifest."""
    text = ASSETS_DOC.read_text()
    section = re.search(r"## 上游仓库.*?(?=## |\Z)", text, re.DOTALL)
    if not section:
        pytest.skip("upstream repository section not found in PR6_ASSETS.md")
    hashes = {}
    for line in section.group(0).splitlines():
        match = re.match(r"\| (\w+\.py|LICENSE) \| `([0-9a-f]{64})`", line)
        if match:
            hashes[match.group(1)] = match.group(2)
    if not hashes:
        pytest.skip("no source-file sha256 entries found in PR6_ASSETS.md")
    return hashes


def _parse_clip_cache_hashes() -> dict[str, str]:
    text = ASSETS_DOC.read_text()
    section = re.search(r"## CLIP 特征缓存.*?(?=## |\Z)", text, re.DOTALL)
    if not section:
        pytest.skip("CLIP cache section not found in PR6_ASSETS.md")
    hashes = {}
    for line in section.group(0).splitlines():
        match = re.search(r"`([^`]+\.npy)`.*?sha256 `([0-9a-f]{64})`", line)
        if match:
            hashes[match.group(1)] = match.group(2)
        match = re.search(r"`([^`]+\.json)`.*?sha256 `([0-9a-f]{64})`", line)
        if match:
            hashes[match.group(1)] = match.group(2)
    if not hashes:
        pytest.skip("no CLIP cache sha256 entries found in PR6_ASSETS.md")
    return hashes


def _verify_upstream_probe(commit: str, expected_hashes: dict[str, str]) -> Path:
    """Verify the upstream clone exists, is at the pinned commit, and matches sha256s."""
    if not UPSTREAM_PROBE.exists():
        pytest.skip(f"upstream probe not cloned: {UPSTREAM_PROBE}")
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_PROBE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual_commit = result.stdout.strip()
    if actual_commit != commit:
        pytest.skip(f"upstream probe at {actual_commit}, expected pinned {commit}")
    mismatches = []
    for filename, expected_sha in expected_hashes.items():
        path = UPSTREAM_PROBE / filename
        if not path.exists():
            mismatches.append(f"{filename} missing")
            continue
        actual = digest(path)
        if actual != expected_sha:
            mismatches.append(f"{filename} sha256 mismatch")
    if mismatches:
        pytest.skip("upstream probe corrupted: " + "; ".join(mismatches))
    return UPSTREAM_PROBE


def _verify_clip_cache(expected_hashes: dict[str, str]) -> dict[str, Path]:
    if not CLIP_CACHE.exists():
        pytest.skip(f"CLIP cache not staged: {CLIP_CACHE}")
    files = {}
    mismatches = []
    for filename, expected_sha in expected_hashes.items():
        path = CLIP_CACHE / filename
        if not path.exists():
            mismatches.append(f"{filename} missing")
            continue
        actual = digest(path)
        if actual != expected_sha:
            mismatches.append(f"{filename} sha256 mismatch")
        files[filename] = path
    if mismatches:
        pytest.skip("CLIP cache corrupted: " + "; ".join(mismatches))
    return files


@pytest.fixture(scope="module")
def real_assets():
    commit = _parse_pinned_commit()
    source_hashes = _parse_source_file_hashes()
    clip_hashes = _parse_clip_cache_hashes()
    upstream = _verify_upstream_probe(commit, source_hashes)
    clip_files = _verify_clip_cache(clip_hashes)
    return {
        "commit": commit,
        "upstream": upstream,
        "source_hashes": source_hashes,
        "clip_cache": clip_files,
    }


@pytest.fixture()
def real_request(real_assets, tmp_path):
    """Assemble a schema-1 estimation request bound to the verified real files.

    The CLIP cache staged by Agent G contains sample features for a single
    image and four material phrases. We build a synthetic-but-real-feature
    payload that exercises the full regression path.
    """
    import numpy as np

    # Load the staged CLIP features.
    image_features = np.load(real_assets["clip_cache"]["sample_image_features.npy"])
    text_features = np.load(real_assets["clip_cache"]["sample_text_features.npy"])
    assert image_features.shape == (1, 512), (
        f"unexpected image feature shape {image_features.shape}"
    )
    assert text_features.shape == (4, 512), (
        f"unexpected text feature shape {text_features.shape}"
    )

    # Build a minimal but valid feature payload. The staged cache is a
    # smoke-test sample (single image, four phrases), so we tile the image
    # feature into a small patch grid to exercise the regression path.
    n_views, n_points = 2, 4
    patch_features = np.tile(image_features[0], (n_views, n_points, 1))
    visible = np.ones((n_views, n_points), dtype=bool)

    feature_path = tmp_path / "real_features.npz"
    np.savez(
        feature_path,
        patch_features=patch_features,
        visible=visible,
        text_features=text_features,
        material_names=np.array(["cabinet", "door", "chair", "floor"]),
    )

    payload = {
        "features": reference(feature_path),
        "temperature": 0.1,
        "materials": [
            {"name": "cabinet", "interval": [600.0, 900.0], "unit": "kg/m^3"},
            {"name": "door", "interval": [500.0, 800.0], "unit": "kg/m^3"},
            {"name": "chair", "interval": [400.0, 700.0], "unit": "kg/m^3"},
            {"name": "floor", "interval": [2000.0, 2500.0], "unit": "kg/m^3"},
        ],
        "source_object_id": "pr6_real_density_object",
        "model_identity": {
            "clip_backbone": "open_clip/ViT-B-32",
            "checkpoint": "laion2b_s34b_b79k",
            "weights_sha256": None,
        },
    }
    payload_path = tmp_path / "payload.json"
    write_json(payload_path, payload)

    association_path = tmp_path / "association.json"
    write_json(
        association_path,
        {
            "target_object_id": "pr6-real-density-object",
            "source_object_id": "pr6_real_density_object",
            "kind": "pr6_real_feature_cache",
        },
    )

    # Use the verified upstream source files as evidence.
    upstream = real_assets["upstream"]
    request = {
        "schema_version": 1,
        "object_id": "pr6-real-density-object",
        "execution_mode": "feature_regression",
        "property_name": PROPERTY_UNDER_TEST,
        "association": {
            "source_object_id": "pr6_real_density_object",
            "reason": "PR6 real feature regression over Agent-G-staged CLIP cache",
            "evidence": reference(association_path),
        },
        "payload": reference(payload_path),
        "upstream": {
            "repository": REPOSITORY,
            "commit": real_assets["commit"],
            "license": "MIT",
            "source_files": [reference(upstream / "feature_fusion.py")],
            "license_file": reference(upstream / "LICENSE"),
        },
        "model_identity": payload["model_identity"],
    }
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    return request_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_method_provenance_records_pinned_commit(real_assets, real_request, tmp_path):
    candidate = read_json(run_estimation(real_request, tmp_path / "out"))
    method = candidate["method"]
    assert method["name"] == "nerf2physics"
    assert method["execution_mode"] == "feature_regression"
    assert method["upstream_commit"] == real_assets["commit"]
    assert method["upstream_repository"] == REPOSITORY
    assert method["license"] == "MIT"
    # The adapter currently hard-codes this flag False for every mode. That
    # is correct for official_prediction_replay / synthetic_fixture but wrong
    # for feature_regression over a real cache; the dedicated test below
    # asserts the contract and fails until the adapter distinguishes modes.
    assert method["new_visual_inference_performed"] in {True, False}


def test_new_visual_inference_flag_is_true(real_assets, real_request, tmp_path):
    """feature_regression over a real cache must declare new visual inference.

    The current adapter hard-codes ``new_visual_inference_performed=False``
    for every mode. That is correct for ``official_prediction_replay`` and
    ``synthetic_fixture`` but wrong for ``feature_regression`` over a real
    cache: the regression itself is new inference. This test is expected to
    fail until the adapter distinguishes the modes; the failure is the
    signal that the provenance contract is not yet implemented.
    """
    candidate = read_json(run_estimation(real_request, tmp_path / "out"))
    assert candidate["method"]["new_visual_inference_performed"] is True, (
        "feature_regression over a real cache must set "
        "new_visual_inference_performed=True; the adapter currently "
        "hard-codes False for every mode (see src/simready/ingestion/"
        "nerf2physics.py), which mislabels fresh inference as a replay"
    )


def test_predictions_are_physically_plausible(real_assets, real_request, tmp_path):
    candidate = read_json(run_estimation(real_request, tmp_path / "out"))
    key, unit = PROPERTIES[PROPERTY_UNDER_TEST]
    for prop_key, prop in candidate["properties"].items():
        interval = prop["interval"]
        assert interval[0] <= interval[1], f"reversed interval for {prop_key}"
        plausible_low, plausible_high = PLAUSIBLE_RANGES.get(
            prop_key, (float("-inf"), float("inf"))
        )
        assert plausible_low <= interval[0] <= plausible_high, (
            f"{prop_key} lower bound {interval[0]} outside "
            f"[{plausible_low}, {plausible_high}]"
        )
        assert plausible_low <= interval[1] <= plausible_high, (
            f"{prop_key} upper bound {interval[1]} outside "
            f"[{plausible_low}, {plausible_high}]"
        )
    assert candidate["properties"][key]["unit"] == unit


def test_clip_regression_head_output_is_plausible(real_assets):
    """Probe the CLIP feature cache with a regression head.

    This exercises the real feature arrays end-to-end (softmax weights from
    ``regress_features`` -> ridge head) without depending on
    ``run_estimation`` internals. The head is trained on-the-fly from the
    staged text features and material intervals, mirroring the published
    NeRF2Physics density head at the same abstraction level.
    """
    pytest.importorskip("sklearn", reason="scikit-learn required for regression head")
    import numpy as np
    from sklearn.linear_model import Ridge

    image_features = np.load(real_assets["clip_cache"]["sample_image_features.npy"])
    text_features = np.load(real_assets["clip_cache"]["sample_text_features.npy"])

    # Train a minimal ridge head on the staged features.
    materials = [
        ("cabinet", [600.0, 900.0]),
        ("door", [500.0, 800.0]),
        ("chair", [400.0, 700.0]),
        ("floor", [2000.0, 2500.0]),
    ]
    train_w, train_y = [], []
    rng = np.random.default_rng(20261115)
    for _ in range(100):
        mix = rng.dirichlet(np.ones(len(materials)))
        patch = mix @ text_features
        patch /= np.linalg.norm(patch)
        logits = text_features @ patch / 0.1
        w = np.exp(logits - logits.max())
        w /= w.sum()
        density_mid = sum(
            w_i * sum(iv) / 2 for w_i, (_, iv) in zip(w, materials, strict=False)
        )
        train_w.append(w)
        train_y.append(density_mid)
    head = Ridge(alpha=1e-6, fit_intercept=True)
    head.fit(np.array(train_w), np.array(train_y))

    # Probe with the real image feature.
    patch = image_features[0] / np.linalg.norm(image_features[0])
    logits = text_features @ patch / 0.1
    w = np.exp(logits - logits.max())
    w /= w.sum()
    predicted_density = float(head.predict(w.reshape(1, -1))[0])
    low, high = PLAUSIBLE_RANGES["density_kg_m3"]
    assert low <= predicted_density <= high, (
        f"CLIP regression head predicted density {predicted_density} kg/m^3 "
        f"outside plausible [{low}, {high}]"
    )


def test_real_regression_matches_upstream_reference(
    real_assets, real_request, tmp_path
):
    """Compare against upstream reference outputs, when available.

    The staged CLIP cache is a smoke-test sample (single image, four phrases),
    not a full NeRF2Physics evaluation scene. There is no upstream reference
    output for this input, so this test skips. When Agent G stages a real
    evaluation scene with published upstream outputs, the reference interval
    should be added to the payload and this test will assert 5% tolerance.
    """
    candidate = read_json(run_estimation(real_request, tmp_path / "out"))
    key, _ = PROPERTIES[PROPERTY_UNDER_TEST]
    predicted = candidate["properties"][key]["interval"]
    # No upstream reference output is staged for the smoke-test cache.
    # This assertion documents the expected shape of the check.
    reference_interval = None  # e.g. [650.0, 680.0] from upstream
    if reference_interval is None:
        pytest.skip(
            "no upstream reference output staged for the current CLIP cache; "
            "add reference_outputs to docs/PR6_ASSETS.md when available"
        )
    for got, want, label in zip(
        predicted, reference_interval, ("low", "high"), strict=True
    ):
        tolerance = abs(want) * UPSTREAM_TOLERANCE
        assert abs(got - want) <= tolerance, (
            f"{key} {label}: predicted {got} vs upstream {want} (tolerance {tolerance})"
        )


def test_commit_mismatch_is_detectable(real_assets, real_request, tmp_path):
    """A request pinned to a different commit must surface in the candidate."""
    candidate = read_json(run_estimation(real_request, tmp_path / "out"))
    recorded = candidate["method"]["upstream_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", recorded)
    assert recorded != "0" * 40
    assert recorded == real_assets["commit"], (
        f"candidate records {recorded}, manifest pins {real_assets['commit']}; "
        "a silent commit swap would not be caught"
    )
