"""Real-image 2D-region CLIP encoding contract tests (no torch required).

These tests validate the production encoding entry's input contract, cache
identity, and honest provenance flags using an injected deterministic fake
encoder. They never import torch/open_clip and never touch the network.
Real-model validation lives in ``test_visual_features_real.py`` behind the
``real_model`` marker and is opt-in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import digest, read_json
from simready.core.evidence import reference
from simready.ingestion.nerf2physics import run_estimation
from simready.ingestion.visual_features import (
    CACHE_LAYOUT_VERSION,
    UPSTREAM_COMMIT,
    encode_regions,
    feature_cache_key,
)

DIM = 8  # small fake feature dimension keeps tests fast


class FakeEncoder:
    """Deterministic encoder standing in for open_clip; counts executions."""

    feature_dim = DIM
    calls = {"images": 0, "texts": 0}

    def encode_images(self, images):
        self.calls["images"] += 1
        # Each crop's per-channel mean seeds its feature: distinct regions differ.
        rows = []
        for image in images:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            channel_means = pixels.mean(axis=(0, 1))
            seed = int(np.round(channel_means * 1_000_003).sum() * 97) % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(DIM).astype(np.float32)
            rows.append(vector * (1.0 + float(channel_means.sum())))
        return np.stack(rows)

    def encode_texts(self, prompts):
        self.calls["texts"] += 1
        rng = np.random.default_rng(7)
        rows = rng.standard_normal((len(prompts), DIM)).astype(np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def fake_factory(model_spec, weights_path, device, cpu_threads):
    assert Path(weights_path).is_file(), "encoder must receive real weights bytes"
    assert model_spec["backbone"] and model_spec["pretrained"]
    report = {
        "load_seconds": 0.0,
        "torch_version": "fake",
        "open_clip_version": "fake",
        "preprocess_pipeline": {"resize_size": 224, "deviation": "test"},
    }
    return FakeEncoder(), report


@pytest.fixture
def scene(tmp_path):
    """A real-decoded PNG image plus a pinned weights stand-in file."""
    image = tmp_path / "source.png"
    array = np.zeros((64, 48, 3), dtype=np.uint8)
    array[16:32, 8:24] = (200, 40, 40)  # region A
    array[40:56, 28:44] = (30, 30, 220)  # region B
    Image.fromarray(array).save(image)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fake-weights-bytes")
    mask = tmp_path / "mask.png"
    Image.fromarray(np.where(array[:, :, 0] > 100, 255, 0).astype(np.uint8)).save(mask)
    return {"dir": tmp_path, "image": image, "weights": weights, "mask": mask}


def request_for(scene, **overrides):
    request = {
        "schema_version": 1,
        "image": reference(scene["image"]),
        "regions": [
            {"region_id": "region_a", "bbox_xyxy": [8, 16, 24, 32]},
            {"region_id": "region_b", "bbox_xyxy": [28, 40, 44, 56]},
        ],
        "model": {
            "backbone": "ViT-B-16",
            "pretrained": "datacomp_xl_s13b_b90k",
            "weights": reference(scene["weights"]),
            "expected_feature_dim": DIM,
        },
        "preprocess": {"mode": "open_clip_default", "input_size": 224},
        "texts": {
            "prompts": ["wood", "metal"],
            "prefix": "a photo of ",
            "suffix": " material",
        },
        "seed": 3,
        "device": "cpu",
        "cpu_threads": 2,
    }
    request.update(overrides)
    path = scene["dir"] / "request.json"
    path.write_text(json.dumps(request))
    return path


def run(scene, out, **overrides):
    return encode_regions(
        request_for(scene, **overrides), out, encoder_factory=fake_factory
    )


def test_encoding_runs_and_produces_valid_npz(scene, tmp_path):
    receipt_path = run(scene, tmp_path / "out")
    receipt = read_json(receipt_path)
    assert receipt["new_visual_inference_performed"] is True
    assert receipt["encoder_executed_this_run"] is True
    assert receipt["cache_hit"] is False
    assert receipt["inference_kind"] == "2d_region_clip_encoding"
    assert receipt["source_type"] == "model_inference"
    with np.load(tmp_path / "out" / "region-features.npz", allow_pickle=False) as data:
        assert data["image_features"].shape == (2, DIM)
        assert data["region_ids"].tolist() == ["region_a", "region_b"]
        assert data["text_features"].shape == (2, DIM)
        assert data["prompts"].tolist() == [
            "a photo of wood material",
            "a photo of metal material",
        ]
        assert np.isfinite(data["image_features"]).all()
        # Image features stay unnormalized (upstream patch-feature convention);
        # text features are L2-normalized (upstream get_text_features).
        norms = np.linalg.norm(data["text_features"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        image_norms = np.linalg.norm(data["image_features"], axis=1)
        assert not np.allclose(image_norms, 1.0, atol=1e-3)
    assert receipt["model"]["feature_dim"] == DIM
    assert receipt["model"]["upstream_commit"] == UPSTREAM_COMMIT
    assert receipt["model"]["upstream_backbone_match"] is True
    assert receipt["image"]["decoded_width"] == 48
    assert receipt["image"]["decoded_height"] == 64
    # Distinct regions must produce distinct features (no broadcasting).
    features = np.load(tmp_path / "out" / "region-features.npz")["image_features"]
    assert not np.allclose(features[0], features[1])


def test_geometry_alignment_disclaimed_and_shapes_distinguished(scene, tmp_path):
    receipt = read_json(run(scene, tmp_path / "out"))
    geometry = receipt["geometry_alignment"]
    assert geometry["aligned_to_3d_points"] is False
    assert geometry["depth_used"] is False
    assert geometry["cameras_used"] is False
    limitations = " ".join(receipt["limitations"])
    assert "V,N,D" in limitations and "not" in limitations
    with np.load(tmp_path / "out" / "region-features.npz", allow_pickle=False) as data:
        assert "patch_features" not in data.files
        assert "visible" not in data.files


def test_bad_requests_are_rejected(scene, tmp_path):
    bad_bboxes = [
        ([24, 16, 8, 32], "reversed"),
        ([8, 16, 8, 32], "empty"),
        ([8, 16, 24, 16], "empty"),
        ([8, 16, 24, 32, 1], "five"),
        ([8, 16, 24, 32.0], "float"),
        ([-1, 16, 24, 32], "negative"),
        ([8, 16, 999, 32], "out of raster"),
    ]
    for bbox, why in bad_bboxes:
        with pytest.raises(ValueError, match="bbox|exceeds"):
            run(
                scene,
                tmp_path / f"out-{why}",
                regions=[{"region_id": "r", "bbox_xyxy": bbox}],
            )
    with pytest.raises(ValueError, match="unique"):
        run(
            scene,
            tmp_path / "out-dup",
            regions=[
                {"region_id": "same", "bbox_xyxy": [0, 0, 4, 4]},
                {"region_id": "same", "bbox_xyxy": [4, 4, 8, 8]},
            ],
        )
    with pytest.raises(ValueError, match="preprocess"):
        run(scene, tmp_path / "out-mode", preprocess={"mode": "center_crop_512"})
    with pytest.raises(ValueError, match="schema_version"):
        run(scene, tmp_path / "out-schema", schema_version=2)
    with pytest.raises(ValueError, match="seed"):
        run(scene, tmp_path / "out-seed", seed=-1)


def test_masked_region_requires_explicit_background(scene, tmp_path):
    masked = {
        "region_id": "r",
        "bbox_xyxy": [0, 0, 30, 30],
        "mask": reference(scene["mask"]),
    }
    with pytest.raises(ValueError, match="background"):
        run(scene, tmp_path / "out-nobg", regions=[masked])
    masked["background"] = {"mode": "not-a-mode"}
    with pytest.raises(ValueError, match="background"):
        run(scene, tmp_path / "out-badbg", regions=[masked])
    masked["background"] = {"mode": "white"}
    receipt = read_json(run(scene, tmp_path / "out-bg", regions=[masked]))
    region = receipt["regions"][0]
    assert region["background"] == "white"
    assert region["mask_sha256"] == digest(scene["mask"])
    assert region["masked_pixels_inside_bbox"] > 0


def test_tampered_inputs_fail_hash_verification(scene, tmp_path):
    # Tampered image content: the reference sha256 no longer matches bytes.
    path = request_for(scene)
    scene["image"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="[Ss][Hh][Aa]256 mismatch"):
        encode_regions(path, tmp_path / "out", encoder_factory=fake_factory)
    # Restore the original image, bind the weights, then tamper the bytes
    # AFTER the request was written.
    scene["image"].unlink()
    array = np.zeros((64, 48, 3), dtype=np.uint8)
    array[16:32, 8:24] = (200, 40, 40)
    array[40:56, 28:44] = (30, 30, 220)
    Image.fromarray(array).save(scene["image"])
    path = request_for(scene)
    scene["weights"].write_bytes(b"different-bytes")
    with pytest.raises(ValueError, match="[Ss][Hh][Aa]256 mismatch"):
        encode_regions(path, tmp_path / "out-weights", encoder_factory=fake_factory)


def test_feature_dimension_and_finiteness_enforced(scene, tmp_path):
    class WrongDim(FakeEncoder):
        def encode_images(self, images):
            rows = super().encode_images(images)
            return np.pad(rows, ((0, 0), (0, 1)))

    def wrong_factory(model_spec, weights_path, device, cpu_threads):
        return WrongDim(), {
            "load_seconds": 0.0,
            "torch_version": "fake",
            "open_clip_version": "fake",
            "preprocess_pipeline": {},
        }

    with pytest.raises(ValueError, match="must be"):
        encode_regions(
            request_for(scene), tmp_path / "out-dim", encoder_factory=wrong_factory
        )

    class NonFinite(FakeEncoder):
        def encode_images(self, images):
            rows = super().encode_images(images)
            rows[0, 0] = np.inf
            return rows

    def nonfinite_factory(model_spec, weights_path, device, cpu_threads):
        return NonFinite(), {
            "load_seconds": 0.0,
            "torch_version": "fake",
            "open_clip_version": "fake",
            "preprocess_pipeline": {},
        }

    with pytest.raises(ValueError, match="finite"):
        encode_regions(
            request_for(scene), tmp_path / "out-inf", encoder_factory=nonfinite_factory
        )


def test_cache_identity_changes_with_every_semantic_input(scene, tmp_path):
    base = json.loads(request_for(scene).read_text())

    def key_for(request):
        region_metas = [
            {
                "region_id": r["region_id"],
                "bbox_xyxy": r["bbox_xyxy"],
                "mask_sha256": None,
                "background": None,
            }
            for r in request["regions"]
        ]
        weights_sha = request["model"]["weights"]["sha256"]
        return feature_cache_key(
            {
                "layout": CACHE_LAYOUT_VERSION,
                "model": {
                    "backbone": request["model"]["backbone"],
                    "pretrained": request["model"]["pretrained"],
                    "weights_sha256": weights_sha,
                },
                "image_sha256": request["image"]["sha256"],
                "regions": region_metas,
                "preprocess": request["preprocess"],
                "texts": request.get("texts"),
                "seed": request["seed"],
                "device": request["device"],
                "cpu_threads": request["cpu_threads"],
            }
        )

    baseline = key_for(base)
    assert feature_cache_key is not None
    variants = {
        "image": {"image": {"path": "other.png", "sha256": "0" * 64}},
        "bbox": {"regions": [dict(base["regions"][0], bbox_xyxy=[9, 16, 24, 32])]},
        "region_id": {"regions": [dict(base["regions"][0], region_id="renamed")]},
        "weights": {
            "model": {
                **base["model"],
                "weights": {"path": "w.pt", "sha256": "1" * 64},
            }
        },
        "backbone": {"model": {**base["model"], "backbone": "ViT-L-14"}},
        "pretrained": {"model": {**base["model"], "pretrained": "laion2b_s34b_b88k"}},
        "preprocess": {**base, "preprocess": {"mode": "squash_224"}},
        "prompt": {"texts": {**base["texts"], "prompts": ["wood", "steel"]}},
        "prefix": {"texts": {**base["texts"], "prefix": ""}},
        "seed": {"seed": 4},
    }
    for name, override in variants.items():
        changed = json.loads(json.dumps(base))
        if name == "preprocess":
            changed["preprocess"] = override["preprocess"]
        else:
            changed.update(override)
        assert key_for(changed) != baseline, f"cache key insensitive to {name}"
    # Identical request reproduces the identical key.
    assert key_for(json.loads(json.dumps(base))) == baseline


def test_cache_hit_replays_without_encoding_and_flags_stay_honest(scene, tmp_path):
    cache = tmp_path / "cache"
    FakeEncoder.calls = {"images": 0, "texts": 0}
    first = read_json(
        encode_regions(
            request_for(scene),
            tmp_path / "run1",
            cache_dir=cache,
            encoder_factory=fake_factory,
        )
    )
    assert first["new_visual_inference_performed"] is True
    assert first["cache_hit"] is False
    assert FakeEncoder.calls["images"] == 1
    first_features = (tmp_path / "run1" / "region-features.npz").read_bytes()

    second = read_json(
        encode_regions(
            request_for(scene),
            tmp_path / "run2",
            cache_dir=cache,
            encoder_factory=fake_factory,
        )
    )
    # The encoder did NOT run again; cached replay must not claim inference.
    assert FakeEncoder.calls["images"] == 1
    assert second["new_visual_inference_performed"] is False
    assert second["encoder_executed_this_run"] is False
    assert second["cache_hit"] is True
    assert second["source_type"] == "cached_feature_replay"
    # first_encoded_at points at the original encoding (cache-meta), which
    # happened strictly before this replay's receipt was written.
    assert second["runtime"]["first_encoded_at"] <= second["observed_at"]
    assert first["observed_at"] >= second["runtime"]["first_encoded_at"]
    assert (tmp_path / "run2" / "region-features.npz").read_bytes() == first_features

    # Changing one bbox invalidates the cache: encoder runs again.
    other = request_for(
        scene,
        regions=[
            {"region_id": "region_a", "bbox_xyxy": [8, 16, 24, 32]},
            {"region_id": "region_b", "bbox_xyxy": [29, 40, 44, 56]},
        ],
    )
    third = read_json(
        encode_regions(
            other, tmp_path / "run3", cache_dir=cache, encoder_factory=fake_factory
        )
    )
    assert third["cache_hit"] is False
    assert FakeEncoder.calls["images"] == 2

    # A tampered cache entry (corrupt npz) is rejected, not silently used.
    for entry in cache.iterdir():
        npz = entry / "features.npz"
        npz.write_bytes(b"not an npz")
    with pytest.raises(ValueError):
        encode_regions(
            request_for(scene),
            tmp_path / "run4",
            cache_dir=cache,
            encoder_factory=fake_factory,
        )


def test_output_rejects_existing_dir(scene, tmp_path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run(scene, tmp_path / "exists")


def test_nerf2physics_flag_stays_false_without_receipt_bound(tmp_path):
    """run_estimation without a visual_inference block keeps cached-replay semantics."""
    from examples.assets.nerf2physics_fixture import create_request

    inputs = tmp_path / "inputs"
    create_request(inputs)
    request = json.loads((inputs / "request.json").read_text())
    assert "visual_inference" not in json.dumps(request)
    request_path = inputs / "request.json"
    request_path.write_text(json.dumps(request))
    candidate = read_json(run_estimation(request_path, tmp_path / "out"))
    assert candidate["method"]["new_visual_inference_performed"] is False
    assert "cached feature replay" in candidate["method"]["visual_inference_note"]


def test_nerf2physics_receipt_binding_requires_matching_feature_bytes(tmp_path):
    """The estimation flag flips true only for byte-identical receipt outputs."""
    from examples.assets.nerf2physics_fixture import create_request

    inputs = tmp_path / "inputs"
    create_request(inputs)
    license_file = inputs / "upstream-license.txt"
    license_file.write_text("MIT license stand-in for the unit contract test\n")
    request_path = inputs / "request.json"
    request = json.loads(request_path.read_text())
    request["execution_mode"] = "feature_regression"
    request["property_name"] = "density"
    request["upstream"]["commit"] = UPSTREAM_COMMIT
    request["upstream"]["source_files"] = [reference(license_file)]
    request["upstream"]["license_file"] = reference(license_file)
    request["model_identity"] = {
        "clip_backbone": "ViT-B-16",
        "checkpoint": "datacomp_xl_s13b_b90k",
        "weights_sha256": None,
    }
    features_path = inputs / "features.npz"
    receipt = {
        "schema_version": 1,
        "inference_kind": "2d_region_clip_encoding",
        "new_visual_inference_performed": True,
        "encoder_executed_this_run": True,
        "model": {
            "backbone": "ViT-B-16",
            "pretrained": "datacomp_xl_s13b_b90k",
            "weights": {"sha256": None},
        },
        "outputs": {"features_npz": reference(features_path)},
    }
    receipt_path = inputs / "encoding-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    request["visual_inference"] = {"receipt": reference(receipt_path)}
    request_path.write_text(json.dumps(request))
    candidate = read_json(run_estimation(request_path, tmp_path / "out-bound"))
    assert candidate["method"]["new_visual_inference_performed"] is True
    assert "receipt-bound" in candidate["method"]["visual_inference_note"]

    # A receipt about different bytes must be rejected.
    other_receipt = dict(
        receipt,
        outputs={
            "features_npz": {
                "path": str(features_path),
                "sha256": "0" * 64,
            }
        },
    )
    other_path = inputs / "other-receipt.json"
    other_path.write_text(json.dumps(other_receipt))
    request["visual_inference"] = {"receipt": reference(other_path)}
    fresh = inputs / "request2.json"
    fresh.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="cannot justify|Payload features"):
        run_estimation(fresh, tmp_path / "out-mismatch")

    # A replay-only receipt (performed=false) never flips the flag.
    lazy_receipt = dict(receipt, new_visual_inference_performed=False)
    lazy_path = inputs / "lazy-receipt.json"
    lazy_path.write_text(json.dumps(lazy_receipt))
    request["visual_inference"] = {"receipt": reference(lazy_path)}
    fresh3 = inputs / "request3.json"
    fresh3.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="real encoder execution"):
        run_estimation(fresh3, tmp_path / "out-lazy")


def test_module_import_stays_light_and_cli_help_works(tmp_path):
    code = (
        "import sys, json; import simready.ingestion.visual_features as m; "
        "heavy = [k for k in ('torch', 'open_clip', 'PIL', 'numpy') if k in sys.modules]; "
        "print(json.dumps(heavy))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == [], "core import must stay stdlib-only"
    for args in (
        ["-m", "simready.ingestion.visual_features", "--help"],
        ["-m", "simready.ingestion.cli", "--help"],
    ):
        result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "request" in result.stdout.lower() or "usage" in result.stdout.lower()
