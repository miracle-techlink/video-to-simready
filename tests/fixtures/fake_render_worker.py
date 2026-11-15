"""Synthetic fake worker. Result does NOT count as real rendering.

This module produces placeholder RGB/depth images from a frozen render
request payload. It exists so that CI can exercise the orchestration and
receipt-binding semantics of PR 2 without pulling in Isaac Sim, Gaussian
splats, or real camera calibration. Every emitted payload carries a
``synthetic_fixture=True`` marker; downstream consumers MUST NOT treat
these outputs as evidence of real rendering.

Inputs are plain dicts. Field names are placeholders for the schema that
PR 1 will freeze; do not couple this worker to PR 1 field details.

Fault injection flags are exposed so the CI gate can verify the failure
semantics required by the PR 5 acceptance matrix (cancel / timeout /
corrupt output paths).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Banner embedded into every emitted artifact so synthetic provenance is
# machine-detectable, not just a docstring promise.
SYNTHETIC_BANNER = "Synthetic fake worker. Result does NOT count as real rendering."


def _placeholder_rgb(view_id: str, width: int, height: int, seed: int) -> np.ndarray:
    """Return a deterministic placeholder RGB image derived from view_id.

    The pattern is a hashed stripe grid so accidental reuse of the wrong
    view's image is pixel-detectable downstream.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{seed}:{view_id}".encode()).digest()[:4], "little")
    )
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    # Stamp the view index in the top-left corner to make per-view images unique.
    base[:8, :8] = (hash(view_id) & 0xFF).to_bytes(1, "big")[0]
    return base


def _placeholder_depth(view_id: str, width: int, height: int, seed: int) -> np.ndarray:
    """Return a deterministic placeholder depth map (camera-z, metres)."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"depth:{seed}:{view_id}".encode()).digest()[:4], "little")
    )
    return rng.uniform(0.5, 5.0, size=(height, width)).astype(np.float32)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_render_request(
    request: dict,
    out_dir: Path,
    *,
    fail_at_view: int | None = None,
    timeout_after: float | None = None,
    corrupt_hash: bool = False,
) -> dict:
    """Execute a fake render request and emit receipts.

    ``request`` is a placeholder dict. Expected placeholder keys (none of
    which are load-bearing yet — PR 1 will freeze the real schema):

    - ``scene_id``: str
    - ``views``: list of dicts with ``view_id``, ``width``, ``height``
    - ``seed``: int
    - ``want_depth``: bool
    - ``render_settings``: dict echoed back as the "actual" settings receipt

    Returns the receipt dict written to ``out_dir / "receipt.json"``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    views = request.get("views", [])
    seed = int(request.get("seed", 0))
    want_depth = bool(request.get("want_depth", False))
    settings = dict(request.get("render_settings", {}))
    scene_id = str(request.get("scene_id", "unknown"))

    started = time.monotonic()
    per_view: list[dict] = []

    for index, view in enumerate(views):
        if timeout_after is not None and (time.monotonic() - started) > timeout_after:
            # Simulated timeout: emit no receipt and exit non-zero so the
            # orchestrator observes a worker failure, not a successful run.
            print(
                json.dumps(
                    {
                        "synthetic_fixture": True,
                        "event": "timeout",
                        "after_seconds": timeout_after,
                        "completed_views": index,
                    }
                ),
                flush=True,
            )
            raise SystemExit(124)

        if fail_at_view is not None and index == fail_at_view:
            print(
                json.dumps(
                    {
                        "synthetic_fixture": True,
                        "event": "injected_failure",
                        "view_index": index,
                    }
                ),
                flush=True,
            )
            raise SystemExit(7)

        view_id = str(view.get("view_id", f"view-{index}"))
        width = int(view.get("width", 64))
        height = int(view.get("height", 64))

        rgb = _placeholder_rgb(view_id, width, height, seed)
        rgb_path = out_dir / f"{view_id}.rgb.npy"
        np.save(rgb_path, rgb)
        rgb_hash = _sha256_bytes(rgb.tobytes())

        depth_hash = None
        if want_depth:
            depth = _placeholder_depth(view_id, width, height, seed)
            depth_path = out_dir / f"{view_id}.depth.npy"
            np.save(depth_path, depth)
            depth_hash = _sha256_bytes(depth.tobytes())

        if corrupt_hash:
            # Fault injection: claim the wrong digest so the orchestrator's
            # hash check fails closed.
            rgb_hash = "0" * 64

        per_view.append(
            dict(
                view_id=view_id,
                width=width,
                height=height,
                rgb_file=rgb_path.name,
                rgb_sha256=rgb_hash,
                depth_file=f"{view_id}.depth.npy" if want_depth else None,
                depth_sha256=depth_hash,
            )
        )

    receipt = dict(
        synthetic_fixture=True,
        banner=SYNTHETIC_BANNER,
        worker="tests/fixtures/fake_render_worker.py",
        scene_id=scene_id,
        seed=seed,
        want_depth=want_depth,
        render_settings=settings,
        views=per_view,
        completed=len(per_view),
        expected=len(views),
        duration_seconds=round(time.monotonic() - started, 6),
    )
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SYNTHETIC_BANNER)
    parser.add_argument("--request", type=Path, required=True, help="Frozen render request JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--fail-at-view", type=int, default=None, help="Inject failure at view index")
    parser.add_argument("--timeout-after", type=float, default=None, help="Inject timeout after N seconds")
    parser.add_argument("--corrupt-hash", action="store_true", help="Inject receipt hash mismatch")
    args = parser.parse_args(argv)

    request = json.loads(args.request.read_text())
    receipt = run_render_request(
        request,
        args.out,
        fail_at_view=args.fail_at_view,
        timeout_after=args.timeout_after,
        corrupt_hash=args.corrupt_hash,
    )
    print(json.dumps({"synthetic_fixture": True, "receipt_views": receipt["completed"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
