# Data needed for future geometry/scale acceptance (research list only)

Lane simready-data-validation, 2026-11-15. **Nothing in this list was
downloaded this round** — no large reconstruction datasets, no Aholo paid
tasks, no fabricated poses or quality GT. This file records official sources
and acquisition conditions so the next lane can plan deliberately.

Why needed: the current lane measures *mask* accuracy on RGB only. The
product target (real2sim navigation inspection) additionally needs
reconstruction-quality evidence — camera-pose error vs an independent
reference, geometry/depth accuracy, and metric scale — none of which RGB
instance masks can prove.

## TUM RGB-D SLAM benchmark — recommended first step

- Official: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset> (TUM Computer
  Vision Group; download page per-sequence, no account needed).
- License: **CC BY 4.0** — permissive (attribution; cite Sturm et al.,
  "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012).
- Provides: synced Kinect v1 color+depth (640×480 @30 Hz) with
  **motion-capture ground-truth trajectories** (mm-level) — exactly the
  independent reference a pose/reconstruction acceptance needs.
- Fit: single-sequence downloads are a few hundred MB (e.g.
  `fr1_xyz`/`fr2_desk` style handheld sequences); a 1–2-sequence subset is
  enough for a small-scale geometry/scale acceptance; avoid pulling the full
  corpus.
- Caveats: indoor office/household scenes; depth noise/Kinect-v1 IR
  artifacts; scale is metric via Kinect baseline (good for scale checks, not
  a photorealism source).

## ScanNet++ — higher fidelity, gated access

- Official: <https://kaldir.vc.in.tum.de/scannetpp> (TU Munich).
- Access: **registration + agreement to the ScanNet++ Terms of Use**
  (non-commercial research/educational use; institutional email; per-release
  download credentials). Companion code is MIT; the data is not.
- Provides: sub-millimeter laser-scanned mesh GT, registered 33-MP DSLR
  images, iPhone RGB-D streams — the strongest available indoor geometry GT.
- Fit: ideal for reconstruction-vs-scan geometry metrics (mesh-to-mesh
  distance, coverage) and semantic-verification studies; heavy (hundreds of
  scenes; per-scene fetch is the sane granularity).
- Lead time: human approval step — request early, do not assume same-day.

## Cheaper complements if a full benchmark is not yet warranted

- **ETH3D SLAM benchmark** (https://www.eth3d.net) — RGB + laser GT
  trajectories, some scenes outdoors; per-scene downloads, free for research
  per its terms page (verify at fetch time).
- **7-Scenes** (Microsoft) — relocalization benchmark with known poses;
  check current hosting/terms before use.
- Any acceptance built on these must freeze pose/geometry metric definitions
  **before** reconstruction runs, mirroring this lane's PROTOCOL.md
  discipline; synthetic pose GT is never a substitute for mo-cap/laser
  references.

## Estimated cost guardrails for the next lane

| source | minimum viable subset | access lead time | license posture |
|---|---|---|---|
| TUM RGB-D | 1–2 sequences (~0.2–1 GB) | none (direct download) | CC BY 4.0 |
| ScanNet++ | 1–2 scenes (per-scene fetch) | approval required | non-commercial ToU |
| ETH3D | 1 scene set | none | research terms |

Rule of thumb from this lane: fetch the smallest official unit that carries
the GT you need (we range-extracted a 6.5 MB annotation member instead of a
241 MB archive); record ETag/sha256/license/split/IDs before running
anything; keep bytes out of git.
