# Real-data segmentation acceptance — execution record & measured accuracy

Lane simready-data-validation, 2026-11-15. This closes the "ran the model but
never measured accuracy" gap (lane B finding F-M4: no human reference masks
existed, so the 30 Starbucks instances carried no IoU/precision/recall).

Execution classification (per lane honesty rules):

- Predictions: **real_inference_new** — production GroundingDINO-tiny + SAM
  ViT-B (pinned revisions from `grounded.py`, cached weights, offline) run on
  CPU on 5 real photographs. Not synthetic, not cached replay.
- Ground truth: **gt_official_coco** — official human COCO val2017 instance
  annotations, converted one-way; model outputs never touch GT.
- Visual overlays were produced for human/Codex review; **no AI visual review
  is claimed as GT** and no human annotation was performed by this lane.

## Measured results (frozen protocol, no quality threshold pre-approved)

τ_mask = 0.50 greedy class-restricted matching (PROTOCOL.md §4–5):

| view | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| **derived production masks (PRIMARY)** | 0.417 | 0.625 | 0.500 | 15/21/9 |
| raw masks, pre overlap-resolution | 0.405 | 0.625 | 0.492 | 15/22/9 |
| derived @ τ 0.30 | 0.500 | 0.750 | 0.600 | 18/18/6 |
| derived @ τ 0.75 | 0.083 | 0.125 | 0.100 | 3/33/21 |
| detector boxes @ τ_box 0.50 | 0.528 | 0.792 | 0.633 | 19/17/5 |

GT: 24 matched instances (of 26 vocabulary annotations; 2 tiny chairs in
image 139 sit below the pipeline's own 256 px floor and are excluded by the
frozen rule, counted explicitly; 0 crowd).

Per class @ τ0.50 (derived): chair P0.588/R0.714/F1 0.645 (matched-IoU mean
0.639); bed 0.333/0.500/0.400 (IoU 0.867); couch 0.250/0.500/0.333
(IoU 0.824); dining_table 0.273/0.500/0.353 (IoU 0.585).
AP@0.5 (all-point, small-sample-unstable descriptor): chair 0.645, bed 0.5,
couch 0.5, dining_table 0.214.

## Honest failure census (findings — nothing was retuned or dropped)

- **Image 10092 scored TP 0 / FP 5 / FN 4**: the largest GT in the subset (a
  55k-px bed) produced no detection at the deployed threshold; a "chair
  couch" multi-word detection covered a real chair at IoU 0.947 but resolved
  to `unknown` (class-resolution failure — same mechanism as lane-B F-M3);
  the dining table was detected as "chair" (IoU 0.711, wrong class).
- Exact-phrase inverse mapping left 1 prediction `unknown` overall
  ("chair couch", score 0.299) → counted FP per protocol.
- Misses concentrate on large objects (bed 55k px, table 16k px, couch 18k
  px) and one small chair (1.8k px); recall at box level (0.792) exceeds
  mask recall — the detector finds objects, SAM masks/instance resolution
  lose the match at τ0.5 (τ0.75 collapse to F1 0.100 shows masks are loose
  vs human polygon GT).
- 139's tiny below-floor chairs: the pipeline could not have emitted them
  (its own floor) — recorded as excluded, not as model misses.

## What these numbers are NOT

- Not Starbucks-scene quality, not substation/drone quality, not product
  accuracy; the training-overlap disclosure (DATASET_CHOICE.md) applies —
  COCO is in-distribution for GroundingDINO-Tiny (GoldG includes COCO train).
- Not a pass/fail verdict: no quality threshold was pre-approved, and none
  was invented after seeing scores. Numbers are findings for owners.
- Not extrapolable beyond 5 images / 24 instances.

## Reproduce

```bash
VENV=<lane venv>  # python with torch-cpu/transformers/numpy/PIL/matplotlib
$VENV scripts/acceptance_real_segmentation.py fetch      # ~7 MB total download
$VENV scripts/acceptance_real_segmentation.py run --cpu-threads 2
$VENV scripts/acceptance_real_segmentation.py evaluate
$VENV scripts/acceptance_real_segmentation.py report
pytest tests/evaluation/test_real_segmentation_acceptance.py -q
```

Artifacts (not in git): input manifest + GT + runs + metrics + overlays under
`.simready-collaboration/parallel-followup-20261115/simready-data-validation/artifacts/`;
each production run's argv/timing/exit in `artifacts/logs/run_<id>.log`;
model revisions + checkpoint sha256 read from each production manifest into
`metrics.json`. Re-running `fetch` re-verifies the annotation sha256
(`e8c7f790…`) and the archive ETag; GT files are write-once.

## Test coverage of the evaluation code itself

`tests/evaluation/test_real_segmentation_acceptance.py` (23 tests, no
network/weights): perfect match, empty predictions (vacuous precision,
nothing skipped), misaligned masks (FN+FP at τ0.5, match at lower τ), wrong
class (never matches; class-error diagnostic), unknown-category FP,
one-to-one greedy (no double counting), tie-break determinism, polygon/RLE
rasterization incl. compressed-RLE refusal, rasterization sanity bounds,
deterministic selection rule (first-5-ascending, decoys ignored), AP sweep
monotonic cases, overlay smoke. One real semantics bug was caught by these
tests before any score existed: greedy pairs below τ initially bound both
sides (silently neither TP nor FN/FP) — fixed to never bind below τ per
PROTOCOL.md §4.
