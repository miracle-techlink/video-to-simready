# Frozen evaluation protocol — real-data segmentation acceptance v1

Frozen 2026-11-15 **before** any DINO/SAM prediction was generated on the
selected images and before any score was computed. Any later change requires a
new protocol version and a rerun; nothing may be adjusted after seeing scores.
No quality threshold was pre-approved for this pipeline, so **no pass/fail
threshold is defined here** — measured numbers are reported as findings.

## 1. Units under test

The production entry `python -m simready.ingestion.segmentation.grounded`
(commit 6a823ef, file `src/simready/ingestion/segmentation/grounded.py`,
worker sha256 recorded in each run manifest), pinned models
`IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71`
and `facebook/sam-vit-base@70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`, run on
CPU with 2 threads, `--local-files-only`, and otherwise **default parameters**
(box_threshold 0.25, text_threshold 0.25, min_area_px 256,
min_retained_fraction 0.2, nms_iou 0.5, max_detections 48). The acceptance
measures the deployed configuration as-is.

Query vocabulary (single shared prompt, alphabetical as required by the
entry): `bed`, `chair`, `couch`, `dining table` (categories
`bed`,`chair`,`couch`,`dining_table`).

## 2. Ground truth construction (one-way, from official COCO annotations)

- Per selected image, every official annotation in the frozen vocabulary is
  converted to a binary mask at native image size: all polygons of the
  annotation are rasterized (even-odd fill per polygon, union across polygons)
  and validated so that mask pixel count ≤ annotation `area` × 1.10 + 8 px
  (rasterization tolerance); uncompressed-RLE (`counts` list) annotations are
  decoded if `iscrowd` is 0.
- `iscrowd == 1` annotations: **excluded** from matching and counted per image
  as `gt_crowd_excluded` (COCO crowd masks are region-level, not instances).
- Non-crowd vocabulary annotations with `area < 256` px² (the pipeline's own
  `min_area_px` floor): **excluded** from matching, counted as
  `gt_below_pipeline_floor` — the pipeline documents it cannot emit masks below
  that floor, so they are outside the measured operating range.
- Non-vocabulary annotations (person, vase, …): outside the evaluation
  vocabulary entirely; predictions can never match them (see FP rule) and they
  are never FN.
- GT artifacts (per-instance PNG + a `gt_manifest.json` with per-mask sha256)
  are written once; the evaluator refuses to overwrite them and records their
  hashes. Model outputs may never be written into GT paths.

## 3. Prediction sets evaluated (two, reported separately)

- `derived` (PRIMARY): the pipeline's `instances` — mutually exclusive masks
  after the production overlap-resolution heuristic. This is the artifact a
  downstream consumer would receive.
- `raw` (SECONDARY): every candidate after class-aware NMS with its raw SAM
  mask (pre-overlap-resolution), including `unknown`-category candidates.
  Shows model capability before the pipeline's disambiguation heuristic.
- Both sets keep the detector `score`, `category`, `raw_label`,
  `sam_predicted_iou`, and box.

## 4. Matching rule (class-restricted greedy, deterministic)

For each image and each prediction set separately:

1. Candidate pairs: (GT instance g, prediction p) with the same mapped
   vocabulary category. COCO name → query category mapping is the identity
   except `dining table` → `dining_table`. Predictions with category `unknown`
   have no candidate pairs.
2. Mask IoU is computed on binary masks at native size:
   `IoU = |g∩p| / |g∪p|` (empty union → 0).
3. Greedy assignment: sort candidate pairs by (−IoU, gt_id, pred_id), accept a
   pair only if both members are still unmatched. Deterministic under ties.
4. At IoU threshold τ: accepted pair with IoU ≥ τ → TP (record the IoU);
   unmatched GT instance → FN; unmatched prediction → FP (this includes every
   `unknown`-category prediction and any prediction whose class differs from
   the GT it overlaps — class errors are FPs in the predicted class and leave
   the GT as FN; they are additionally tallied as `class_error_pairs` when the
   best cross-class IoU ≥ τ, purely diagnostic).

## 5. Metrics (all computed per image, per class, and micro-averaged)

Primary operating point (frozen): τ_mask = 0.50, prediction set `derived`,
score cutoff = none extra (production default box_threshold 0.25 already
applied by the entry).

- Precision = TP/(TP+FP) (null when TP+FP = 0), Recall = TP/(TP+FN)
  (0 when GT = 0 is impossible here — every selected image has ≥3 GT),
  F1 (null when P and R both null/0 denominator).
- Matched IoU: mean, median, min, max over TPs; per-instance table with
  gt_id/pred_id/category/IoU/score/areas.
- Miss/fault census: FN list with areas (misses), FP list with
  `raw_label`+score (false alarms incl. `unknown`), crowd-excluded and
  below-floor GT counts.
- Secondary descriptors (not decision inputs): same census at τ ∈ {0.30,
  0.75}; box-level greedy matching (GT bbox from official `bbox` field vs
  `detection_bbox_xyxy`) at τ_box 0.50; the `raw` prediction set at τ 0.50;
  score-sorted precision/recall points and all-point AP per class over the
  detector score axis (flagged `small_sample_ap_unstable`, n=26 GT).

An image with zero predictions is never skipped: all its GT instances are FN
and it contributes TP=0 to micro counts.

## 6. Overlays (visual evidence for review)

Per image and prediction set: source JPEG with GT contours in green, TP masks
outlined blue, FP outlined red, FN GT filled yellow; PNG written next to
metrics with sha256 recorded. Overlays are produced from the same matched
result files — review aid, not a scoring input.

## 7. Reproducibility record (required in every run)

Input manifest (image ids, file names, official URLs, license ids/urls,
per-file sha256), annotation-file sha256, GT manifest with per-mask sha256,
exact production CLI argv, model revisions + checkpoint sha256 (read from the
production manifest — not re-declared by hand), python/torch/transformers
versions, thread caps, wall-clock timings, and the full per-instance JSON.
Execution classification per output: this lane reports
`real_inference_new` for DINO/SAM runs and `gt_official_coco` for GT; no
synthetic number may be presented as a real-data score.

## 8. Honesty rules carried into the report

- These scores are COCO-val in-distribution measurements of the perception
  front-end. They are not Starbucks/substation scene quality and not product
  accuracy (see DATASET_CHOICE.md overlap disclosure).
- Failures are findings: no image may be dropped, no threshold retuned, no GT
  edited after the fact. If a score looks bad it is reported as measured.
