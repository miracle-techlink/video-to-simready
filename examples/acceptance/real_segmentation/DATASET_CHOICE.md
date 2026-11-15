# Real-data segmentation acceptance — dataset choice & license record

Status: FROZEN 2026-11-15, before any model prediction was produced or any score
computed. This lane's gap: the 2026-11-15 lane-B acceptance ran DINO+SAM on 3
Starbucks frames with no human reference masks, so accuracy ("ran the model but
do not know the IoU") could not be measured (findings F-M4). This file records
where independent human ground truth comes from.

## Why COCO val2017 (and why not local data)

- Preferred order was "already-local, reliable-provenance data first". Searched
  the machine: no COCO or other real-image instance-mask dataset exists locally
  (HF hub cache holds EgoDemo / RoboTwin2.0 / GenieSimAssets / 10Kh-RealOmin /
  PhysicalAI-SimReady-Warehouse — simulation or non-instance-mask data; none is
  a real-image human-instance-mask benchmark).
- COCO 2017 instance val split is the standard public human-annotated instance
  segmentation benchmark; annotations are drawn by trained human annotators
  (not model outputs), which is exactly the missing ingredient. It is fetched
  from the official endpoint only.

## Official source and exact acquisition (minimal download, no full archives)

- Annotation archive: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  - ETag `f4bbac642086de4f52a3fdda2de5fa2c`, Content-Length 252907541,
    Last-Modified 2018-07-10, `Accept-Ranges: bytes` (AWS S3).
  - Only the member `annotations/instances_val2017.json` was range-extracted:
    6,529,031 compressed bytes fetched (~6.5 MB, 2.6% of the archive), inflated
    to 19,987,840 bytes,
    sha256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
  - Nothing else from the archive was downloaded (no train2017 annotations, no
    captions, no keypoints).
- Images: fetched individually from `http://images.cocodataset.org/val2017/<file_name>`
  (S3 serves single objects). 5 JPEGs, on the order of 100–200 KB each.
- No other mirror, no third-party rehost, no API keys, no paid service.

## License / usage constraints

- COCO annotations and images: the annotations are licensed by the COCO
  consortium; images retain their original Flickr licenses. Per-image license
  ids/urls come from `licenses` inside the official annotation file and are
  copied verbatim into `input_manifest.json` per image.
- The selected images are CC BY / CC BY-SA / CC BY-NC. Internal evaluation and
  research use is permitted; NC images must not be reused commercially;
  attribution is recorded per image (`coco_url`, `flickr_url`, license url).
  Because of these terms **no image bytes or annotation-derived binaries are
  committed to this repository** — they live only in the lane artifacts
  directory recorded in the run manifest, together with a re-download script.

## Frozen deterministic selection rule (defined before looking at any model output)

Universe: all 5,000 `val2017` images in the official instance annotation file.

1. Vocabulary (frozen, 4 classes): COCO categories `chair` (id 62), `couch`
   (63), `bed` (65), `dining table` (60) — furniture matches the product
   domain (indoor simready scenes) and the pipeline's existing query style.
2. Candidate filter: an image qualifies if it has ≥ 3 non-crowd annotations in
   the vocabulary, each with `area ≥ 1000` px². (GT-side statistics only;
   250 of 5,000 images qualify.)
3. Selection: sort qualifying images by `image_id` ascending, take the first
   **5** (task allows 2–5). No visual inspection, no difficulty filtering, no
   post-hoc removal. If an image later fails to download or is corrupt, the
   next image in this order substitutes and the substitution is recorded.

Resulting image ids: **139, 1993, 10092, 10707, 13659** (26 vocabulary GT
annotations total: chair ×16, couch ×2, bed ×2, dining table ×6 — counts from
GT annotations only, published here before inference so set composition cannot
be tuned after seeing scores; per the protocol, 2 of the 26 (tiny chairs in
image 139, area 210/91 px² < 256) fall below the pipeline's own min-area floor
and are excluded from matching with an explicit count).

## Training-set overlap disclosure (must accompany any score from this subset)

- `grounding-dino-tiny` was trained on O365, GoldG, Cap4M and Flickr30k-style
  grounding corpora; GoldG **includes COCO train** annotations. COCO **val**
  images are not in that training set, but the domain is in-distribution.
- `sam-vit-base` was trained on SA-1B only (no COCO); its mask quality was
  benchmarked on COCO val by the SAM authors.
- Consequence: scores on this subset are an in-distribution upper-ish reference
  for the pipeline's perception front-end. They must **not** be described as
  Starbucks-scene GT, substation-drone performance, or out-of-distribution
  zero-shot quality, and must not be extrapolated to overall product accuracy.

## Scope labels used in all outputs of this lane

Every metric file/figure carries `data_kind: "real"` and
`gt_provenance: "human_coco_val2017_official"`, plus this caveat block. Model
predictions are never written into GT files; GT conversion is one-way and both
sides record independent sha256 provenance.
