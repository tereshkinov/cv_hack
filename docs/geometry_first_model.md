# Geometry-first global association

This implementation deliberately does not build an occupancy grid. It consumes
the existing camera topology, route timing, same-camera boundary motion and
floor/Kalman evidence when calibration is available.

## Modes

- `geometry` builds global tracks from geometry only. Embeddings are not
  required and do not affect the score.
- `geometry_visual` makes the same geometric decision first, then permits a
  bounded visual confirmation. The default visual adjustment is at most `0.08`
  and missing or physically contradictory geometry cannot be rescued by visual
  similarity.

Both modes require `--fragment-mode rebuild`. This prevents provisional online
visual IDs from leaking into the final partition. The fragment graph solves the
global linking problem rather than making a sequence of irreversible greedy
decisions.

## Geometric evidence and constraints

The boundary model uses, in descending order of physical specificity:

1. floor coordinates and Kalman prediction when camera calibration exists;
2. configured route length and travel-time feasibility between cameras;
3. same-camera image-space position, velocity and apparent scale;
4. camera overlap/transition topology as an eligibility constraint.

Negative sequential time and trustworthy spatial/travel-time contradictions are
hard refusals. Simultaneous fragments cannot be merged indirectly through a
later fragment when they lack direct geometric support. This prevents a common
star-shaped false merge.

Low-reliability geometry is pulled toward neutral rather than silently treated
as exact. The normalized geometry score is:

```text
0.5 + (compatibility - 0.5) * (0.5 + 0.5 * reliability)
```

The optional appearance adjustment is:

```text
2 * (appearance - 0.5) * visual_weight * appearance_reliability
```

## Reproducible local run

```bash
.venv/bin/python -m src.association \
  --cameras runs/test-stand/20261107-104047-7c33a92f/cameras.json \
  --observations runs/test-stand/20261107-104047-7c33a92f/vision/observations.jsonl \
  --project configs/test_stand.local.json \
  --output-dir runs/test-stand/geometry-only \
  --decision-mode geometry \
  --fragment-mode rebuild \
  --no-offline-reconcile \
  --tracklet-threshold 0.62

.venv/bin/python -m src.association \
  --cameras runs/test-stand/20261107-104047-7c33a92f/cameras.json \
  --observations runs/test-stand/20261107-104047-7c33a92f/vision/observations.jsonl \
  --project configs/test_stand.local.json \
  --output-dir runs/test-stand/geometry-visual \
  --decision-mode geometry_visual \
  --geometry-visual-weight 0.08 \
  --fragment-mode rebuild \
  --no-offline-reconcile \
  --tracklet-threshold 0.62

.venv/bin/python scripts/compare_geometry.py \
  --geometry runs/test-stand/geometry-only \
  --geometry-visual runs/test-stand/geometry-visual \
  --output runs/test-stand/geometry-comparison
```

`summary.json` reports diagnostic counts. `edge_review.csv` is the small manual
test stand: review the selected fragment links and fill `human_same_person` and
`comment`. Track count alone is never reported as accuracy.

## Current limitation

The local test project has no camera-to-floor calibration. Therefore the current
run can use route timing and same-camera motion, but it intentionally leaves
ambiguous simultaneous cross-camera appearances unresolved. Once another
component supplies floor projection/calibration, the existing floor/Kalman
provider will be consumed by these modes without changing the decision policy.
