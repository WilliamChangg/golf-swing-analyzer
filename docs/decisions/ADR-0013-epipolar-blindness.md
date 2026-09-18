# ADR-0013: The reprojection residual is not the quality of a reconstruction

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 9 — Multi-view 3D reconstruction

## Context

Two calibrated cameras, two rays, one point. Triangulation is the operation every
earlier phase has been qualifying its output against: Phase 5 labelled its angles
`PROJECTED_ANGLE` because one camera cannot see depth, Phase 6 tagged every
metric with the camera view because a projection means different things from
different places, and Phase 8 returned unit bearings and deliberately shipped no
function that turns one into a point.

The obvious way to report how good a reconstruction is, and the way every
textbook treatment scores the "gold standard" two-view method, is the
**reprojection error**: take the fitted 3D point, project it back into both
images, and measure how far it lands from the two detections. It is the quantity
the non-linear refinement minimises. It has the right units. It is small when
things are going well.

Phase 8 had already found that the analogous number for a calibration — RMS
reprojection error over the board views — is blind to the failure that matters,
and that the fix was to gate on the **capture** (`CoverageReport`) rather than on
the fit. Phase 9 asked the same question one layer up, and the answer is the
same shape and a different fact.

## Decision

**The ray convergence angle is the gate. Bone-length consistency is the
independent check. The reprojection residual is reported, and is neither.**

A reconstructed point is refused when the two rays met at less than
`min_convergence_deg` (default 15°), before its residual is looked at. The
residual gates too, at a deliberately loose `max_reprojection_px` (default 6 px),
and what it catches there is gross failure — a mis-paired instant, one camera
tracking the wrong wrist — not mediocrity.

## Why: the residual is blind along the epipolar line

A point detected in camera 1 defines a ray. Every 3D point on that ray projects
into camera 2 along a single line — the epipolar line. Split camera 2's detection
error into two components:

- **Across the epipolar line.** No 3D point on camera 1's ray projects there. The
  two rays are skew, the optimiser splits the difference, and the miss lands in
  the residual. **Visible.**
- **Along the epipolar line.** There _is_ a point on camera 1's ray that projects
  exactly there — further along, or nearer. The fit is perfect and the depth is
  wrong. **Invisible.**

So the residual measures one of the two components of correspondence error and
cannot see the other, and the other is the one that moves the answer in depth.

`scripts/benchmark_reconstruct.py --sweep epipolar` displaces every landmark in
one view along its epipolar direction and measures both:

| displacement | median 3D error | **reprojection** | bone variation |
| ------------ | --------------- | ---------------- | -------------- |
| 0 px         | 0.0 mm          | **0.00 px**      | 2.6%           |
| 1 px         | 1.4 mm          | **0.00 px**      | 4.8%           |
| 2 px         | 2.7 mm          | **0.00 px**      | 9.6%           |
| 4 px         | 5.4 mm          | **0.00 px**      | 19.1%          |
| 8 px         | 10.8 mm         | **0.00 px**      | 37.8%          |

The residual is not merely insensitive here, it is **exactly zero** across a
sweep that moves the answer by a centimetre, because every one of those displaced
detections has a perfect 3D explanation.

The complement matters as much, and it is why the residual is still reported and
still gates. The same pixel counts applied _isotropically_:

| sigma | median 3D error | reprojection |
| ----- | --------------- | ------------ |
| 1 px  | 2.0 mm          | 0.31 px      |
| 2 px  | 4.1 mm          | 0.63 px      |
| 4 px  | 8.1 mm          | 1.25 px      |
| 8 px  | 16.3 mm         | 2.51 px      |

Isotropic noise has an across-epipolar half, so the residual tracks it
proportionally. The residual is a real measurement of a real thing. It is simply
not the accuracy.

For the camera pair this system is built for — one face-on, one down-the-line,
about a right angle apart — the epipolar lines run nearly horizontally in both
images, and nothing about a pose estimator makes its horizontal error smaller
than its vertical one. Roughly half the error by variance lands in the half the
residual cannot see.

## Why the convergence angle is the gate

How much 3D error a given along-epipolar displacement produces depends only on
the angle between the two rays where they meet. Depth error scales as
`sigma_px * Z / (f * sin(theta))`.

`--sweep convergence`, at the 2.7 px landmark scatter Phase 3 measured on real
footage:

| camera separation | ray angle | median error | **reprojection** | uncertainty |
| ----------------- | --------- | ------------ | ---------------- | ----------- |
| 90°               | 94°       | 5.5 mm       | **0.85 px**      | 5.4 mm      |
| 60°               | 64°       | 5.8 mm       | **0.84 px**      | 6.7 mm      |
| 45°               | 48°       | 6.5 mm       | **0.84 px**      | 8.7 mm      |
| 30°               | 32°       | 8.2 mm       | **0.84 px**      | 12.7 mm     |
| 15°               | 16°       | 14.3 mm      | **0.84 px**      | 25.0 mm     |
| 8°                | 9°        | 26.0 mm      | **0.84 px**      | 46.6 mm     |

**The residual is flat to two decimal places across a range over which the error
grows 4.7x.** That is the same shape as Phase 8's tilt sweep, and it is the
reason the gate describes the capture rather than the fit: the convergence angle
is a property of where the tripods went, computable per point with no ground
truth, and it is what decides whether the data could have determined the answer.

A capture shot as `data/README.md` asks — one camera face-on, one down the line —
converges at nearly 90° and is nowhere near the bound. The bound exists for the
capture where both cameras ended up on the same side of the player.

## Why bone length is the independent check

A point sliding along its own ray changes its distance to its neighbours, so the
displacement the residual cannot see is fully visible in the length of the bones
that end at it. And a bone does not change length during a swing, which makes
this a check that needs no ground truth and no anatomical table: the **variation**
of a reconstructed segment across a clip is error, whatever its absolute value.

The epipolar table above is the evidence: bone variation goes 2.6% → 37.8% across
a sweep the residual reports as 0.00 px throughout.

Two qualifications, both measured:

- **Not every segment in `POSE_CONNECTIONS` is a bone.** Shoulder-to-hip on each
  side genuinely changes length as a torso twists — the synthetic body, whose limb
  lengths are exact by construction, varies those two by 2.6% across a swing from
  the shoulders and pelvis turning different amounts. That is the floor in the
  table above, and it is why `max_bone_variation` is 10% rather than 2%.
- **Left-right symmetry is corroboration, not a second instrument.** It was built
  on the reasoning that a consistent depth bias would give a _stably_ wrong length
  that variation could not see. Measured, that reasoning is wrong: a swing rotates
  the body, so a displacement constant in the camera's frame is not constant
  relative to the bone. Displacing one elbow 5 cm along the optical axis makes the
  forearm's variation 11% — past the bound — while the left-right disagreement is
  2.6%, well inside it. It is kept because it localises: it says which _side_ is
  being reconstructed worse, which a per-segment number does not.

## What this costs, and what the alternative would have been

Gating on the residual alone would accept the 8° capture in the table above —
26 mm of error, reported with a 0.84 px residual that looks indistinguishable
from the 90° capture's 0.85 px. It would also accept any amount of
along-epipolar landmark error, which is where the pose estimator's error
preferentially lands on the very camera geometry this system recommends.

Reporting all three, each labelled with the question it answers, costs three
columns in a table:

```
reprojection_px      how well the two views agree, across the epipolar line
convergence_deg      what the capture could possibly determine       the gate
bone variation       an independent check the residual cannot make
uncertainty_m        what the first two imply, in metres             the answer
```

## Consequences

- `ReconstructionQuality` carries all four, and `CalibrationQuality`'s shape is
  deliberately echoed so a reader who has met one meets the other.
- `min_convergence_deg` is checked **before** `max_reprojection_px`, and the
  order is load-bearing: a point that fails the angle test will usually pass the
  residual test, and refusing it for a residual it did not fail would tell
  someone to re-calibrate when the fix is to move a camera.
- `positional_uncertainty` propagates a **measured** per-view pixel sigma — the
  filter's own residual RMS on that clip — through the triangulation Jacobian,
  which carries the convergence angle automatically. It is the number to quote.
- A residual of exactly zero from the filter is not treated as evidence of
  perfect landmarks: that is what an exactly-determined local fit produces, so it
  falls back to the 0.0014 frame widths Phase 3 measured on real footage.

## Alternatives considered

**Gate on the residual, as the textbooks score it.** Rejected on the two sweeps
above. It is the number that a degenerate capture makes _better_, because two
nearly-parallel rays can be brought into agreement by sliding a point a long way
in depth.

**Gate on the propagated uncertainty alone.** It does carry the convergence
angle, so it would catch the geometry — but it also grows with distance and with
the pixel sigma, so a refusal would not say which of the three to fix. Both are
reported and the angle is the gate, for the same reason Phase 8 reports the
parameter uncertainty and gates on coverage.

**Compare against MediaPipe's `HIP_LOCAL` output as a sanity check.** Rejected:
those are a single-camera model's guess at a body-centred pose, not a
measurement, and using them to validate a measurement would make the measurement
answerable to the guess.

**Reconstruct into a world frame and check the ground plane is level.** That
would be a real independent check, and it needs a gravity direction this build
does not measure. Recorded in `docs/coordinate-systems.md` as what a capture
would have to contain.

## References

- `analyzer/contracts/reconstruction.py` — the argument, as a module docstring
- `analyzer/reconstruction/triangulate.py` — the three quantities
- `analyzer/reconstruction/skeleton.py` — the independent check
- `scripts/benchmark_reconstruct.py` — `--sweep epipolar`, `--sweep convergence`
- `python/tests/test_reconstruction.py` — the negatives, as tests
- [ADR-0012](ADR-0012-calibration-coverage.md) — the same shape, one layer down
