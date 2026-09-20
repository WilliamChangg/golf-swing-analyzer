# ADR-0019: A viewpoint decides which errors a 3D picture can show, so the viewport reports which one it is in

**Status:** Accepted
**Date:** 2026-09-18
**Phase:** 15 — 3D visualisation

## Context

Phase 9 triangulates two calibrated views into metres and reports, per point, a
**positional uncertainty**: the standard deviation along the worst-determined
direction, propagated from a measured pixel sigma through the triangulation's own
Jacobian. That number is honest, it is gated on, and every report in this project
quotes it.

Phase 15 draws those points. And a drawing has a property a report does not: it
is made **from somewhere**.

A joint is drawn as a dot, and a dot carries no direction. So everything the
reconstruction does not know about that joint lands in one of two places. Across
the line of sight, where it spreads over the picture and a reader can see it. Or
**along** the line of sight, where it sits behind the dot and contributes
nothing — and the picture then looks exactly as confident as a picture of a
perfectly determined point.

Which of those happens is decided entirely by where the viewer's camera is. And
the camera almost nobody moves is the default one.

### The measurement

`scripts/benchmark_viewport.py --sweep convergence`, on the synthetic stereo
fixture at the 2.7 px landmark scatter Phase 3 measured on real footage. The
cameras are brought together from a right angle, which is what a person filming
with two phones on one side of a hitting bay actually does:

| separation | ray angle | true sigma  | visible fraction | **sigma on screen** |
| ---------- | --------- | ----------- | ---------------- | ------------------- |
| 90°        | 94°       | 5.5 mm      | 0.96             | **5.3 mm**          |
| 60°        | 64°       | 7.3 mm      | 0.74             | **5.3 mm**          |
| 45°        | 48°       | 9.4 mm      | 0.57             | **5.3 mm**          |
| 30°        | 32°       | 13.7 mm     | 0.39             | **5.3 mm**          |
| 20°        | 21°       | 20.2 mm     | 0.26             | **5.3 mm**          |
| 15°        | 17°       | **25.6 mm** | 0.20             | **5.2 mm**          |

**The last column is flat to one decimal place across a range over which the
third column grows 4.7x.** The picture a reader sees from the default viewpoint
is identical for a capture that determines a joint to five millimetres and one
that determines it to twenty-six.

This is not a coincidence and it is not a bug. A stereo pair constrains a point
well **across** each camera's own image plane — at roughly `sigma_px * Z / f`,
which does not depend on where the second camera is — and badly along the
bisector of the two rays. As the cameras close up, the bisector swings towards
the reference camera's optical axis and the poorly-determined direction goes
into depth, where a viewer standing at that camera cannot see it. The error
grows, and grows in exactly the direction the default view is blind to.

It is the same shape as two findings this project has already made, one layer up
each time:

- **Phase 8**: the calibration reprojection residual is flat (0.21–0.27 px) while
  the true focal-length error varies by a factor of four hundred.
- **Phase 9**: the triangulation reprojection residual is flat (0.84–0.85 px)
  while the true 3D error grows 4.7x with camera separation, and is _exactly_
  zero for an along-epipolar displacement.
- **Phase 15**: the uncertainty a viewport draws is flat while the uncertainty it
  is drawing grows 4.7x.

Three different numbers, three layers apart, each blind to the capture in the
same direction. The third one is worse than the first two in one respect: the
first two are numbers a reader might think to distrust, and the third is a
picture.

### What a single figure cannot express

`ReconstructionQuality.median_uncertainty_m` is a radius. A radius drawn around a
point is a sphere, and the thing being described is not a sphere:

| separation | ratio of longest to shortest axis |
| ---------- | --------------------------------- |
| 90°        | ~1.1                              |
| 30°        | ~2.6                              |
| 15°        | ~5.0                              |

Drawing the worst axis as a circle overstates the other two, and — the part that
matters — makes every viewpoint look equally informative, which is the precise
claim the table above falsifies.

## Decision

**Ship the whole covariance, not its largest eigenvalue; open the viewport at the
camera that took the footage; and report, as a number that changes when the
reader drags the mouse, what fraction of the uncertainty the current viewpoint
can show.**

Four parts.

**1. `uncertainty_covariance` returns the `3x3`.** Phase 9's
`positional_uncertainty` takes the smallest eigenvalue of `J'J` and is unchanged
— it is what every gate and report uses, it is cheaper, and it is what a reader
of numbers needs. The new function builds `sigma^2 (J'J)^-1` from that matrix's
own eigendecomposition rather than by inverting it, because the systems a shallow
convergence angle produces are nearly singular and a direct inverse loses its
precision first in the smallest eigenvalue — which is the largest axis of the
covariance and the entire reason for computing one. A test asserts the two agree
to `1e-9`, because "the same by construction" across two functions is a claim and
not a guarantee.

**2. `PointUncertainty` carries six numbers and the scalar.** The six unique
elements of the covariance in scene metres squared, plus `sigma_m` — so a
consumer that needs both cannot produce two figures that disagree. The viewport
draws `A S A'` for its own projection's `2x3` Jacobian, which is the ellipse the
covariance really projects to: small and round from the reference camera on a
shallow capture, and opening out as the reader orbits.

**3. `visible_uncertainty_fraction` is the number the viewport reports.** The
largest standard deviation across the line of sight, over the largest there is:

    visible = sqrt(lambda_max(P S P')) / sqrt(lambda_max(S))

with `P` an orthonormal basis of the plane across the view. It is a ratio, so it
says nothing about whether a reconstruction is good — a precise point and a
hopeless one both score 1.0 from a viewpoint that shows their uncertainty
honestly. **It is a property of the view**, which is what a viewport control needs
to report and what no number Phase 9 produced could have been.

`ViewpointPanel` shows it beside what the measurement is worth, and warns below
0.7 with the one instruction that helps: orbit, because the error does not change
— only whether you can see it.

**4. The default is the reference camera, exactly.** Not "a view that resembles
the video": the scene carries that camera's real `fx, fy, cx, cy`, and the
viewport at azimuth zero is the projection that produced the footage, run again
on the reconstruction. That makes the 3D skeleton and the 2D overlay two
renderings of one measurement, so a disagreement between them is a real
disagreement rather than a rendering artefact — and it is asserted, in both
languages, against the engine's own `StereoGeometry.project`.

Opening at the most flattering viewpoint may look like the wrong choice given
everything above. It is the right one, for the same reason the overlay draws
filtered landmarks rather than raw ones: **the default view is the one that can
be checked against the video**. A viewport that opened at an arbitrary angle
would be showing a picture nothing on screen could contradict. The answer to a
flattering default is not a different default — it is a viewport that says how
flattering it is being.

## Consequences

**A capture that cannot be trusted now says so twice**, in a number and in the
shape of the ellipses, at the moment a reader is looking at the body rather than
at a table.

**The covariance costs bytes.** A scene point carries a position, six covariance
elements and three diagnostics, against an overlay point's two coordinates:
12.5 KB per frame measured, against roughly 3 KB for an overlay frame. That set
`MAX_SCENE_FRAMES` at 600 — measured at 158 ms to build, 24 ms to serialise and
12 ms for the browser to parse a 312-frame scene, against roughly 1.3 s per clip
to extract poses. A columnar encoding would cut it several-fold and was not
taken: the per-point object is what carries the "null exactly when refused"
invariant into TypeScript, and a positional array would move that check from the
type system into every consumer.

**The `sigma_m`/covariance pair can in principle disagree.** They are produced by
two functions from the same inputs, which is why the anti-drift test exists and
why `visibleFraction` clamps at 1.0 rather than reporting an impossible fraction.

**There is still no gravity.** The orbit turns about the reference camera's own
vertical, which is a statement about the picture and not about the world — the
scene is `CAMERA` metres, and `ReconstructionScene` says a viewport drawing a
ground plane or a horizon would be drawing an assumption. The viewport draws
neither.

**Nothing here has been run on a real reconstruction.** Every figure above is
from the synthetic stereo fixture, whose body is an input and which contains no
pose estimator — so the millimetres are a floor. What is exact is the viewpoint
arithmetic, which is geometry and does not care whether the points came from a
real camera. This is Phase 9's limitation inherited unchanged: this repository
contains no two clips that are one swing, and no calibration of the phones that
shot the clips it does contain.
