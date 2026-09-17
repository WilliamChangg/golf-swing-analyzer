# ADR-0011: An affine time map whose second parameter is usually refused

**Status:** Accepted · **Date:** 2026-09-17 · **Phase:** 7

## Context

Two cameras filming one swing keep two clocks. Each was started by hand at an
arbitrary moment; each runs at its own nominal rate, which the container reports
and the hardware only approximates; and either may be slow motion, whose factor
nothing in the file records. Phases 8 and 9 triangulate points from two views,
and triangulation is only meaningful for two views of the _same instant_, so the
relation between those clocks is a prerequisite for everything above it — and the
error in that relation propagates into every reconstructed point.

Three questions had to be settled: what shape the relation takes, what determines
each of its parameters, and what the system is allowed to claim about how well it
knows them.

## Decision

### The map is affine, stated at a pivot

```
target_s = reference_s + offset_s + (rate - 1) * (reference_s - pivot_s)
```

`pivot_s` is the centroid of the anchors. That choice is not cosmetic: it makes
the two fitted parameters uncorrelated, so `offset_uncertainty_s` and
`rate_uncertainty` can be reported as independent numbers and combined in
quadrature by `TimeMap.uncertainty_at`. Quoted at time zero they would be
strongly correlated, and two independent-looking error bars would overstate the
error near the anchors and understate it far from them.

### The rate must earn its degree of freedom, twice

A rate is fitted only when the anchors span at least `min_rate_span_s` (1.0 s by
default), and then kept only when it sits more than `rate_significance` standard
errors from 1.0.

The second gate came from a measurement. On a clean synthetic pair whose clocks
genuinely run in step, the fit recovers a rate of 1.0000 and the map's
uncertainty at the edges of the overlap is 2.6x its uncertainty at the pivot —
because a fitted rate has a lever arm and a constant offset does not. Keeping an
insignificant rate therefore makes the map _worse_ everywhere except between the
anchors, in exchange for describing noise. Dropping it moved the reported
confidence on a clean pair from 0.72 to 0.95 with no change in accuracy.

### The offset comes from the cross-correlation; the events supply the rest

This reverses the original design, and the benchmark is why.

Four swing events look like four clocks and are not. Perturbing the landmarks at
the noise level Phase 3 measured from real footage (sigma = 0.0014 frame widths)
and watching where each event lands over eight seeds:

| Event    | Spread     | Why                                       |
| -------- | ---------- | ----------------------------------------- |
| top      | 8 ms       | a speed minimum at the highest hand point |
| impact   | 117 ms     | a speed maximum, and a broad one          |
| finish   | 350 ms     | a threshold crossing on a decaying signal |
| takeaway | **542 ms** | a threshold crossing from rest            |

One seed put the takeaway on frame zero. Meanwhile a normalised cross-correlation
over the same pair averages several hundred samples:

| Method             | Median abs. error, 120 fps pair |
| ------------------ | ------------------------------- |
| anchored on events | 14–27 ms                        |
| cross-correlation  | **0.4–1.8 ms**                  |

Against a 3.4 ms frame-rate floor. So the offset is taken from the estimator that
has the evidence behind it, and the events are kept for the two things a
correlation cannot produce: a clock rate, and a residual.

This is **not** averaging two estimates, which this layer refuses to do. It is
using each for the quantity it can determine. Where a rate _is_ significant the
events fit stands whole, because substituting a single lag into a map whose
offset varies with time would introduce exactly the bias
`SyncQuality.method_disagreement_ms` exists to report.

### Nothing may claim to be better than the frame-rate floor

An instant located to the nearest frame carries a uniform error one frame
interval wide, standard deviation `interval / sqrt(12)`; two clips contribute one
each, in quadrature. Every reported uncertainty is the larger of the observed
scatter and that floor, so a set of anchors that agree better than their frame
rates allow is reported at the frame rates' limit rather than at its own lucky
rounding.

The floor is also the number to know before buying a camera. Two at 30 fps give
13.6 ms. Replacing one with a 240 fps camera gives 9.7 ms — a 29% gain, and it
can never beat 9.6 ms however fast that camera gets, because the 30 fps clip
contributes that much alone. Replacing both gives 1.7 ms. **Upgrading one camera
of a pair is worth a factor of sqrt(2) at most.**

## Consequences

### A second camera can measure what one camera cannot

Phase 6 recorded that a slow-motion factor "is supplied, not measured, and
nothing in the video can recover it." That is true of one clip and false of two.
Supply a factor that is 10% wrong and the fit returns a clock rate of 0.887
against a truth of 0.909; at 20% wrong, 0.787 against 0.833. The _ratio_ of two
supplied factors is recoverable even though neither clip can recover its own, and
a rate further than `clock_drift_tolerance` (2%) from 1.0 is reported as a
slow-motion problem rather than as a clock difference — because real camera
clocks do not differ by anything approaching that.

### The system cannot tell whether two clips show the same swing

It aligns swing-shaped signals. Handed two different swings by the same player it
will find the alignment that fits them best and report it. The one thing it
cannot do is make their phase durations agree, so a mismatch survives as residual
— and that residual is the only evidence available on the question.

**This limitation is not theoretical here.** The only two-angle pair in this
project, `rory_face_on.mp4` and `rory_dtl.mp4`, aligns at −258 ms with a residual
of 95.8 ms rms against a 2.4 ms floor: 40x. The four anchors disagree by −82,
−92, +90 and −116 ms. The correct reading is that these are two different swings,
which they are, and the reported confidence of 0.10 says so.

A second consequence of the same measurement is less comfortable: under realistic
landmark noise, a correctly-aligned pair scatters 34 ms and a genuinely different
swing scatters 44 ms. **The residual does not separate those cases on noisy
footage.** What does is the correlation peak's margin over its nearest rival —
0.36 for one swing, 0.09 at an 8% tempo difference, 0.01 at 15% — which is why
`SyncConfidence.agreement` is that margin for a map whose offset came from the
correlation.

### Both clips must share one smoothing window

A wider window flattens and slightly shifts the speed features both methods key
on, so smoothing two clips differently would bias the alignment by an amount
nothing measures. The consequence is that the _coarser_ clip sets the window for
the pair, and a 30 fps camera cannot support the engine's 0.10 s default at all —
Phase 3's frame-rate floor, arriving here as a refusal that names the window the
clip would support.

## Alternatives considered

**A constant offset only.** Simpler, and correct for two cameras whose clocks
agree — which is the common case, and is what the map collapses to. Rejected as
the only option because it cannot represent a wrong slow-motion factor, which is
both the likeliest real mismatch and the one thing a second camera makes
measurable.

**Weighting the least-squares fit by anchor confidence.** A weight asserts an
inverse variance, and nothing here has measured that relationship: Phase 4's
confidence is a product of signal margin, landmark visibility and frame-rate
resolution, only the last of which is a statement about _when_. Every anchor
counts once and its confidence is reported rather than applied.

**Gating anchors on a minimum confidence.** Measured and abandoned: Phase 4
scores a takeaway at 0.95 on seeds where it landed 500 ms from the truth, so its
confidence does not detect this failure and a gate on it changes nothing (22–28 ms
error at every threshold from 0.0 to 0.5). Worth recording as a Phase 4 finding
rather than a Phase 7 fix.

**Averaging the two estimators.** Rejected on principle. Averaging two estimates
that disagree produces a third number that matches neither and hides the fact
that something is wrong. They are kept apart and their difference is reported.

**Robust line fitting (Theil–Sen) for the outlier check.** Measured and rejected:
with four anchors and one outlier, three of six pairwise slopes are contaminated,
which is exactly the breakdown point. A median-offset check at 0.30 s does the job
— it drops the spurious takeaway a clip starting mid-backswing produces (0.47 s
out) while keeping all four anchors at slow-motion factor errors up to 40%.
