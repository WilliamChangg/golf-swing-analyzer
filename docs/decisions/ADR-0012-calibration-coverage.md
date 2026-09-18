# ADR-0012: A calibration is gated on coverage, not on reprojection error

**Status:** Accepted · Phase 8 · 2026-09-17

## Context

Camera calibration has one number everybody quotes. Every tool prints it, every
tutorial says to get it under a pixel, and it is the first thing anyone asks
about a calibration: the **RMS reprojection error**.

It measures how far each detected board corner sits from where the fitted model
puts it. Phase 8 needed a rule for deciding whether a calibration may be used,
and gating on that number is the obvious rule.

It is also wrong, in a way that is worse than merely uninformative: the number
gets _better_ as the calibration gets worse.

## The measurement

`scripts/benchmark_calibration.py` renders a Charuco board through a camera whose
intrinsics are inputs, runs the real detector on the rendered images, fits a
calibration, and subtracts. Sweeping the one thing a person controls — how much
they tilt the board — gives this (median of 3 seeds, 14 views each):

| tilt spread | RMS px | reported σ(fx) | **true fx error** | edge error px |
| ----------- | ------ | -------------- | ----------------- | ------------- |
| ±0°         | 0.220  | 0.012%         | **6.33%**         | 6.3           |
| ±2°         | 0.210  | 0.012%         | **10.04%**        | 4.0           |
| ±5°         | 0.218  | 5.17%          | 1.30%             | 6.0           |
| ±10°        | 0.269  | 1.22%          | **23.25%**        | 21.6          |
| ±20°        | 0.245  | 0.656%         | 0.10%             | 2.4           |
| ±35°        | 0.218  | 0.314%         | **0.05%**         | 2.1           |

The residual is flat. It varies by a quarter across a range over which the focal
length error varies by a factor of **four hundred**. A gate at "RMS below one
pixel" passes every row of that table.

The reason is a degeneracy. A board held parallel to the sensor at a fixed
distance cannot separate focal length from distance: a longer lens further away
makes very nearly the same picture. Tilting the board breaks it, because
foreshortening across a tilted plane depends on the focal length in a way a pure
scale does not. With no tilt, the fit is free to choose almost any focal length
and find a distance that matches — and it fits those views beautifully, because
they are exactly the views it was free to fit.

## What was expected to catch it, and did not

`cv2.calibrateCameraExtended` returns a standard deviation for every intrinsic,
propagated from the residuals through the fit's Jacobian. The design for this
phase assumed that was the answer: an ill-conditioned problem should produce an
uncertainty that explodes, and gating on it would be gating on the right thing.

**Measured, it does the opposite.** The reported σ(fx) is 0.012% on the capture
whose focal length is wrong by 6%, and 0.314% on the capture that is right to
0.05% — smallest by a factor of thirty exactly where the answer is worst.

That is not a bug in OpenCV. The degeneracy is not left unresolved: the
distortion coefficients take it up. On the ±0° row the fit returns k1 = −0.757
against a truth of −0.28, and that combination of focal length and distortion
reproduces those centred, square-on views almost perfectly. The parameter set is
genuinely well determined **by the views it was given**, and a covariance
computed from those same views has no way to know they were the wrong views. It
is answering "how much would this answer move if these measurements jittered",
when the question is "what did these measurements fail to constrain at all".

## Decision

**`CameraCalibration.usable` is decided on coverage — what the board views
sampled — and not on how well the model fits them.**

`CoverageReport` measures three things, each detecting a degeneracy that the
residual makes look _better_:

| field            | degeneracy it detects                              |
| ---------------- | -------------------------------------------------- |
| `tilt_range_deg` | focal length not separable from distance           |
| `image_fraction` | distortion fitted to a region that has almost none |
| `scale_range`    | the lens measured at a single working distance     |

The reprojection error is still computed, still reported, and still gates — at
1.0 px, deliberately loose. It catches gross failure: a mis-specified board, the
wrong ArUco dictionary, a sheet that is not flat. Those produce residuals of
several pixels, and the bound is there for them rather than for mediocrity.

The parameter uncertainty is still reported, with a 2% bound, as a backstop that
catches a visibly ill-conditioned fit. Its documentation says plainly that it
misses the case above.

Three numbers, three questions, and the UI prints all three next to what each
one actually means:

    rms_reprojection_px   how well the model fits the data          the fit
    fx_uncertainty        what the fit says about its own spread    a check
    CoverageReport        what the views could possibly determine   the gate

## Consequences

**A good capture is defined by what it samples, not by what it scores.** The
capture protocol in `data/README.md` now asks for tilt in the same breath as it
asks for the board to reach the frame corners, and both are stated as things the
system will check rather than as advice.

**Two thresholds are stated policy, informed by measurement.** `min_tilt_range_deg`
is 20° because that is where the error collapses in the sweep above and it is far
more tilt than anyone produces by accident. `min_image_fraction` is 0.35 on the
frame-coverage sweep, where 25% reach gives a 44 px edge error and 50% gives 20 px.
Neither is an optimum; both are points on a measured curve.

**The distortion model became a decision rather than a default.** OpenCV fits
five coefficients. On a good capture the fifth buys nothing measurable and costs
a factor of four at the frame edge (2.1 px against 9.5 px of edge error); on a
centred capture it is catastrophic, 180.6 px against 34.4 px, because an
undetermined term does not sit near zero — it takes whatever value cancels the
residual over the region the board occupied and then diverges outside it. The
default is four coefficients, and the fifth is available for a lens that needs
it.

**Nothing downstream may read "calibrated" as a boolean.** `CalibrationStatus`
has three values because intrinsics and stereo permit different claims, and the
metric layer gates on it centrally in `biomechanics/compute.py`. On this build
the gate blocks nothing: every shipped metric measures the image plane, which an
uncalibrated camera supplies, and a calibration makes those cleaner without
promoting any of them to a statement about three dimensions. The gate is enforced
and tested now so that Phase 9's metrics are refused by machinery that predates
them.

## What this does not establish

The camera here is synthetic. The renderer applies the exact distortion model the
fit inverts, and the images contain no motion blur, no rolling shutter, no
defocus, no JPEG ringing on the marker borders, and no printed sheet that has
bowed a millimetre off flat. **Every error in the table above is a floor.** What
the sweep establishes is the _shape_ of the relationship — that the residual is
blind to a degeneracy and coverage is not — which is a claim about the estimator
rather than about any particular camera, and it is the claim the decision rests
on.

`benchmark_calibration.py --real <dir-or-video>` runs the same pipeline over real
board footage and reports what it says, scoring nothing, because no ground truth
for a real camera exists in this project.

## Alternatives considered

**Gate on RMS alone.** Rejected on the measurement above: it passes a capture
whose focal length is wrong by 23%.

**Gate on the reported parameter uncertainty alone.** Rejected on the same
measurement, and this is the alternative that would have been chosen without it
— it is the principled-looking answer, it is what the first draft of this phase
did, and it is wrong in the one case the gate exists for.

**Fit fewer distortion coefficients so the degeneracy cannot be absorbed.**
Tested: the pinhole model on a good capture gives a 1.487 px residual, a 9.64%
focal error and 202 px of edge error. Removing the terms that absorb the
degeneracy does not recover the focal length; it just fits everything worse.

**Cross-validate: fit on half the views, measure on the other half.** This would
detect the degeneracy honestly, and it was not built because it needs the held-out
views to differ from the fitted ones in the way that matters — which is the same
coverage question, measured indirectly and more expensively. `CoverageReport`
asks it directly and can say _which_ axis is missing, which is what a person
holding a board can act on.
