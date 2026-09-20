# ADR-0009: Local polynomial regression on real timestamps, not Savitzky–Golay on a resampled grid

**Status:** Accepted, one consequence superseded by [ADR-0021](ADR-0021-resolving-the-smoothing-window.md) · **Date:** 2026-09-16 · **Phase:** 3

## Context

Phase 3 has to turn per-frame landmark detections into a trajectory that can be
differentiated. Velocity and acceleration are what Phase 4 uses to find swing
events and what Phase 5 turns into metrics, so the error in them bounds
everything downstream.

The standard tool is a Savitzky–Golay filter: a local least-squares polynomial
fit whose coefficients collapse into a fixed convolution kernel, with analytic
derivatives available from the same fit. It is the right family of method. The
problem is its central assumption.

**Savitzky–Golay requires uniformly spaced samples.** The kernel is precomputed
for a fixed sample spacing, and Phase 1 established that consumer footage
frequently violates that: phone slow-motion switches capture rate mid-clip,
frames get dropped, and presentation timestamps jitter. Phase 1 went to some
trouble to read real presentation timestamps rather than computing
`frame_index / fps`; applying a uniform-grid kernel here would throw that away
at the first numerical step.

Applying it anyway does not fail loudly. It biases every derivative in
proportion to the local timing error, and the output still looks like a smooth
velocity curve.

## Options considered

**1. Resample onto a uniform grid, then Savitzky–Golay.** The conventional
remedy, and it works. But it interpolates the data before anything has been
measured, and commits to a grid spacing, a resampling kernel and that kernel's
frequency response — three choices made before the smoother has run, each
affecting the result, none visible in the output. It also means every value
downstream passes through two filters when it was only ever described as
passing through one.

**2. Fit a local polynomial at each sample, on the real timestamps.** Solve the
weighted least-squares problem per evaluation point instead of applying a
precomputed kernel. Samples stay where they were recorded. Savitzky–Golay is
then not an alternative but the special case this reduces to when the spacing
happens to be uniform.

**3. A Kalman smoother with a constant-acceleration model.** Principled, handles
irregular sampling natively, and gives uncertainty estimates for free. Rejected
for now on complexity: it needs a process-noise model that nothing here can yet
justify, and tuning it without ground-truth landmarks (Phase 12) would mean
choosing parameters by eye. The `FilterStage` seam is there so it can be added
later and compared against this baseline on the same input.

## Decision

**Option 2.** `filtering/localpoly.py` solves a weighted local least-squares fit
at every sample, over a window defined in _seconds_ rather than in samples, and
reads position, velocity and acceleration off the fitted polynomial's
coefficients.

Three consequences worth stating:

- **The window is a span of time.** A fixed sample count means a different
  duration on every clip, and a different duration within one clip once the rate
  varies — which is the input this exists to handle.
- **Derivatives come from the fit, not from the fitted data.** Velocity is the
  linear coefficient and acceleration twice the quadratic one. Finite-differencing
  smoothed positions would apply a second, unstated filter with a much worse noise
  response, and would leave the reported velocity inconsistent with the reported
  position.
- **Generality is verifiable, not asserted.** On a uniform grid the output must
  equal SciPy's `savgol_filter` to floating-point precision, and the test suite
  asserts exactly that for value, velocity and acceleration. If the general
  method quietly did something else, that test fails.

## Measurements

From `scripts/benchmark_filter.py` on the reference machine (Apple M1 Pro,
macOS 26.4.1). Velocity RMS error against trajectories with closed-form
derivatives, at 120 fps with the shipped defaults.

**What assuming uniform sampling costs**, measured at the landmark noise level
measured from real footage (sigma = 0.0014 normalized_frame):

| Sampling        | True timestamps | Assumed uniform | Ratio     |
| --------------- | --------------- | --------------- | --------- |
| uniform         | 0.0308          | 0.0309          | **1.00x** |
| jitter, 25%     | 0.0381          | 0.0416          | 1.09x     |
| jitter, 50%     | 0.0402          | 0.0567          | 1.41x     |
| rate change, 4x | 0.0836          | 0.8926          | **10.7x** |
| dropped frames  | 0.0637          | 0.5740          | **9.0x**  |

Noise-free, the same comparison reaches 177–1468x on a rate change, because
there the only error left is the timing error itself.

The first row is the one that makes the decision cheap: on genuinely uniform
input, doing it the general way costs nothing measurable. The rate-change and
dropped-frame rows are the ones that make it necessary, and both are ordinary
properties of phone footage rather than contrived cases.

**Cost.** Solving per window rather than convolving a kernel is more work, so it
was measured rather than assumed: filtering all 33 landmarks in three axes takes
12.3 ms for a 68-frame clip and 17.5 ms for a 240-frame clip — against the
1.2 s pose extraction takes for the same 68 frames. It is not a bottleneck and
nothing is cached as a result.

Within that, the least-squares solve uses the normal equations rather than a
pseudo-inverse per window: 9x faster for the solve step, 3.2–4.0x for the whole
fit, agreeing with the SVD to 4e-15 relative. The normal equations square the
condition number, which is affordable because the local coordinate is scaled
into [-1, 1] — a degree-3 Vandermonde over spread nodes is conditioned at order
10, so squaring costs about two of sixteen digits. A singular window falls back
to the SVD rather than being assumed away.

## Consequences

**A frame-rate floor became visible, and is now stated rather than discovered.**
The defaults (0.10 s window, degree 4) need five samples per window, which 30 fps
footage cannot supply. The filter emits nothing there and says why, including the
minimum window that clip's measured rate _would_ support. Both reference clips in
this repository are 24–30 fps, so this is the common case rather than an edge
one — and it is quantitative backing for the ≥120 fps the capture protocol asks
for, which until now was stated as reasoning rather than measurement.

The alternative — silently widening the window to whatever the clip can support —
was rejected. It produces numbers, and they would be worse in a way nothing
reported.

> **Superseded by [ADR-0021](ADR-0021-resolving-the-smoothing-window.md).** The
> sentence above this one — "Both reference clips in this repository are 24–30
> fps, so this is the common case" — turned out to be the whole argument against
> the decision it introduces. Emitting nothing on the common case meant five of
> the eight clips in `data/` were refused with "the hands were never tracked",
> pointing at pose estimation rather than at the window; and the floor was not
> accepted anywhere in practice, but worked around with a hand-picked
> `window_s=0.17` in every benchmark that touched a 30 fps clip. The window is
> now resolved against the clip's measured rate and the widening is reported in
> the result, which answers the "silently" the rejection turned on. Everything
> else in this ADR stands.

**Gap policy is separate from the fit.** Bracketing inside the fit prevents
extrapolation beyond the observed range. It deliberately does _not_ prevent
interpolating across an interior gap, because with observations on both sides
the polynomial is constrained and that is the legitimate thing a local fit does.
How far interpolation may go is a policy question, answered by `GapPolicy` and
enforced by a `blocked` mask the fit honours — and re-checked by the pipeline on
the way out, so a stage added later cannot quietly fill a refused gap.

**The seam survives.** `FilterStage` is a Protocol, so the Kalman smoother in
option 3, an outlier rejector keyed on bone-length consistency, or a learned
denoiser from Phase 12 can each replace or precede this without any caller
changing — provided it can report what it did.

## Revisit when

- Phase 12 produces labelled landmarks. Every accuracy figure here is against
  analytical trajectories chosen to resemble swing motion, not against ground
  truth, and the defaults should be re-derived when real ground truth exists.
- Phase 17 measures the pipeline end to end. If filtering has become a
  bottleneck by then, sharing one factorisation across a landmark's three axes
  is a straightforward 3x that was deliberately not taken here.
