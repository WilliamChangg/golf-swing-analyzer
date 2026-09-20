"""Local polynomial regression on real timestamps.

This is the numerical core of the analysis pipeline, and the place where a
standard tool would have been quietly wrong.

**Why not Savitzky-Golay directly.** Savitzky-Golay is a local least-squares
polynomial fit whose coefficients collapse into a fixed convolution kernel
*because the samples are uniformly spaced*. Phone footage is routinely variable
rate -- slow-motion capture especially -- so the spacing assumption fails on
exactly the clips this system is built for. Applying the kernel anyway does not
error; it biases every derivative by an amount proportional to the local timing
error, which then propagates into every metric.

The usual remedy is to resample onto a uniform grid and then filter. That is one
interpolation of the data before any of it is measured, and it commits to a
grid, a resampling kernel, and its own frequency response, all before the
smoother has run. Instead this module keeps the samples where they are and
solves the local weighted least-squares problem at each evaluation point
directly. Savitzky-Golay is then not an alternative but a special case: on a
uniform grid with uniform weights, this produces the Savitzky-Golay result to
floating-point precision, which `tests/test_localpoly.py` asserts against SciPy.

**Derivatives come from the fit, not from the fitted data.** Having fitted
p(u) = c0 + c1*u + c2*u^2 + ... locally, velocity is c1 and acceleration is 2*c2,
read straight off the polynomial. Finite-differencing a smoothed signal instead
would apply a second, unstated filter with its own -- much worse -- noise
response, and would leave the reported velocity inconsistent with the reported
position.

**Where it refuses.** A fit is emitted only when its window holds enough real
observations to determine the polynomial, and only when those observations
bracket the evaluation point in time. Bracketing rules out *extrapolation* --
producing a value before the first observation or after the last one, where the
polynomial is unconstrained and diverges fastest.

Bracketing deliberately does **not** rule out interpolating across an interior
gap: with observations on both sides the polynomial is constrained, and
recovering a value between them is the legitimate thing a local fit does. How
far that is allowed to go is a policy question rather than a numerical one, and
it is answered in `gaps.py`, which marks over-long absences `blocked` and passes
the mask in here. The two checks are separate because they prevent different
failures, and neither substitutes for the other.

Everywhere the fit declines, the output is NaN and the reason is counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Guard on the number of samples one window may contain. A window is sized in
# seconds, so a densely sampled clip makes it large, and the solver allocates
# (samples x window x order) floats. 1024 samples in a window is already far
# past anything a smoothing window should cover; hitting it means the window or
# the clip's sample rate is not what the caller thought.
MAX_WINDOW_SAMPLES = 1024


class LocalPolynomialError(ValueError):
    """The fit was asked for something it cannot compute."""


def narrowest_window_s(interval_s: float, required_observations: int) -> float:
    """Narrowest window holding `required_observations` samples at this spacing.

    Samples sit `interval_s` apart, a window of width w centred on one of them
    reaches `w / 2` either side, so it holds `2 * floor(w / (2 * interval)) + 1`
    of them. Inverting that for the fewest a fit needs gives `(n - 1) * interval`;
    the half-interval of headroom here keeps the window's boundary off a sample,
    where floating-point luck would otherwise decide whether it is counted.

    **Narrowest, not comfortable.** Every extra sample averages over more of the
    swing, and the velocity peak is what phase detection keys on: on the 30 fps
    reference footage, widening past this by a single frame costs about a quarter
    of the impact confidence, and by three frames costs all of it.
    """
    if not np.isfinite(interval_s) or interval_s <= 0:
        return float("nan")
    return float((required_observations - 0.5) * interval_s)


def sampling_interval_s(t: NDArray[np.float64]) -> float:
    """Median spacing of a clip's timestamps, or NaN if there is no spacing to measure.

    Median rather than mean: a variable-rate clip with one long stall between
    frames has an honest typical spacing and a mean that describes neither part
    of it.
    """
    if t.size < 2:
        return float("nan")
    interval = float(np.median(np.diff(t)))
    return interval if interval > 0 else float("nan")


@dataclass(frozen=True)
class LocalPolynomialFit:
    """The fit evaluated at every input sample.

    `value`, `velocity` and `acceleration` come from one polynomial per sample
    and are therefore mutually consistent: differentiating `value` numerically
    would not reproduce `velocity`, and should not be expected to -- `velocity`
    is the better estimate of the two.
    """

    value: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    supported: NDArray[np.bool_]
    observations: NDArray[np.int64]
    residual_rms: float
    window_samples_max: int

    @property
    def unsupported_count(self) -> int:
        return int(np.count_nonzero(~self.supported))


def _windows(
    t: NDArray[np.float64], half: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Half-open index bounds of the time window centred on each sample.

    Bounds are found by binary search on the timestamps rather than by offsetting
    indices, which is what makes the window a fixed span of *time* on input whose
    sample spacing varies.
    """
    lo = np.searchsorted(t, t - half, side="left").astype(np.int64)
    hi = np.searchsorted(t, t + half, side="right").astype(np.int64)
    width = int((hi - lo).max()) if t.size else 0
    return lo, hi, width


def _solve_windows(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    supported: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Least-squares coefficients for every supported window.

    Solved through the normal equations rather than by taking a pseudo-inverse
    of each window. Measured against the SVD by `scripts/benchmark_filter.py`:
    the solve step alone is about 9x faster, the whole fit 3.2-4.0x faster, and
    the two agree to 4e-15 relative on the fitted value. The normal equations
    square the condition number, which is affordable here for a specific reason:
    the local coordinate is scaled into [-1, 1], so a degree-3 Vandermonde over
    reasonably spread nodes has a condition number of order 10, and squaring it
    costs about two of the sixteen digits available.

    That argument depends on the nodes being spread across the window. They stop
    being spread only if a window's observations all cluster into a tiny fraction
    of its span, which needs a burst of frames thousands of times faster than the
    clip's nominal rate followed by a gap -- and `MAX_WINDOW_SAMPLES` catches the
    sample rates where that becomes possible. An exactly singular window is still
    handled rather than assumed away: the whole batch falls back to the SVD.

    Only supported windows are solved. The rest are masked to NaN by the caller,
    so computing them would be work thrown away.
    """
    count, _, order = design.shape
    coefficients = np.zeros((count, order), dtype=np.float64)
    if not np.any(supported):
        return coefficients

    block, rhs_block = design[supported], target[supported]
    transposed = np.swapaxes(block, 1, 2)
    gram = transposed @ block
    rhs = transposed @ rhs_block[..., None]

    try:
        coefficients[supported] = np.linalg.solve(gram, rhs)[..., 0]
    except np.linalg.LinAlgError:
        # A window that is singular despite the observation-count and bracketing
        # checks. The SVD returns a finite minimum-norm answer instead of taking
        # the batch down, and the caller's `supported` mask still governs what is
        # reported -- this only prevents one degenerate window from destroying
        # the fit for an entire clip.
        coefficients[supported] = np.einsum("npk,nk->np", np.linalg.pinv(block), rhs_block)

    return coefficients


def local_polynomial_fit(
    t: NDArray[np.float64],
    value: NDArray[np.float64],
    weight: NDArray[np.float64],
    *,
    window_s: float,
    polyorder: int,
    min_observations: int,
    blocked: NDArray[np.bool_] | None = None,
    residual_mask: NDArray[np.bool_] | None = None,
) -> LocalPolynomialFit:
    """Fit a weighted local polynomial at every sample and read off derivatives.

    `value` may contain NaN, and `weight` may be zero; both mean the sample
    contributes nothing to any fit. `blocked` marks samples at which no value may
    be emitted however well-supported the window is -- the gap policy's veto.
    """
    if polyorder < 0:
        raise LocalPolynomialError(f"polyorder must be non-negative, got {polyorder}.")
    if window_s <= 0:
        raise LocalPolynomialError(f"window_s must be positive, got {window_s}.")
    if min_observations < polyorder + 1:
        raise LocalPolynomialError(
            f"A degree-{polyorder} fit needs at least {polyorder + 1} observations; "
            f"min_observations={min_observations} could not determine one."
        )

    count = t.size
    nan = np.full(count, np.nan, dtype=np.float64)
    if count == 0:
        return LocalPolynomialFit(
            value=nan,
            velocity=nan.copy(),
            acceleration=nan.copy(),
            supported=np.zeros(0, dtype=np.bool_),
            observations=np.zeros(0, dtype=np.int64),
            residual_rms=float("nan"),
            window_samples_max=0,
        )

    half = window_s / 2.0
    lo, hi, width = _windows(t, half)
    if width > MAX_WINDOW_SAMPLES:
        raise LocalPolynomialError(
            f"A {window_s:g} s window covers up to {width} samples on this clip, past the "
            f"{MAX_WINDOW_SAMPLES} sample guard. Either the window is far wider than the "
            "motion it is meant to smooth, or the clip's sample rate is not what it seems."
        )

    # Gather each window into one rectangular block, padded to the widest.
    # Padding slots are marked unusable rather than dropped, so every window is
    # solved by the same vectorised expression regardless of how many real
    # samples it holds.
    offsets = np.arange(width, dtype=np.int64)
    index = lo[:, None] + offsets[None, :]
    in_window = index < hi[:, None]
    index = np.where(in_window, index, 0)

    neighbour_t = t[index]
    neighbour_value = value[index]
    neighbour_weight = weight[index]

    usable = in_window & np.isfinite(neighbour_value) & (neighbour_weight > 0.0)
    observations = usable.sum(axis=1).astype(np.int64)

    # Scaled local coordinate. Fitting in raw seconds would raise a window
    # half-width of ~0.05 s to the polyorder, so the Vandermonde columns would
    # differ by several orders of magnitude and the normal equations would lose
    # precision to conditioning alone. In u the columns are all O(1).
    u = (neighbour_t - t[:, None]) / half

    # Bracketing: at least one observation at or before the evaluation point and
    # one at or after it. Without this the fit extrapolates into an absence and
    # returns a value indistinguishable from a measured one.
    brackets = np.any(usable & (u <= 0.0), axis=1) & np.any(usable & (u >= 0.0), axis=1)

    supported = (observations >= min_observations) & brackets
    if blocked is not None:
        supported &= ~blocked

    sqrt_weight = np.sqrt(np.where(usable, neighbour_weight, 0.0))
    design = (u[..., None] ** np.arange(polyorder + 1)) * sqrt_weight[..., None]
    target = np.where(usable, neighbour_value, 0.0) * sqrt_weight

    coefficients = _solve_windows(design, target, supported)

    fitted = np.where(supported, coefficients[:, 0], np.nan)

    if polyorder >= 1:
        velocity = np.where(supported, coefficients[:, 1] / half, np.nan)
    else:
        # A constant fit carries no slope. Reporting zero would assert the signal
        # is stationary, which is a claim the fit never made.
        velocity = nan.copy()

    if polyorder >= 2:
        acceleration = np.where(supported, 2.0 * coefficients[:, 2] / (half * half), np.nan)
    else:
        acceleration = nan.copy()

    if residual_mask is None:
        residual_mask = np.isfinite(value) & (weight > 0.0)
    scored = residual_mask & supported
    if np.any(scored):
        residual_rms = float(np.sqrt(np.mean((value[scored] - fitted[scored]) ** 2)))
    else:
        residual_rms = float("nan")

    return LocalPolynomialFit(
        value=fitted,
        velocity=velocity,
        acceleration=acceleration,
        supported=supported,
        observations=observations,
        residual_rms=residual_rms,
        window_samples_max=width,
    )
