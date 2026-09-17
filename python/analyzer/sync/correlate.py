"""Aligning two clips by the shape of their hand-speed signals.

The estimate that uses no golf knowledge beyond which landmark to follow, and
therefore the one that can check the event-based fit without sharing its
assumptions. It answers one question -- by how much must the target clip be
shifted to make its speed curve line up with the reference's -- and it answers it
with a number and a measure of how much better that number was than its nearest
rival.

## Three things that make a naive correlation wrong here

**Partial overlap.** Two clips rarely cover the same interval, and at extreme
lags only a handful of samples coincide. A correlation computed over those looks
excellent, because three points can be fitted by anything. So every lag is scored
over exactly the samples both clips actually supplied there, and a lag with less
than `min_overlap_s` of common signal is not scored at all rather than scored
optimistically. That is also what stops a zero-padded FFT correlation from
preferring the lag with the most padding.

**Gaps.** A landmark the estimator lost leaves NaN, and a NaN cannot be
interpolated over silently -- Phase 3's whole gap policy exists to stop exactly
that. Resampling onto the common grid therefore carries a validity mask: a grid
point further than one native frame from any real observation is marked absent,
not filled.

**Different units, different shapes.** Speed in one clip is in that clip's frame
widths per second; the two cameras stand at different distances and see different
components of the motion. Pearson correlation removes the scale and the mean,
which handles the units. It does not handle the shapes: a face-on camera sees the
hands sweep across the frame and a down-the-line camera sees much of that sweep
come towards it and vanish. The instants survive -- the hands are slowest at the
top and fastest near impact from any angle -- which is why this works, and
`peak_correlation` is the measurement of how well it worked rather than a promise
that it did.

## The rival, and why it is reported

A swing's speed signal has two humps: the backswing and the much larger
downswing. There is therefore always a second local maximum in the correlation,
at the lag that lays one clip's backswing over the other's downswing. On a clean
pair it is far below the peak. On a pair where one clip is cut off before impact
it may not be, and the difference between the two is the honest measure of
whether the peak means anything. Reporting the rival is what lets a reader see
that the method nearly chose something else.

## Masked correlation by FFT

Every quantity needed for a Pearson correlation at every lag -- the overlap
count, both sums, both sums of squares and the cross term -- is itself a
cross-correlation of masked arrays, so all six come from FFTs in O(N log N)
rather than from a loop over lags in O(N*L). The index convention is pinned by a
test rather than trusted: `_correlate(a, b)[m]` is `sum_i a[i] * b[i + m]`, so a
positive `m` means the target ran ahead of the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate as _scipy_correlate  # type: ignore[import-untyped]

from analyzer.contracts.sync import CorrelationReport, SyncConfig

# Variances computed through an FFT pick up rounding of order the machine
# epsilon times the signal energy; anything at or below this is treated as zero
# rather than square-rooted into a nonsense denominator.
_VARIANCE_EPS = 1e-12


@dataclass(frozen=True)
class SpeedTrack:
    """One clip's hand speed through time, on its own real-seconds clock."""

    t: NDArray[np.float64]
    speed: NDArray[np.float64]
    """NaN wherever the hand had no filtered velocity. Never filled here."""

    @property
    def median_interval_s(self) -> float:
        if self.t.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.t)))


def _lags(length: int) -> NDArray[np.int64]:
    """Every lag `_correlate` returns, in samples, in the order it returns them."""
    return np.arange(-(length - 1), length, dtype=np.int64)


def _correlate(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`out[k] = sum_i a[i] * b[i + lags[k]]`, for every lag with any overlap.

    Both inputs must be the same length, which they are here because both clips
    are resampled onto one shared grid. The reindexing from SciPy's convention is
    the only subtle line in this module and has its own test.
    """
    full: NDArray[np.float64] = _scipy_correlate(b, a, mode="full", method="fft")
    return full


def _resample(
    track: SpeedTrack, grid: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Put one track on the shared grid, marking where it has no support.

    Linear interpolation between real observations, and nothing at all further
    than one native frame from the nearest of them. That tolerance is the clip's
    own sampling interval: beyond it the value would be an extrapolation dressed
    as a sample, and the whole point of the mask is that such a value never
    enters a correlation.
    """
    finite = np.isfinite(track.speed)
    values = np.zeros(grid.size, dtype=np.float64)
    valid = np.zeros(grid.size, dtype=np.bool_)
    if np.count_nonzero(finite) < 2:
        return values, valid

    times, speeds = track.t[finite], track.speed[finite]
    interpolated = np.interp(grid, times, speeds)

    # Distance from each grid point to the nearest observation that survived.
    index = np.searchsorted(times, grid)
    left = times[np.clip(index - 1, 0, times.size - 1)]
    right = times[np.clip(index, 0, times.size - 1)]
    distance = np.minimum(np.abs(grid - left), np.abs(grid - right))

    tolerance = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    supported = distance <= tolerance
    values[supported] = interpolated[supported]
    valid[supported] = True
    return values, valid


def cross_correlate(
    reference: SpeedTrack,
    target: SpeedTrack,
    config: SyncConfig,
) -> CorrelationReport | None:
    """The lag that best aligns two speed signals, or None if none could be scored.

    None is returned rather than a low-confidence answer when the clips share too
    little usable signal for any lag to clear `min_overlap_s`. That is a fact
    about the pair, and it is the caller's business to report it; inventing a
    lag here would put a number where there is no evidence.
    """
    step = config.grid_interval_s
    if step is None:
        intervals = [reference.median_interval_s, target.median_interval_s]
        finite = [value for value in intervals if np.isfinite(value) and value > 0.0]
        if not finite:
            return None
        # The finer of the two. Resampling the coarser clip up adds no
        # information, but it is what lets the lag search resolve below the
        # coarser clip's frame interval, which is where the fast camera in a
        # mismatched pair earns anything at all.
        step = min(finite)

    if reference.t.size < 2 or target.t.size < 2:
        return None

    # One grid, phase-locked to the reference clip, spanning both clips. Sharing
    # the phase is what makes an integer index shift an exact multiple of the
    # step, so a lag index converts to seconds without a residual phase term.
    origin = float(reference.t[0])
    start = min(origin, float(target.t[0]))
    end = max(float(reference.t[-1]), float(target.t[-1]))
    first = int(np.floor((start - origin) / step))
    last = int(np.ceil((end - origin) / step))
    grid = origin + np.arange(first, last + 1, dtype=np.float64) * step
    if grid.size < 2:
        return None

    ref_values, ref_valid = _resample(reference, grid)
    tgt_values, tgt_valid = _resample(target, grid)
    if not np.any(ref_valid) or not np.any(tgt_valid):
        return None

    ref_mask = ref_valid.astype(np.float64)
    tgt_mask = tgt_valid.astype(np.float64)
    ref_masked = ref_values * ref_mask
    tgt_masked = tgt_values * tgt_mask

    counts = np.rint(_correlate(ref_mask, tgt_mask))
    sum_x = _correlate(ref_masked, tgt_mask)
    sum_xx = _correlate(ref_masked**2, tgt_mask)
    sum_y = _correlate(ref_mask, tgt_masked)
    sum_yy = _correlate(ref_mask, tgt_masked**2)
    sum_xy = _correlate(ref_masked, tgt_masked)

    lags = _lags(grid.size)
    minimum_samples = max(2, int(np.ceil(config.min_overlap_s / step)))
    scorable = counts >= minimum_samples
    if config.max_lag_s is not None:
        scorable &= np.abs(lags) * step <= config.max_lag_s
    if not np.any(scorable):
        return None

    safe = np.where(scorable, counts, 1.0)
    var_x = sum_xx - sum_x**2 / safe
    var_y = sum_yy - sum_y**2 / safe
    covariance = sum_xy - sum_x * sum_y / safe

    denominator = np.sqrt(np.clip(var_x, 0.0, None) * np.clip(var_y, 0.0, None))
    # A constant signal on one side has no correlation with anything; it is not
    # a correlation of zero, it is an unanswerable question, so the lag is
    # dropped from scoring rather than scored at zero.
    usable = scorable & (var_x > _VARIANCE_EPS) & (var_y > _VARIANCE_EPS)
    if not np.any(usable):
        return None

    scores = np.full(lags.size, -np.inf, dtype=np.float64)
    scores[usable] = np.clip(covariance[usable] / denominator[usable], -1.0, 1.0)

    peak = int(np.argmax(scores))
    peak_score = float(scores[peak])

    # Parabolic refinement through the peak and its two neighbours, which is
    # what puts the answer below the grid step. Clamped to half a step: a fitted
    # vertex outside that means the three points do not describe a peak, and the
    # saturation is reported rather than hidden.
    shift = 0.0
    if 0 < peak < lags.size - 1 and usable[peak - 1] and usable[peak + 1]:
        before, here, after = scores[peak - 1], scores[peak], scores[peak + 1]
        curvature = before - 2.0 * here + after
        if curvature < 0.0:
            shift = float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))

    separation = max(1, int(np.ceil(config.rival_separation_s / step)))
    far = usable & (np.abs(lags - lags[peak]) >= separation)
    rival_score: float | None = None
    rival_offset: float | None = None
    if np.any(far):
        rival = int(np.argmax(np.where(far, scores, -np.inf)))
        rival_score = float(scores[rival])
        rival_offset = float(lags[rival] * step)

    return CorrelationReport(
        peak_correlation=peak_score,
        peak_offset_s=float((lags[peak] + shift) * step),
        rival_correlation=rival_score,
        rival_offset_s=rival_offset,
        grid_interval_s=step,
        overlap_s=float(counts[peak] * step),
        samples=int(counts[peak]),
        sub_grid_shift_s=float(shift * step),
    )
