"""Bringing two cameras' instants together, where nothing can be held still.

Phase 8 established that two unsynchronised cameras can be stereo-calibrated, and
the mechanism was a multiplication:

    pairing_error_px  =  time_error_s  x  image_speed_px_s

A board held still for a second makes the second factor nearly zero, and with it
the entire cost of not having a genlock. That is why the capture instruction is
"hold the board still" and why it is a measurement rather than folklore.

**A swing has no such instruction.** The hands reach twenty torso lengths per
second; on a 1080p frame filling with a player that is several thousand pixels
per second, so at 30 fps pairing the nearest frames costs a quarter of a frame
interval times that -- tens of pixels of displacement, against the sub-pixel
residuals the stereo calibration was gated on. Nearest-frame pairing is not
merely worse here, it is the dominant error term by two orders of magnitude, and
`scripts/benchmark_reconstruct.py --sweep pairing` measures it.

So the target clip is **resampled onto the reference clip's clock** instead.

## Resampled with the filter's own velocity, not with a fresh interpolation

Phase 3 fitted a local polynomial to each landmark's trajectory and read the
position and the velocity off the same polynomial, so the two are consistent by
construction. Cubic Hermite interpolation between two neighbouring samples uses
exactly those two quantities at both ends and nothing else -- no new kernel, no
new frequency response, no second smoothing pass over data that has already been
smoothed once. It reproduces the underlying local polynomial to third order,
which over a single frame interval is far below the landmark noise.

Linear interpolation was the alternative and is worse in a way that matters
here: it is a chord across an arc, so it cuts the corner at precisely the
instants a swing is most curved -- the top, and the bottom of the downswing --
which are the two instants every metric is anchored to.

## What it refuses

Interpolation happens **between two neighbouring samples that both carry a
value**. A query inside an interval whose endpoints are not both valid produces
nothing, which keeps the gap policy's decision intact: Phase 3 decided how far a
trajectory may be carried across an absence, and this layer is not entitled to
re-open that with a different rule. Queries outside the target clip's own
timestamps produce nothing either, because that is extrapolation.

## And what survives

The map's own uncertainty. Phase 7 reports `TimeMap.uncertainty_at`, and it does
not go away by resampling -- resampling removes the *quantisation*, not the
error in knowing where the two clocks stand relative to each other. That term is
converted into pixels here the same way Phase 8 converted it, by multiplying it
by the landmark's measured image speed, so it can be compared with the
reprojection error instead of being mistaken for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.pose import FrameGeometry, Landmark
from analyzer.filtering.landmarks import FilteredSequence


@dataclass(frozen=True)
class ViewTrack:
    """One camera's filtered landmarks, in pixels, on that camera's own clock.

    Pixels rather than frame widths, because that is the unit a calibration is
    measured in and the unit a projection produces; converting once here keeps
    every reconstruction expression in one system. The conversion is exact and
    its inverse is `coordinates.frame_widths_to_pixels`, so nothing is lost --
    but the y axis flips back to pointing down, which is the camera's convention
    and not the measurement frame's, and that is worth knowing at the boundary.
    """

    t: NDArray[np.float64]
    """Timestamps, in real seconds on this clip's own clock."""
    pixels: NDArray[np.float64]
    """(frames, landmarks, 2), NaN where the filter produced no position."""
    velocity: NDArray[np.float64]
    """(frames, landmarks, 2) in pixels per second, from the filter's own fit."""
    valid: NDArray[np.bool_]
    """(frames, landmarks): where a complete position exists."""
    visibility: NDArray[np.float64]
    """(frames, landmarks), as the estimator reported it."""
    geometry: FrameGeometry
    landmarks: tuple[Landmark, ...]

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def speed_px_s(self) -> NDArray[np.float64]:
        """Image speed per landmark per frame. The factor that turns time into pixels."""
        return np.linalg.norm(self.velocity, axis=2)

    @property
    def median_interval_s(self) -> float:
        if self.t.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.t)))


@dataclass(frozen=True)
class SampledTrack:
    """A view's landmarks evaluated at instants that are not its own frames."""

    pixels: NDArray[np.float64]
    velocity: NDArray[np.float64]
    valid: NDArray[np.bool_]
    visibility: NDArray[np.float64]
    inside: NDArray[np.bool_]
    """Per requested instant: whether it fell inside the clip's timestamps at all.

    Kept separate from `valid` because the two are different failures with
    different fixes. Outside means the clips do not overlap there -- start the
    second camera earlier. Invalid inside the overlap means the landmark was not
    tracked -- an occlusion or a blur."""


def track_from(
    filtered: FilteredSequence, landmarks: tuple[Landmark, ...] | None = None
) -> ViewTrack:
    """Convert a filtered sequence into pixel-space tracks for triangulation.

    The landmarks arrive in FRAME_WIDTHS: isotropic, y upward, origin at the
    bottom-left, and already undistorted if a calibration was supplied. Going
    back to pixels is the affine map `x_px = x * W`, `y_px = H - y * W`, whose
    derivative flips the sign of the vertical velocity and scales both axes by
    the width. Applying it to the velocity as well as the position is what keeps
    the two consistent; a velocity converted with the wrong sign produces a
    reconstruction whose positions are right and whose speeds point the wrong way
    vertically, which nothing downstream would catch.
    """
    selected = landmarks if landmarks is not None else tuple(Landmark)
    geometry = filtered.geometry
    width = float(geometry.width)
    height = float(geometry.height)

    frames = int(filtered.t.size)
    pixels = np.full((frames, len(selected), 2), np.nan, dtype=np.float64)
    velocity = np.full((frames, len(selected), 2), np.nan, dtype=np.float64)
    valid = np.zeros((frames, len(selected)), dtype=np.bool_)
    visibility = np.zeros((frames, len(selected)), dtype=np.float64)

    for column, landmark in enumerate(selected):
        entry = filtered[landmark]
        pixels[:, column, 0] = entry.position[:, 0] * width
        pixels[:, column, 1] = height - entry.position[:, 1] * width
        velocity[:, column, 0] = entry.velocity[:, 0] * width
        velocity[:, column, 1] = -entry.velocity[:, 1] * width
        valid[:, column] = entry.valid
        visibility[:, column] = entry.visibility

    return ViewTrack(
        t=np.asarray(filtered.t, dtype=np.float64),
        pixels=pixels,
        velocity=velocity,
        valid=valid,
        visibility=visibility,
        geometry=geometry,
        landmarks=selected,
    )


def _bracketing(t: NDArray[np.float64], times: NDArray[np.float64]) -> NDArray[np.int64]:
    """Index of the sample at or before each query time, clamped into range."""
    if t.size == 0:
        return np.zeros(times.size, dtype=np.int64)
    left = np.searchsorted(t, times, side="right") - 1
    return np.clip(left, 0, max(t.size - 2, 0)).astype(np.int64)


def resample(track: ViewTrack, times: NDArray[np.float64]) -> SampledTrack:
    """Evaluate a view's landmarks at arbitrary instants, by cubic Hermite.

    The interpolant on `[t0, t1]` with `h = t1 - t0` and `s = (t - t0) / h` is

        p(s) = h00 p0 + h10 h v0 + h01 p1 + h11 h v1

    the standard Hermite basis, which matches position *and* velocity at both
    ends. Its derivative is returned too, and it is the interpolated velocity
    rather than a difference of interpolated positions -- for the same reason
    Phase 3 reads velocity off the polynomial rather than differencing the
    smoothed signal.

    Where either endpoint has no velocity -- a degree-0 filter emits none -- the
    scheme degrades to linear between the two positions rather than refusing.
    That is a real loss of accuracy and it is the honest fallback: a Hermite
    interpolation with an invented velocity would be worse and would not say so.
    """
    times = np.asarray(times, dtype=np.float64)
    count = times.size
    width = len(track.landmarks)

    pixels = np.full((count, width, 2), np.nan, dtype=np.float64)
    velocity = np.full((count, width, 2), np.nan, dtype=np.float64)
    valid = np.zeros((count, width), dtype=np.bool_)
    visibility = np.zeros((count, width), dtype=np.float64)

    if len(track) < 2 or count == 0:
        return SampledTrack(
            pixels=pixels,
            velocity=velocity,
            valid=valid,
            visibility=visibility,
            inside=np.zeros(count, dtype=np.bool_),
        )

    inside = (times >= track.t[0]) & (times <= track.t[-1])
    low = _bracketing(track.t, times)
    high = low + 1

    span = track.t[high] - track.t[low]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(span > 0.0, (times - track.t[low]) / span, 0.0)
    s = np.clip(s, 0.0, 1.0)

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    d00 = 6.0 * s2 - 6.0 * s
    d10 = 3.0 * s2 - 4.0 * s + 1.0
    d01 = -6.0 * s2 + 6.0 * s
    d11 = 3.0 * s2 - 2.0 * s

    both = track.valid[low] & track.valid[high] & inside[:, None] & (span > 0.0)[:, None]

    p0, p1 = track.pixels[low], track.pixels[high]
    v0, v1 = track.velocity[low], track.velocity[high]
    have_velocity = np.isfinite(v0).all(axis=2) & np.isfinite(v1).all(axis=2)

    h = span[:, None, None]
    cubic = (
        h00[:, None, None] * p0
        + h10[:, None, None] * h * v0
        + h01[:, None, None] * p1
        + h11[:, None, None] * h * v1
    )
    cubic_rate = (
        d00[:, None, None] * p0
        + d10[:, None, None] * h * v0
        + d01[:, None, None] * p1
        + d11[:, None, None] * h * v1
    ) / np.where(h > 0.0, h, np.nan)

    linear = (1.0 - s)[:, None, None] * p0 + s[:, None, None] * p1
    linear_rate = (p1 - p0) / np.where(h > 0.0, h, np.nan)

    chosen = np.where(have_velocity[..., None], cubic, linear)
    chosen_rate = np.where(have_velocity[..., None], cubic_rate, linear_rate)

    pixels[both] = chosen[both]
    velocity[both] = chosen_rate[both]
    # Visibility is interpolated linearly. It is a confidence rather than a
    # trajectory -- it has no fitted derivative, and a cubic through two
    # confidences can overshoot outside [0, 1], which would be a confidence the
    # estimator never reported.
    blended = (1.0 - s)[:, None] * track.visibility[low] + s[:, None] * track.visibility[high]
    visibility[both] = blended[both]
    valid |= both

    return SampledTrack(
        pixels=pixels,
        velocity=velocity,
        valid=valid,
        visibility=visibility,
        inside=inside,
    )


def nearest(track: ViewTrack, times: NDArray[np.float64]) -> SampledTrack:
    """Take each instant's closest frame outright, as Phase 8 pairs board views.

    Kept because it is the comparison, not because it is an option worth
    choosing. `scripts/benchmark_reconstruct.py --sweep pairing` runs both and
    reports the difference, which is what turned "resample rather than pair" from
    a preference into a measurement -- and what makes Phase 8's "hold the board
    still" instruction legible as the special case it is, rather than as general
    advice that Phase 9 quietly stopped following.
    """
    times = np.asarray(times, dtype=np.float64)
    count = times.size
    width = len(track.landmarks)

    pixels = np.full((count, width, 2), np.nan, dtype=np.float64)
    velocity = np.full((count, width, 2), np.nan, dtype=np.float64)
    valid = np.zeros((count, width), dtype=np.bool_)
    visibility = np.zeros((count, width), dtype=np.float64)

    if len(track) == 0 or count == 0:
        return SampledTrack(
            pixels=pixels,
            velocity=velocity,
            valid=valid,
            visibility=visibility,
            inside=np.zeros(count, dtype=np.bool_),
        )

    inside = (times >= track.t[0]) & (times <= track.t[-1])
    after = np.searchsorted(track.t, times, side="left")
    before = np.clip(after - 1, 0, track.t.size - 1)
    after = np.clip(after, 0, track.t.size - 1)
    closer = np.abs(track.t[after] - times) < np.abs(times - track.t[before])
    index = np.where(closer, after, before)

    picked = track.valid[index] & inside[:, None]
    pixels[picked] = track.pixels[index][picked]
    velocity[picked] = track.velocity[index][picked]
    visibility[picked] = track.visibility[index][picked]
    valid |= picked

    return SampledTrack(
        pixels=pixels,
        velocity=velocity,
        valid=valid,
        visibility=visibility,
        inside=inside,
    )
