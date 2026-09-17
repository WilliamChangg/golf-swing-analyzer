"""Reference frames, and the conversions between the ones that exist.

Sits below everything. `ingestion` decodes pixels, `pose` produces landmarks in
one frame, and every layer above measures in another; this module owns the
translation and the conventions, so that no two callers can disagree about which
way y points or what a distance of 0.3 means.

## The frames

    IMAGE          x/W, y/H          y down   [0,1]x[0,1]   anisotropic  stored
    FRAME_WIDTHS   x/W, (H-y_px)/W   y up     [0,1]x[0,a]   isotropic    derived
    HIP_LOCAL      approximate metres, hip-centred, body-oriented          stored
    CAMERA         metres, camera-centred                       Phase 8, absent
    WORLD          metres, scene-fixed                          Phase 9, absent

`a` is the aspect ratio, height over width.

## Why FRAME_WIDTHS exists

IMAGE is what a pose estimator emits and the only frame a landmark can be drawn
in without conversion, but it is a bad frame to measure in, for two reasons that
both fail silently.

**It is anisotropic.** x is divided by the frame width and y by the frame
height, so on a 1080x1920 clip the same displacement in pixels comes out 0.5625
times as large going down as going across. A distance mixing the two is wrong, an
angle taken from them is wrong by a factor that depends only on the shape of the
frame -- a true 45 degrees reads as 29.4 -- and nothing about either number looks
wrong.

**Its y points down.** Every statement a swing analysis makes is about something
being high or low, and a sign error inverts the top of the backswing into the
bottom without failing anywhere.

FRAME_WIDTHS fixes both, and is deliberately fixed **once, here, below the
filter**. Doing it above means every consumer repeats the correction, and the
one that forgets produces a plausible number. Doing it below also means the
derivatives come out right for free: the filter is linear, so a sign flip
applied to positions before fitting emerges correctly signed in the velocity and
the acceleration, rather than needing a second correction that has to be kept in
step with the first.

## What FRAME_WIDTHS is not

It is **not metric**. It is a picture measured in units of its own width, and
1.0 is "the width of the frame", not a metre. Two clips of the same swing from
different distances give different numbers in it. Metric scale needs a
calibrated camera, and that is why `CAMERA` and `WORLD` are named above but
cannot be produced: naming them costs nothing and keeps a later phase from
inventing the concept alongside the capability.

Measurements that must survive a change of camera distance divide by something
the subject brings with them -- the biomechanics layer uses the torso -- rather
than by a frame width.

## The z channel

Carried, converted and not to be trusted. MediaPipe's IMAGE z is a depth
estimate from a single camera, documented by its authors as roughly the same
scale as x -- so it is already in frame widths and passes through unchanged.
That makes the conversion coherent rather than requiring a two-dimensional
special case, and it does not make the number a measurement. Nothing above this
module reads z for any reported quantity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.pose import (
    UNREACHABLE_SPACES,
    FrameGeometry,
    LandmarkSpace,
)


class CoordinateError(ValueError):
    """A conversion was asked for that this build cannot perform."""


def require_reachable(space: LandmarkSpace) -> None:
    """Raise if `space` is a frame this build cannot produce.

    Named separately so the error says which phase supplies the frame, rather
    than reporting an unsupported value and leaving a reader to work out whether
    it is a typo or a capability that does not exist yet.
    """
    reason = UNREACHABLE_SPACES.get(space)
    if reason is not None:
        raise CoordinateError(
            f"Landmarks in {space.value} coordinates cannot be produced: {reason}."
        )


def image_to_frame_widths(
    points: NDArray[np.float64], geometry: FrameGeometry
) -> NDArray[np.float64]:
    """Normalised IMAGE coordinates to isotropic, upward-positive frame widths.

    Takes `(..., 3)` and returns `(..., 3)`. Both corrections happen here:

        x' = x
        y' = (1 - y) * aspect
        z' = z

    The y expression is a flip and a scale together. `1 - y` turns a downward
    coordinate into a height above the bottom of the frame; multiplying by the
    aspect ratio puts that height in the same unit as x. Doing them in the other
    order, or only one of them, both produce something that still looks like a
    coordinate.
    """
    aspect = geometry.aspect_ratio
    converted = np.asarray(points, dtype=np.float64).copy()
    converted[..., 1] = (1.0 - converted[..., 1]) * aspect
    return converted


def frame_widths_to_image(
    points: NDArray[np.float64], geometry: FrameGeometry
) -> NDArray[np.float64]:
    """The exact inverse of `image_to_frame_widths`."""
    aspect = geometry.aspect_ratio
    converted = np.asarray(points, dtype=np.float64).copy()
    converted[..., 1] = 1.0 - converted[..., 1] / aspect
    return converted


def frame_widths_to_pixels(
    points: NDArray[np.float64], geometry: FrameGeometry
) -> NDArray[np.float64]:
    """Frame widths to displayed pixel coordinates, origin top-left, y down.

    What an overlay needs. Both axes scale by the **width**, which is the whole
    point of the frame: one number converts it, and a skeleton drawn this way
    landing on the person is the visible confirmation that the aspect handling
    upstream is right.
    """
    scale = float(geometry.width)
    converted = np.asarray(points, dtype=np.float64)
    x = converted[..., 0] * scale
    y = float(geometry.height) - converted[..., 1] * scale
    return np.stack((x, y), axis=-1)


def image_to_pixels(points: NDArray[np.float64], geometry: FrameGeometry) -> NDArray[np.float64]:
    """Normalised IMAGE coordinates to displayed pixels, origin top-left."""
    converted = np.asarray(points, dtype=np.float64)
    return np.stack(
        (converted[..., 0] * geometry.width, converted[..., 1] * geometry.height), axis=-1
    )


def convert(
    points: NDArray[np.float64],
    source: LandmarkSpace,
    target: LandmarkSpace,
    geometry: FrameGeometry,
) -> NDArray[np.float64]:
    """Convert landmarks between frames, refusing anything not defined.

    Only the conversions that are actually meaningful exist. HIP_LOCAL is not
    reachable from IMAGE in either direction: it is hip-centred and
    body-oriented, and recovering it from a picture would mean recovering the
    depth that was lost when the picture was taken. That is what Phase 9 is for,
    and until then the refusal is the correct answer rather than a gap.
    """
    if source is target:
        return np.asarray(points, dtype=np.float64).copy()

    require_reachable(source)
    require_reachable(target)

    if source is LandmarkSpace.IMAGE and target is LandmarkSpace.FRAME_WIDTHS:
        return image_to_frame_widths(points, geometry)
    if source is LandmarkSpace.FRAME_WIDTHS and target is LandmarkSpace.IMAGE:
        return frame_widths_to_image(points, geometry)

    raise CoordinateError(
        f"No conversion from {source.value} to {target.value}. "
        "HIP_LOCAL is hip-centred and body-oriented, so reaching it from a picture "
        "would mean recovering the depth the picture lost; that needs the "
        "triangulation Phase 9 provides."
    )
