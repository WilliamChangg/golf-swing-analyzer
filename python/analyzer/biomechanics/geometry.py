"""Vector and angle primitives for the biomechanics layer.

Pure geometry: everything here takes points and returns lengths and angles, and
nothing knows what a swing is or which frame the points arrived in.

That last part is new in Phase 6. This module used to own the conversion out of
IMAGE coordinates as well -- the aspect correction and the y flip -- which made
it the one place those could be applied and also the place a metric module had
to remember to go through. Both now happen once below the filter, in
`analyzer/coordinates.py`, so every layer above receives FRAME_WIDTHS: isotropic,
upward-positive, and the same frame the phase detector reads. The functions
below simply assume that, which is why none of them takes a `FrameGeometry` any
more.

The z channel never reaches here. In a single-camera recording it is MediaPipe's
own depth guess on an unstated scale, and a distance that included it would mix a
measured quantity with a guessed one inside one number with no way to tell
afterwards which part was which. Real depth arrives in Phase 9.

## Conventions

Coordinates are `(..., 2)` arrays, so every function takes either one point or a
whole trajectory. Angles are degrees. NaN propagates: a frame with no landmark
produces a NaN angle rather than a plausible one, which is what lets the layer
above refuse to emit a metric instead of emitting a guess.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lengths(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Euclidean length along the last axis."""
    return np.linalg.norm(vectors, axis=-1)


def midpoint(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """The point halfway between two points."""
    return (a + b) / 2.0


def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Straight-line distance between two points."""
    return lengths(b - a)


def _safe_unit(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vectors, NaN where the input has no direction.

    A zero-length vector has no direction, and the honest answer for its angle
    is "undefined". Returning NaN says that; returning zero would quietly make
    the angle come out as whatever the arithmetic happened to produce.
    """
    magnitudes = lengths(vectors)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / magnitudes[..., np.newaxis]
    return np.where((magnitudes > 0.0)[..., np.newaxis], unit, np.nan)


def angle_between_deg(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unsigned angle between two vectors, in degrees, on [0, 180].

    Computed through the dot product of unit vectors, clipped before the arccos:
    rounding can put the cosine a few ulps outside [-1, 1] for parallel vectors,
    and `arccos` of 1.0000000000000002 is NaN rather than zero.
    """
    cosine = np.sum(_safe_unit(u) * _safe_unit(v), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def interior_angle_deg(
    first: NDArray[np.float64], vertex: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The angle at `vertex` in the path first -> vertex -> second, in degrees.

    180 degrees is a straight line through the vertex. This is the form every
    joint angle takes: knee flexion is the angle at the knee between the hip and
    the ankle, elbow extension the angle at the elbow between shoulder and wrist.
    """
    return angle_between_deg(first - vertex, second - vertex)


def tilt_from_vertical_deg(
    lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Lean of the segment `lower` -> `upper` away from straight up, in degrees.

    Zero is vertical. **Signed**: positive when the upper end leans towards +x,
    which is towards the right of the frame as displayed. Ranges over
    (-180, 180], though anything approaching the extremes means the segment is
    upside down and the landmarks are more likely swapped than the subject
    inverted.
    """
    delta = upper - lower
    return np.degrees(np.arctan2(delta[..., 0], delta[..., 1]))


def line_tilt_deg(start: NDArray[np.float64], end: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tilt of the line through two points away from level, in degrees, on [-90, 90].

    A shoulder line has an **orientation, not a direction**, so the answer must
    not depend on which of the two points was passed first. It is made canonical
    by measuring left to right *across the frame*: whichever point has the
    smaller x is treated as the start. **Positive therefore means the endpoint
    further right in the frame is the higher one**, and the caller's argument
    order does not enter the result.

    Two wrong ways to do this are worth naming, because each looks right.

    Taking the plain vector angle leaves the signal beside the +-180
    discontinuity whenever the line is near level, so it jumps the full 360 every
    time the tilt crosses zero -- through a swing, most of the clip.

    Folding that angle onto (-90, 90] fixes the jump and silently redefines the
    sign. Folding reverses any vector pointing left, which swaps which endpoint
    the sign refers to; a caller passing (left_shoulder, right_shoulder) and
    reading "positive means the right shoulder is higher" is then correct only
    while the player's right shoulder happens to be on the right of the frame.
    It is not: a player facing the camera has their right on the frame's left, so
    the documented meaning inverts for exactly the footage this system is aimed
    at. Ordering the points first makes the orientation explicit instead of
    arriving at it by a modulo.

    An anatomical reading -- lead shoulder down, trail shoulder up -- needs to
    know which side of the body is which, and belongs with the view tagging in
    Phase 6 rather than hidden in a sign here.
    """
    delta = end - start
    # Point the segment rightwards across the frame, so the tilt describes the
    # line rather than the order the endpoints arrived in.
    rightwards = np.where((delta[..., 0] < 0.0)[..., np.newaxis], -delta, delta)
    return np.degrees(np.arctan2(rightwards[..., 1], rightwards[..., 0]))


def foreshortening_angle_deg(
    span: NDArray[np.float64] | float, reference: float
) -> NDArray[np.float64]:
    """Rotation away from the camera implied by a shortened segment, in degrees.

    A rigid segment of length L seen rotated by theta about an axis in the image
    plane projects to L*cos(theta). Inverting that gives theta from two lengths
    and no camera parameters at all, which is why it is available here and a real
    3D angle is not.

    Three things it cannot do, all of which the caller must handle rather than
    forget:

    * **It is blind to direction.** cos is even, so a turn one way and its mirror
      project identically. The result is a magnitude on [0, 90].
    * **It is only as good as `reference`.** That length is the longest the
      segment was ever *seen*, which is a lower bound on its true length -- if
      the subject was never square to the camera, every angle here is
      under-reported.
    * **It is ill-conditioned near zero.** d(theta)/d(span) goes as 1/sin(theta),
      so a small rotation is swamped by landmark noise while a large one is
      measured sharply. `sin(theta)` is exactly the conditioning factor, which is
      what the confidence model uses.

    A span longer than the reference clips to zero rotation rather than
    producing NaN: it means noise pushed one frame past the maximum, not that the
    segment grew.
    """
    if not np.isfinite(reference) or reference <= 0.0:
        return np.full(np.shape(span), np.nan, dtype=np.float64)
    ratio = np.clip(np.asarray(span, dtype=np.float64) / reference, 0.0, 1.0)
    return np.degrees(np.arccos(ratio))


def project_onto(
    vector: NDArray[np.float64], direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Signed length of `vector` along `direction`.

    Positive when the two point the same way. Used to ask which side of the body
    a point sits on: project the hand's offset from the chest onto the shoulder
    line and the sign names the shoulder.
    """
    return np.sum(vector * _safe_unit(direction), axis=-1)
