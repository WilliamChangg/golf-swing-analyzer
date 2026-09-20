"""Finding ball candidates in one frame: ROI, top-hat, components, geometry.

The classical implementation of `BallDetector`, and the only module in this
package that touches OpenCV. Four stages, each of which throws away a different
kind of thing:

    ROI         everything that is not on the ground near the player's feet
    top-hat     everything that is not small and locally distinct
    components  every set of pixels that is not a connected region
    geometry    every region that could not be a golf ball

The first stage is where the anchor arrives, and `analyzer/ball/extract.py` says
why it is the feet rather than the hands. In short: a teed ball rests on the
ground, and "within a club length of the hands" is a much weaker statement that
on a tall frame admits the sky.

## Why a top-hat rather than a threshold or a circle transform

The obvious two tools are both worse here for reasons worth stating, because both
are what a reader would expect to find.

**A brightness threshold encodes one scene.** A golf ball is white, so thresholding
high finds it -- against grass, against a green mat, against a shaded tee box. It
finds nothing against a bright sky, a frost-covered range or a white winter mat,
and it finds the player's shirt, their shoes and the range's yardage boards
everywhere. Phase 10 met this exact problem with Canny's thresholds and solved it
the same way: derive the operator from the region actually being searched rather
than from a constant.

**`HoughCircles` is the shape-matched tool and it is the wrong shape of answer.**
It ranks by accumulator votes, which is a vote count, and Phase 10 spent most of
its argument on why a vote count is the number that goes wrong -- it prefers
whatever is largest and sharpest, and it reports nothing about what it was
choosing between. It is also strikingly sensitive to its gradient threshold on
objects this small, where a ball spans a dozen pixels and its gradient ring is
two of them.

A **morphological top-hat** is the operator that matches the actual statement
being made about a golf ball, which is not "it is bright" and not "it is round"
but *it is small and it differs from what surrounds it*. Opening with a structuring
element larger than a ball erases the ball and leaves the background; subtracting
that from the original leaves exactly the things too small to survive the opening.
The background can be any brightness and the operator does not care.

It is run **both ways**. A white top-hat finds small bright regions and a black
top-hat small dark ones, and both are kept, because the sign of the step is a fact
about the mat rather than about the ball. A detector that only looked upward would
encode one convention about golf and fail silently on the other -- and silently is
the operative word: it would report `NO_CANDIDATE`, which reads as a framing
problem.

## The two responses are thresholded separately

Taking an elementwise maximum of the two responses first is cheaper by one pass
and merges a ball with its own shadow into a single non-round region, which the
circularity filter then rejects. A ball on a sunlit mat has an attached shadow
roughly always, so that is not an edge case. They are thresholded and contoured
apart, and the candidate lists merged afterwards by position.
"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.ball.detector import (
    BallAnchor,
    BallCandidate,
    BallDetectionError,
    BallDetectionResult,
)
from analyzer.contracts.ball import BallConfig, BallDetectorInfo, BallRefusal
from analyzer.ingestion.reader import VideoFrame

DETECTOR_NAME = "contrast_blob"
DETECTOR_METHOD = "roi -> tophat/blackhat -> components -> geometry"

# How much larger than the ball the structuring element is. The opening must
# erase the ball completely or the top-hat leaves only its rim, so this has to
# exceed 1.0 by a real margin; past about 3 it starts erasing things that are
# genuinely background texture and the response picks up mat seams and grass
# clumps. Measured across the sweep in `scripts/benchmark_ball.py`: detection is
# flat from 2.0 to 3.0 and falls away either side.
_STRUCTURING_ELEMENT_BALLS = 2.5

# Radii the interior and the surrounding ring are sampled at, as fractions of the
# candidate's own radius. The interior stops short of the rim because a ball's
# edge is antialiased and, at a dozen pixels across, the rim is a sixth of it; the
# ring starts outside the rim for the same reason and stops before it reaches the
# next object along.
_INTERIOR_FRACTION = 0.70
_RING_INNER_FRACTION = 1.40
_RING_OUTER_FRACTION = 2.20

# The intensity range a contrast step is measured against, as percentiles of the
# searched region rather than its full min-to-max.
#
# Not a robustness flourish: one specular highlight on a club face or a patch of
# blown-out sky puts the maximum at 255 and the minimum at 0 on almost any
# outdoor frame, which makes every contrast score a fraction of 255 and collapses
# the distinction between a ball and a grass clump. The 5th and 95th percentiles
# describe the range the scene actually occupies.
_RANGE_LOW_PERCENTILE = 5.0
_RANGE_HIGH_PERCENTILE = 95.0
_MIN_INTENSITY_RANGE = 8.0

# Turning a median absolute deviation into a standard deviation, for a normal
# distribution. Sensor noise is close enough to normal for this, and the MAD is
# used rather than the deviation itself because the response is *meant* to have a
# few large values in it -- the objects being looked for -- and those would
# inflate an ordinary standard deviation towards the thing it is supposed to
# distinguish them from.
_MAD_TO_SIGMA = 1.4826

# How far clear of that noise the threshold sits. Five is high enough that a
# region of pure noise produces essentially no connected components, and far
# below the tens of levels a real ball's response carries. See `_threshold`.
_NOISE_SIGMAS = 5.0

# Two candidates whose centres sit closer than this many of their own radii are
# the same object found twice -- once in each of the two top-hat responses, which
# both fire on a ball whose rim shades into its own shadow. Only the stronger is
# kept, for the reason `hough.py` merges near-duplicate lines: leaving both in
# would make the runner-up a copy of the winner and the margin check meaningless
# in exactly the frames it exists for.
_DUPLICATE_RADII = 1.0


class ContrastBlobDetector:
    """`BallDetector` by morphological top-hat and connected components.

    Stateless, as the Protocol requires: every call is a pure function of the
    frame, the anchor and the configuration. Two calls on the same frame return
    the same candidates, and the frames of a clip may be visited in any order.
    """

    def __init__(self, config: BallConfig | None = None) -> None:
        self._config = config or BallConfig()
        self._info = BallDetectorInfo(
            name=DETECTOR_NAME,
            method=DETECTOR_METHOD,
            # Measured from the library that is loaded, not copied from the
            # dependency pin. The pin says what should be installed; a result
            # should record what was.
            opencv_version=str(cv2.__version__),
        )

    @property
    def info(self) -> BallDetectorInfo:
        return self._info

    @property
    def config(self) -> BallConfig:
        return self._config

    def detect(self, frame: VideoFrame, anchor: BallAnchor) -> BallDetectionResult:
        """Every candidate ball in one frame, ranked best-first by contrast."""
        image = frame.image
        if image.ndim not in (2, 3):
            raise BallDetectionError(
                f"A ball detector needs a 2D or 3D image array; got shape {image.shape}."
            )

        config = self._config
        region = _region_of_interest(image, anchor, config.search_radius_torso)
        if region is None:
            return BallDetectionResult(
                candidates=(), refusal=BallRefusal.OUT_OF_FRAME, region_pixels=0
            )
        patch, (offset_x, offset_y) = region

        grey = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # A light blur for Canny's reason one package along: the operators below
        # are differences, so without it they respond to per-pixel sensor noise.
        # Three rather than five because a ball twelve pixels across has little
        # shape to spare.
        blurred = cast(NDArray[np.uint8], cv2.GaussianBlur(grey, (3, 3), 0))

        radius_px = config.ball_radius_torso * anchor.torso_px
        if radius_px < config.min_radius_px:
            # The ball is smaller in this framing than a disc can be assessed for
            # roundness at. Reported as the framing fact it is rather than as an
            # empty search, which would read as a contrast problem.
            return BallDetectionResult(
                candidates=(), refusal=BallRefusal.OUT_OF_FRAME, region_pixels=int(blurred.size)
            )

        intensity_range = _intensity_range(blurred)
        candidates = _candidates(blurred, radius_px, intensity_range, config)
        if not candidates:
            return BallDetectionResult(
                candidates=(), refusal=BallRefusal.NO_CANDIDATE, region_pixels=int(blurred.size)
            )

        return BallDetectionResult(
            candidates=tuple(
                BallCandidate(
                    x=candidate.x + offset_x,
                    y=candidate.y + offset_y,
                    radius_px=candidate.radius_px,
                    contrast=candidate.contrast,
                    circularity=candidate.circularity,
                )
                for candidate in candidates
            ),
            refusal=None,
            region_pixels=int(blurred.size),
        )


def _region_of_interest(
    image: NDArray[np.uint8], anchor: BallAnchor, radius_torso: float
) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """The searched patch and where its origin sits in the full frame.

    Square rather than circular for `hough.py`'s reason: every operation
    downstream wants a rectangular array, and masking a disc out of one would
    cost a copy to save nothing. The corners contribute a little extra background
    and cost a little extra work; nothing downstream is misled by them, because
    the ball is identified by agreement across frames rather than by being the
    only candidate.

    Returns None when the anchor is far enough outside the frame that no usable
    patch overlaps it.
    """
    height, width = image.shape[:2]
    radius = radius_torso * anchor.torso_px
    x0 = max(0, math.floor(anchor.x - radius))
    y0 = max(0, math.floor(anchor.y - radius))
    x1 = min(width, math.ceil(anchor.x + radius))
    y1 = min(height, math.ceil(anchor.y + radius))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image[y0:y1, x0:x1], (x0, y0)


def _intensity_range(patch: NDArray[np.uint8]) -> float:
    """The intensity span the scene actually occupies, as a robust range.

    See `_RANGE_LOW_PERCENTILE`: a full min-to-max range is 255 on almost any
    outdoor frame, because one blown highlight and one deep shadow are enough.
    Floored so that a genuinely flat patch cannot divide a contrast score by
    nearly zero and report every speck at 1.0.
    """
    low, high = np.percentile(patch, (_RANGE_LOW_PERCENTILE, _RANGE_HIGH_PERCENTILE))
    return max(_MIN_INTENSITY_RANGE, float(high) - float(low))


def _top_hats(patch: NDArray[np.uint8], radius_px: float) -> tuple[NDArray[np.uint8], ...]:
    """Small bright regions and small dark ones, as two separate responses.

    **A rectangular structuring element, and the reason is cost rather than
    shape.** OpenCV decomposes a rectangular erosion and dilation into two
    separable one-dimensional passes and cannot do that for any other shape, so
    the two are not close: measured on this phase's own fixture, a 59-pixel
    element costs 66 ms per frame as an ellipse and 1.8 ms as a rectangle. That is
    the difference between this phase costing four times what pose estimation
    costs on the same footage and costing a tenth of it.

    An ellipse is the intuitive choice and it buys nothing here. The element's
    only job is to be large enough that the opening erases the ball completely,
    and a square that contains a disc does that exactly as well as the disc does.
    What it also erases is slightly more of the background near the ball, which
    changes the residual's level and not its shape -- and the shape is what the
    circularity filter reads, because it comes from thresholding the residual
    rather than from the element.
    """
    size = max(3, 2 * round(_STRUCTURING_ELEMENT_BALLS * radius_px) + 1)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return (
        cast(NDArray[np.uint8], cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, element)),
        cast(NDArray[np.uint8], cv2.morphologyEx(patch, cv2.MORPH_BLACKHAT, element)),
    )


def _threshold(response: NDArray[np.uint8], intensity_range: float, config: BallConfig) -> float:
    """How strong a top-hat response has to be to be worth contouring.

    **Two floors, and the second is the one that matters.**

    The first is the scene's: the same fraction of the region's own intensity
    range that `min_contrast` demands of a finished candidate. Thresholding lower
    and filtering later would spend the work of contouring every grass clump.

    The second is the response's own noise, and leaving it out is a failure that
    only shows up on the easiest-looking frames. A top-hat response is near zero
    everywhere except at small objects, so on a **flat** region -- a ball on an
    even mat, a tight framing with nothing else in shot -- it is almost entirely
    sensor noise. The scene floor is a fraction of an intensity range that is
    itself only a few levels wide on such a frame, so it lands below the noise,
    the thresholded image becomes a field of speckle, and the largest connected
    regions are noise clumps rather than the ball. Measured: a white ball on a
    flat field was not found at all, while the same ball in a frame that also
    contained a body and a club was found in every frame -- because the body
    widened the range and dragged the threshold up with it.

    So the noise floor is estimated from the response itself, robustly: the
    median absolute deviation scaled to a standard deviation, times a factor that
    puts the threshold clear of it. A real ball's response is the full step
    between it and its surround, which is tens of levels, so this costs nothing
    on a frame where there is anything to find.
    """
    deviation = float(np.median(np.abs(response.astype(np.float32) - np.median(response))))
    noise = _MAD_TO_SIGMA * deviation
    return max(config.min_contrast * intensity_range, _NOISE_SIGMAS * noise)


def _regions(
    response: NDArray[np.uint8],
    threshold: float,
    radius_px: float,
    config: BallConfig,
) -> list[tuple[float, float, float, float]]:
    """Connected regions of a ball's size, as (x, y, radius, circularity).

    The radius is the **equivalent-area** radius, `sqrt(area / pi)`, rather than
    the radius of a minimum enclosing circle. One stray pixel joined to a region
    by a corner inflates an enclosing circle and leaves an area untouched, and at
    a dozen pixels across that is the difference between a ball and a rejected
    candidate.

    Contours are traced without chain approximation. The perimeter is half of the
    circularity score, and the approximated chain shortens a small blob's
    perimeter by enough to turn a mat seam into a disc.

    **The size band is applied before the cap on candidates, not after**, and
    that ordering is not an optimisation. Sorting by area and keeping the largest
    is the obvious way to bound the work and it discards a golf ball: a ball is one
    of the *smallest* things in the response, and on real footage a frame offers
    dozens of larger ones -- grass clumps, compression blocks, the shadow under a
    shoe. Measured on `data/rory/face-on/rory_face_on.mp4`, the ball sat well outside
    the largest 32 regions of its own frame. What remains after the band is then
    ranked by **closeness to a ball's expected size**, so the cap only ever drops
    candidates that are already less ball-like than the ones it keeps.
    """
    _, binary = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    low = max(radius_px * (1.0 - config.radius_tolerance), config.min_radius_px)
    high = radius_px * (1.0 + config.radius_tolerance)

    found: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            continue
        radius = math.sqrt(area / math.pi)
        if not (low <= radius <= high):
            continue
        circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < config.min_circularity:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            continue
        found.append(
            (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
                radius,
                circularity,
            )
        )

    found.sort(key=lambda region: abs(region[2] - radius_px))
    return found[: config.max_candidates]


def _contrast(
    patch: NDArray[np.uint8], x: float, y: float, radius: float, intensity_range: float
) -> float:
    """How far the candidate's interior stands from the ring just outside it.

    The **magnitude** of the step and not its sign, so a white ball on a dark mat
    and a dark ball on a bright one score alike. See the module docstring on why
    the alternative fails silently.

    Sampled with masks over a small window rather than by iterating pixels: a
    clip is a few hundred frames and a frame offers a few dozen candidates, so
    this runs tens of thousands of times per clip.
    """
    outer = _RING_OUTER_FRACTION * radius
    x0 = max(0, math.floor(x - outer))
    y0 = max(0, math.floor(y - outer))
    x1 = min(patch.shape[1], math.ceil(x + outer) + 1)
    y1 = min(patch.shape[0], math.ceil(y + outer) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0

    window = patch[y0:y1, x0:x1].astype(np.float32)
    rows = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    columns = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    distance = np.hypot(rows, columns)

    interior = distance <= _INTERIOR_FRACTION * radius
    ring = (distance >= _RING_INNER_FRACTION * radius) & (distance <= outer)
    if not interior.any() or not ring.any():
        return 0.0

    step = abs(float(window[interior].mean()) - float(window[ring].mean()))
    return float(np.clip(step / intensity_range, 0.0, 1.0))


def _candidates(
    patch: NDArray[np.uint8],
    radius_px: float,
    intensity_range: float,
    config: BallConfig,
) -> list[BallCandidate]:
    """Every region of a ball's size and shape, scored and ranked best-first.

    The size and shape bounds are applied inside `_regions`, in **pixels** derived
    from the torso scale rather than in torso lengths, because that is the unit
    the region was found in and converting a tolerance twice invites the error the
    conversion exists to prevent. Contrast is scored here, after them, because it
    is the expensive one -- it samples two masked neighbourhoods per candidate --
    and there is no reason to spend it on a region that is not a ball's size.
    """
    found: list[BallCandidate] = []
    for response in _top_hats(patch, radius_px):
        threshold = _threshold(response, intensity_range, config)
        for x, y, radius, circularity in _regions(response, threshold, radius_px, config):
            contrast = _contrast(patch, x, y, radius, intensity_range)
            if contrast < config.min_contrast:
                continue
            found.append(
                BallCandidate(
                    x=x, y=y, radius_px=radius, contrast=contrast, circularity=circularity
                )
            )

    found.sort(key=lambda candidate: candidate.contrast, reverse=True)
    return _deduplicate(found)[: config.max_candidates]


def _deduplicate(candidates: list[BallCandidate]) -> list[BallCandidate]:
    """Drop candidates that are the same object as a stronger one already kept.

    The two top-hat responses both fire on a ball whose rim shades into its own
    shadow, and they place it a pixel or so apart. See `_DUPLICATE_RADII`: leaving
    both in would put a copy of the winner in the runner-up slot and drive every
    margin to zero on exactly the clean frames the identification depends on.
    """
    kept: list[BallCandidate] = []
    for candidate in candidates:
        limit = _DUPLICATE_RADII * candidate.radius_px
        if any(math.hypot(candidate.x - other.x, candidate.y - other.y) <= limit for other in kept):
            continue
        kept.append(candidate)
    return kept
