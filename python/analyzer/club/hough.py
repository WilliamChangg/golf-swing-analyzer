"""Finding shaft candidates in one frame: ROI, Canny, probabilistic Hough, geometry.

The classical implementation of `ClubDetector`, and the only module in this
package that touches OpenCV. Four stages, each of which throws away a different
kind of thing:

    ROI       everything that is not near the hands
    Canny     everything that is not an edge
    HoughP    every set of edge pixels that is not a line
    geometry  every line that could not be a club somebody is holding

The order is not arbitrary and the first stage is not an optimisation. Restricting
the search to a disc around the hands is what makes the last stage's job
tractable: a Hough transform over a whole frame of a driving range returns the
horizon, the mat edges, the netting and the bay dividers, and a filter asked to
pick a club out of that is being asked the wrong question. The one thing known
for certain about a golf shaft is that somebody is holding it, and the cheapest
place to use that fact is before any transform runs.

## The thresholds are derived from the frame, not fixed

Canny's two thresholds are the single most common place a classical vision
pipeline encodes one lighting condition and calls it a constant. A pair that
finds a dark shaft against grass finds nothing against a bright sky, and the
failure is silent -- an empty edge map produces no lines, which arrives at the
tracker as "no candidate" and reads like a blurred club.

They are taken from the median intensity of the region actually being searched,
plus and minus a fraction of it (`ClubConfig.canny_sigma`). That is a standard
construction and it is used here for a specific reason rather than because it is
standard: the region is a disc around the hands, so its median is the local
exposure where the club is, which is what the thresholds should track. A median
over the whole frame would be pulled by the sky.

## What `support` measures, and why it is not the Hough score

`HoughLinesP` scores a line by how many edge pixels voted for it, and returns a
segment. Both are properties of the line the transform found. Neither says
anything about the stretch between that segment and the hands.

So support is measured over the ray **from the anchor to the segment's far
end** -- the thing that is actually being claimed -- as the fraction of sampled
points along it that land on an edge pixel. A segment collinear with the club but
beginning a hand's width past the grip is penalised for the empty stretch in
between, and a segment that genuinely runs from the hands outward is not. That
makes "the shaft reaches the hands" a measured quantity instead of a second
threshold, and it is what the tracker's `support` factor reports.

It also has the property this phase needs most: it degrades **gradually** as the
club head smears. The far part of the ray stops landing on edges before the near
part does, so a partly-blurred shaft reports a support of 0.6 rather than
vanishing, and the length at which the evidence stopped is recoverable. A
vote-count score does neither.
"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.club.detector import (
    ClubDetectionError,
    DetectionResult,
    GripAnchor,
    ShaftCandidate,
)
from analyzer.club.geometry import angle_deg, angle_difference_deg, point_segment_distance_px
from analyzer.contracts.club import ClubConfig, ClubDetectorInfo, ShaftRefusal
from analyzer.ingestion.reader import VideoFrame

DETECTOR_NAME = "hough_shaft"
DETECTOR_METHOD = "roi -> canny -> hough_p -> geometry"

# Two candidates whose directions differ by less than this are the same object
# seen twice, and only the better of them is kept.
#
# This is not a tuning knob, it is what makes the margin check mean anything. A
# probabilistic Hough transform routinely returns three or four collinear
# fragments of one edge; leaving them in would make the runner-up a copy of the
# winner, the margin zero, and every frame ambiguous -- in precisely the frames
# where the club is well seen. Five degrees is comfortably above the spread of
# genuine duplicates (one to two degrees, measured) and far below the separation
# of two real objects in a swing frame.
_DUPLICATE_ANGLE_DEG = 5.0

# How finely the ray is sampled when measuring support: one sample per pixel of
# its length. Sampling coarser would let a ray step over the gaps that are the
# whole signal here; finer measures the same edge pixel twice.
_SUPPORT_SAMPLES_PER_PX = 1.0

# Half-width of the band around the ray that support is measured in, as a
# fraction of the torso length.
#
# This is not a tolerance, it is a correction for a geometric fact that is easy
# to get wrong: **a bar has no edge down its middle.** Canny finds the two sides
# of a shaft, and the ray from the grip to the tip runs between them, so a
# support score sampled on the ray alone measures the gap rather than the club.
# Measured on the synthetic fixture, a clean shaft scores 0.005 that way and
# 1.000 once the band covers its own thickness.
#
# It scales with the torso because the shaft's apparent thickness does: both are
# the image scale of the same scene. Two per cent of a torso is four pixels on
# the fixture's 200-pixel torso, which covers a three-pixel antialiased shaft
# with a pixel of rasterisation slop either side. It does not blunt the
# measurement -- a shaft smeared over 35 px still scores 0.27 against a clean
# one's 1.00 -- because a smear is a fan with nothing along most of it, not a
# thicker line.
_SUPPORT_BAND_TORSO = 0.02
_MIN_SUPPORT_BAND_PX = 2

# Below this many edge pixels the region has nothing to run a line search on, and
# the honest answer is "no edges" rather than "no candidate" -- they are a
# lighting problem and a blur problem respectively. Scaled by the region's area
# so it means the same on any framing.
_MIN_EDGE_FRACTION = 0.0005

# The shortest *fragment* the transform is asked for, as a fraction of the
# shortest shaft the geometry will accept.
#
# These are two different lengths and setting them equal is a mistake that costs
# the club. `ClubConfig.min_shaft_length_torso` is how far a shaft must reach
# **from the hands**, which is what the geometry filter measures. The transform's
# `minLineLength` is how long one unbroken run of collinear edge pixels must be,
# and an antialiased diagonal does not produce one long run: measured on the
# synthetic fixture, a clean 275-pixel shaft comes back as four fragments of 100
# to 200 pixels, every one of which a 200-pixel minimum discards. The frame then
# reports no candidate while the club sits in plain view with 779 edge pixels on
# it.
#
# So the transform is asked for fragments and the geometry is asked about
# shafts. A quarter is low enough to survive that fragmentation and high enough
# that a fragment still means something; what stops the short ones becoming
# candidates is the length filter, which measures the right thing.
_MIN_FRAGMENT_FRACTION = 0.25


class HoughShaftDetector:
    """`ClubDetector` by edge detection and a probabilistic Hough transform.

    Stateless, as the Protocol requires: every call is a pure function of the
    frame, the anchor and the configuration. Two calls on the same frame return
    the same candidates, and the frames of a clip may be visited in any order --
    which is what lets the tracker start where the evidence is best.
    """

    def __init__(self, config: ClubConfig | None = None) -> None:
        self._config = config or ClubConfig()
        self._info = ClubDetectorInfo(
            name=DETECTOR_NAME,
            method=DETECTOR_METHOD,
            # Measured from the library that is loaded, not copied from the
            # dependency pin. The pin says what should be installed; a result
            # should record what was.
            opencv_version=str(cv2.__version__),
        )

    @property
    def info(self) -> ClubDetectorInfo:
        return self._info

    @property
    def config(self) -> ClubConfig:
        return self._config

    def detect(self, frame: VideoFrame, anchor: GripAnchor) -> DetectionResult:
        """Every candidate shaft in one frame, ranked best-first by support."""
        image = frame.image
        if image.ndim not in (2, 3):
            raise ClubDetectionError(
                f"A club detector needs a 2D or 3D image array; got shape {image.shape}."
            )

        config = self._config
        region = _region_of_interest(image, anchor, config.roi_radius_torso)
        if region is None:
            return DetectionResult(
                candidates=(), refusal=ShaftRefusal.NO_EDGES, edge_pixels=0, lines_found=0
            )
        patch, (offset_x, offset_y) = region

        edges = _edge_map(patch, config.canny_sigma)
        edge_pixels = int(np.count_nonzero(edges))
        if edge_pixels < max(1, int(_MIN_EDGE_FRACTION * edges.size)):
            return DetectionResult(
                candidates=(),
                refusal=ShaftRefusal.NO_EDGES,
                edge_pixels=edge_pixels,
                lines_found=0,
            )

        segments = _hough_segments(edges, anchor.torso_px, config)
        if not segments:
            return DetectionResult(
                candidates=(),
                refusal=ShaftRefusal.NO_CANDIDATE,
                edge_pixels=edge_pixels,
                lines_found=0,
            )

        # Support is measured in a band around the ray, for the reason in
        # `_SUPPORT_BAND_TORSO`, and widening the edges once here rather than
        # sampling perpendicular offsets per candidate is the difference between
        # one morphology call and sixty-four ray scans of several samples each.
        band = max(_MIN_SUPPORT_BAND_PX, round(_SUPPORT_BAND_TORSO * anchor.torso_px))
        dilated = cast(
            NDArray[np.uint8],
            cv2.dilate(edges, np.ones((2 * band + 1, 2 * band + 1), np.uint8)),
        )

        candidates = _shaft_candidates(segments, dilated, anchor, (offset_x, offset_y), config)
        if not candidates:
            return DetectionResult(
                candidates=(),
                refusal=ShaftRefusal.NO_CANDIDATE,
                edge_pixels=edge_pixels,
                lines_found=len(segments),
            )

        return DetectionResult(
            candidates=candidates,
            refusal=None,
            edge_pixels=edge_pixels,
            lines_found=len(segments),
        )


def _region_of_interest(
    image: NDArray[np.uint8], anchor: GripAnchor, radius_torso: float
) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """The searched patch and where its origin sits in the full frame.

    Square rather than circular, because every operation downstream of it wants a
    rectangular array and masking a disc out of one would cost a copy to save
    nothing. The corners contribute a little extra background, which the geometry
    filter removes on distance anyway.

    Returns None when the anchor is far enough outside the frame that no patch
    overlaps it, which happens when the hands leave the picture.
    """
    height, width = image.shape[:2]
    radius = radius_torso * anchor.torso_px
    left = math.floor(anchor.x - radius)
    top = math.floor(anchor.y - radius)
    right = math.ceil(anchor.x + radius)
    bottom = math.ceil(anchor.y + radius)

    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, right), min(height, bottom)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return image[y0:y1, x0:x1], (x0, y0)


def _edge_map(patch: NDArray[np.uint8], sigma: float) -> NDArray[np.uint8]:
    """Canny edges, with thresholds taken from this patch's own median intensity.

    The light blur before Canny is three pixels rather than the conventional
    five. Canny differentiates, so some smoothing is needed or it responds to
    sensor noise; but the signal being looked for here is a shaft already
    degraded by motion, and a wider kernel takes the last of its gradient with
    it. Three is the smallest that suppresses per-pixel noise.
    """
    grey = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    median = float(np.median(blurred))
    low = int(max(0.0, (1.0 - sigma) * median))
    high = int(min(255.0, (1.0 + sigma) * median))
    if high <= low:
        # A patch with a median at either end of the range -- a silhouette against
        # the sky, or a shadowed interior -- collapses the band. Falling back to a
        # fixed pair here is not a lighting assumption: it is the only remaining
        # way to have a band at all, and the support score still judges whatever
        # it produces.
        low, high = 50, 150
    # Cast for the same reason `ingestion.reader` casts its rotations: OpenCV's
    # stubs return a loosely-typed array, and the dtype is the one thing about
    # an edge map that is never in doubt.
    return cast(NDArray[np.uint8], cv2.Canny(blurred, low, high, L2gradient=True))


def _hough_segments(
    edges: NDArray[np.uint8], torso_px: float, config: ClubConfig
) -> list[tuple[int, int, int, int]]:
    """Line segments in the edge map, in patch coordinates.

    A **fragment** finder, not a shaft finder: see `_MIN_FRAGMENT_FRACTION` for
    why the minimum asked of it is a quarter of the minimum shaft length rather
    than equal to it. Deciding which fragments could be a club is
    `_shaft_candidates`'s job, and it measures from the hands, which is the thing
    the bound is actually about.

    The angular resolution is half a degree. A shaft is a long lever, so half a
    degree at the grip is a couple of pixels at the head, which is below what the
    edge map itself resolves; going finer subdivides the accumulator without
    adding evidence and costs linearly.
    """
    min_length = max(4.0, _MIN_FRAGMENT_FRACTION * config.min_shaft_length_torso * torso_px)
    max_gap = max(1.0, config.max_line_gap_torso * torso_px)
    found = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 360.0,
        threshold=config.hough_threshold,
        minLineLength=round(min_length),
        maxLineGap=round(max_gap),
    )
    if found is None:
        return []
    return [tuple(int(v) for v in row[0]) for row in found]  # type: ignore[misc]


def _support_of(
    start: tuple[float, float], end: tuple[float, float], edges: NDArray[np.uint8]
) -> float:
    """Fraction of the ray from `start` to `end` that lands on an edge pixel.

    Both points are in patch coordinates. Samples outside the patch count as
    unsupported rather than being skipped: a ray that leaves the searched region
    is a ray whose far half nothing looked at, and skipping those samples would
    score it on the half that was convenient.
    """
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    samples = max(2, round(length * _SUPPORT_SAMPLES_PER_PX))
    ts = np.linspace(0.0, 1.0, samples)
    xs = np.rint(start[0] + ts * (end[0] - start[0])).astype(np.int64)
    ys = np.rint(start[1] + ts * (end[1] - start[1])).astype(np.int64)

    height, width = edges.shape[:2]
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(inside):
        return 0.0
    hits = np.zeros(samples, dtype=bool)
    hits[inside] = edges[ys[inside], xs[inside]] > 0
    return float(np.count_nonzero(hits) / samples)


def _shaft_candidates(
    segments: list[tuple[int, int, int, int]],
    edges: NDArray[np.uint8],
    anchor: GripAnchor,
    offset: tuple[int, int],
    config: ClubConfig,
) -> tuple[ShaftCandidate, ...]:
    """Apply the geometry, score what survives, and merge the duplicates.

    Every bound is stated in torso lengths and multiplied by the anchor's scale
    here, so the same configuration means the same physical tolerance whether the
    subject fills the frame or a tenth of it.
    """
    offset_x, offset_y = offset
    local_anchor = (anchor.x - offset_x, anchor.y - offset_y)
    max_grip = config.max_grip_distance_torso * anchor.torso_px
    min_length = config.min_shaft_length_torso * anchor.torso_px
    max_length = config.max_shaft_length_torso * anchor.torso_px

    scored: list[ShaftCandidate] = []
    for x0, y0, x1, y1 in segments:
        endpoints = ((float(x0), float(y0)), (float(x1), float(y1)))
        # The segment must come close to the hands *somewhere along itself*, not
        # merely lie on a line that passes near them: a fence post behind the
        # player is collinear with the grip in plenty of frames and is nowhere
        # near it. See `point_segment_distance_px`.
        if point_segment_distance_px(local_anchor, *endpoints) > max_grip:
            continue

        # **Both** ends are offered as tips, not just the far one, and that is
        # the fix for a failure this detector had and could not report.
        #
        # A club has a grip end: the edges stop at the hands. A door frame does
        # not -- it runs past them, and the geometry then admits two readings of
        # the same pixels that differ by 180 degrees. Taking the farther endpoint
        # silently picks one, and measured on the synthetic swing it picks wrong
        # for runs of several frames and reports it at a confidence of 1.00, with
        # no runner-up and nothing in the output to suggest a problem.
        #
        # Offering both puts the two readings against each other. With no
        # temporal information their supports are equal, the margin collapses and
        # the frame is refused as `AMBIGUOUS` -- which is the honest answer,
        # because the image on its own does not say which way the club points.
        # With a tracked neighbour the predicted direction separates them, and
        # the frame resolves correctly. It is the clearest case in this phase of
        # the tracker earning what the detector cannot supply.
        for tip in endpoints:
            length = math.hypot(tip[0] - local_anchor[0], tip[1] - local_anchor[1])
            if not (min_length <= length <= max_length):
                continue
            scored.append(
                ShaftCandidate(
                    tip_x=tip[0] + offset_x,
                    tip_y=tip[1] + offset_y,
                    support=_support_of(local_anchor, tip, edges),
                    length_px=length,
                    angle_deg=angle_deg(tip[0] - local_anchor[0], tip[1] - local_anchor[1]),
                )
            )

    return _merge_duplicates(scored, config.max_candidates)


def _merge_duplicates(candidates: list[ShaftCandidate], limit: int) -> tuple[ShaftCandidate, ...]:
    """Keep the best of each group of near-parallel candidates, ranked by support.

    Ties in support go to the longer candidate. That matters more than it looks:
    a probabilistic Hough transform fragments one edge into several collinear
    pieces, all of which score the same support along the same ray, and taking
    the longest of them is what makes the reported tip the far end of the
    evidence rather than the far end of whichever fragment came back first.
    """
    ordered = sorted(candidates, key=lambda c: (c.support, c.length_px), reverse=True)
    kept: list[ShaftCandidate] = []
    for candidate in ordered:
        if any(
            abs(angle_difference_deg(candidate.angle_deg, other.angle_deg)) < _DUPLICATE_ANGLE_DEG
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return tuple(kept)
