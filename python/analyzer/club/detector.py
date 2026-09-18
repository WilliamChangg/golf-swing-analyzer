"""The club detection seam.

`ClubDetector` is to this phase what `PoseEstimator` is to Phase 2 and
`FrameSource` is to Phase 1: one Protocol, one classical implementation behind
it, and nothing of that implementation's vocabulary reaching a caller. No
`cv2.Mat`, no Hough parameter and no OpenCV type appears in what `detect`
returns, which is what would let a learned shaft detector trained in Phase 12 be
dropped in without touching the tracker, the report or the overlay.

## Unlike `PoseEstimator`, a detector here is stateless

`PoseEstimator` documents that implementations are stateful and must be called in
increasing timestamp order, because MediaPipe carries a tracker between frames. A
`ClubDetector` is the opposite by design: it sees one frame, knows nothing about
any other, and returns **every candidate that could be a shaft** rather than a
decision about which one is.

All the temporal reasoning lives in `analyzer.club.track`, and putting it there
buys the same thing Phase 4 bought by nesting its searches rather than running
them in order. A frame-by-frame tracker has to start at frame zero and commit;
this one runs offline over a clip that already exists, so it can start where the
evidence is **best** -- typically at address or the top, where the club is nearly
still and its edges are sharp -- and grow outward into the frames where the club
is fastest and the evidence is worst. Those are the frames that matter, and they
are the ones no forward-only tracker reaches in good condition.

It also keeps a failure out of the design. A detector handed the tracker's
current belief could use it to narrow its own search, which sounds like an
improvement and is how a tracker locks onto a door frame and stays locked: every
subsequent frame is then searched near the wrong answer and confirms it. The
detector here cannot do that, because it is never told what the tracker thinks.

## What a candidate is anchored to

`GripAnchor` is not a measurement this layer makes. It is the hand position the
pose and filtering layers already produced, handed down so the search has
somewhere to start, together with the subject's torso length in pixels -- which
is the scale every bound in `ClubConfig` is stated in, so that the same
configuration describes the same physical tolerance whatever the framing.

A detector that needed neither would be a line detector rather than a club
detector. The single most discriminating fact available about a golf shaft is
that somebody is holding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from analyzer.contracts.club import ClubDetectorInfo, ShaftRefusal
from analyzer.ingestion.reader import VideoFrame


class ClubDetectionError(RuntimeError):
    """A club detector could not be created or could not run.

    Carries a remediation for the same reason `PoseEstimationError` does: a frame
    size that does not match the anchors, or a subject too small in frame to
    measure a torso from, is something the caller fixes rather than something the
    engine recovers from.
    """

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class GripAnchor:
    """Where the hands are in one frame, in pixels, and how big the subject is.

    Pixels rather than frame widths because this is the one place in the engine
    that works on the pixel grid: the search region, the Canny kernel and the
    Hough accumulator are all defined on it, and converting into frame widths and
    back around a transform that lives in pixels would be arithmetic performed
    twice for no gain.

    `torso_px` is the scale. Every geometric bound the detector applies is stated
    in torso lengths and multiplied by it here, so a subject filling half the
    frame and one filling a tenth of it are judged by the same physical
    tolerances -- the same reasoning that makes Phase 4 measure hand travel
    against the torso rather than against the frame.
    """

    x: float
    y: float
    torso_px: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.torso_px) or self.torso_px <= 0.0:
            raise ClubDetectionError(
                f"A grip anchor needs a positive torso scale; got {self.torso_px}.",
                remediation=(
                    "The subject's shoulder-to-hip span could not be measured from this "
                    "clip. Check that the whole body is in frame."
                ),
            )


@dataclass(frozen=True)
class ShaftCandidate:
    """One line in one frame that could be the shaft, in pixels.

    Deliberately not a `ShaftObservation`. An observation is a measurement the
    system stands behind, in the frame everything else measures in; a candidate
    is a thing the image offered, in the frame it was found in, before anything
    temporal has been asked of it. Most candidates never become observations.

    `tip_x`/`tip_y` is the far end of the segment as found. `support` is the
    fraction of the ray **from the anchor to that tip** that lies on an edge
    pixel, which is a stricter thing to ask than what the Hough transform scored:
    a segment collinear with the club but starting a hand's width past the grip
    has a stretch of nothing between the hands and its near end, and that stretch
    counts against it here. It is what makes "the shaft reaches the hands" a
    measured property rather than a second threshold.
    """

    tip_x: float
    tip_y: float
    support: float
    length_px: float
    angle_deg: float
    """Direction from the anchor to the tip, in **pixel convention**: degrees on
    (-180, 180] measured from +x with y increasing *downward*, as the image grid
    has it. `analyzer.club.geometry` flips it once, on the way into a
    `ShaftObservation`, and nothing between here and there compares it with an
    angle from any other layer."""


@dataclass(frozen=True)
class DetectionResult:
    """Everything one frame offered, ranked, or the reason it offered nothing.

    `candidates` is ordered best-first by support, with near-duplicates already
    merged -- two Hough segments a degree apart are one object seen twice, and
    leaving both in would make the runner-up a copy of the winner and the margin
    check meaningless in exactly the frames it exists for.

    `refusal` is set precisely when `candidates` is empty, and it distinguishes
    the three things that produce nothing: a region with no edges in it at all, a
    region with edges that formed no line, and lines that formed but could not be
    a shaft. Those are a lighting problem, a blur problem and a framing problem
    respectively.
    """

    candidates: tuple[ShaftCandidate, ...]
    refusal: ShaftRefusal | None
    edge_pixels: int
    lines_found: int

    def __post_init__(self) -> None:
        if bool(self.candidates) == (self.refusal is not None):
            raise ClubDetectionError(
                "A detection result carries candidates or a refusal, never both and "
                f"never neither; got {len(self.candidates)} candidates and {self.refusal}."
            )


class ClubDetector(Protocol):
    """Finds the lines in one frame that could be a golf shaft.

    Stateless and order-independent: `detect` may be called on the frames of a
    clip in any order, repeatedly, and must return the same result each time. See
    the module docstring for why that is a requirement rather than an accident.
    """

    @property
    def info(self) -> ClubDetectorInfo:
        """What actually ran, for provenance on the result."""
        ...

    def detect(self, frame: VideoFrame, anchor: GripAnchor) -> DetectionResult:
        """Every candidate shaft in `frame`, anchored at `anchor`."""
        ...
