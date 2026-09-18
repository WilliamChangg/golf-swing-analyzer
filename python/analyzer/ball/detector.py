"""The ball detection seam.

`BallDetector` is to this phase what `ClubDetector` is to Phase 10, `PoseEstimator`
to Phase 2 and `FrameSource` to Phase 1: one Protocol, one classical
implementation behind it, and nothing of that implementation's vocabulary
reaching a caller. No `cv2.Mat`, no threshold and no OpenCV type appears in what
`detect` returns, which is what would let a learned ball detector trained in
Phase 12 be dropped in without touching the tracker, the report or the fusion.

## Stateless, and returning candidates rather than a decision

Exactly `ClubDetector`'s contract, for exactly its reason. The detector sees one
frame, knows nothing about any other, and returns **every region that could be a
ball** rather than a verdict on which one is.

All the temporal reasoning lives in `analyzer.ball.track`, and it has more to do
here than the club's tracker does. Which blob is the ball is not a question one
frame can answer -- a white shoe and a teed ball are the same object in a still
picture -- and the answer this phase needs is not a position at all but the frame
at which one particular position stops being occupied. Neither question exists
inside a single frame, so neither belongs behind this seam.

It also keeps the same failure out of the design. A detector handed the tracker's
current belief could narrow its search to it, which is how a tracker locks onto a
tee marker and confirms it every frame thereafter. This detector is never told
what the tracker thinks, so a lost tracker's candidate list still contains the
ball.

## What a search region is anchored to

`BallAnchor` is not a measurement this layer makes. It is a position the pose and
filtering layers already produced, handed down so the search has somewhere to
look, together with the subject's torso length in pixels -- which is the scale
every bound in `BallConfig` is stated in, so the same configuration describes the
same physical tolerance whatever the framing.

**Which position, and why it is not the hands.** Phase 10 anchors on the grip,
because a club is *held* and the shaft therefore certainly starts there. A ball
is merely **addressed**, which says only that it is within about a club length of
the hands and in no direction a single frame can name -- a disc of some 2.5 torso
lengths centred chest-high, which on a tall frame contains the sky. The stronger
fact is that a teed ball rests on the **ground**, and the pose layer knows where
the ground is because it knows where the ankles are. `analyzer/ball/extract.py`
chooses the position and records what anchoring on the hands cost when it was
tried.

This layer does not care which it was given; it searches a disc of
`search_radius_torso` about whatever it is handed. The distinction is recorded
here because a reader arriving from `club/detector.py` will expect the grip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from analyzer.contracts.ball import BallDetectorInfo, BallRefusal
from analyzer.ingestion.reader import VideoFrame


class BallDetectionError(RuntimeError):
    """A ball detector could not be created or could not run.

    Carries a remediation for the reason `ClubDetectionError` does: a subject too
    small in frame to measure a torso from, or a framing that puts the ball
    outside the picture, is something the caller fixes rather than something the
    engine recovers from.
    """

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class BallAnchor:
    """Where to search in one frame, in pixels, and how big the subject is.

    Ordinarily the player's ankle midpoint at address -- a teed ball rests on the
    ground -- but this layer treats it as an opaque position, and the module
    docstring says why that choice belongs one layer up.

    Pixels rather than frame widths for `GripAnchor`'s reason: this is the one
    place in the engine that works on the pixel grid, and converting into frame
    widths and back around a search that lives in pixels would be arithmetic
    performed twice for no gain.

    `torso_px` is the scale. Every geometric bound the detector applies -- the
    radius of the region, the size of the ball, how far it may have drifted -- is
    stated in torso lengths and multiplied by it here, so a subject filling half
    the frame and one filling a tenth are judged by the same physical tolerances.
    """

    x: float
    y: float
    torso_px: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.torso_px) or self.torso_px <= 0.0:
            raise BallDetectionError(
                f"A ball anchor needs a positive torso scale; got {self.torso_px}.",
                remediation=(
                    "The subject's shoulder-to-hip span could not be measured from this "
                    "clip. Check that the whole body is in frame."
                ),
            )


@dataclass(frozen=True)
class BallCandidate:
    """One region in one frame that could be the ball, in pixels.

    Deliberately not a `BallObservation`. An observation is a measurement the
    system stands behind, in the frame everything else measures in; a candidate is
    a thing the image offered, in the frame it was found in, before anything
    temporal has been asked of it. On a driving range most candidates are never
    observations, and on a good clip most frames offer several.

    `contrast` is the fraction of the region's own intensity range that separates
    the candidate's interior from the ring of background just outside it. The
    **magnitude** of that step and not its sign, so a white ball on a mat and a
    dark ball on snow score alike -- a detector that only looked for bright things
    would encode one convention about golf balls and fail silently on the other.
    """

    x: float
    y: float
    radius_px: float
    contrast: float
    circularity: float
    """How round the region is, as 4*pi*area / perimeter^2 -- 1.0 for a disc.
    Carried separately from `contrast` because they fail apart: a shadow is round
    and low-contrast, a mat edge is high-contrast and not round, and only a thing
    that is both is worth a second look."""


@dataclass(frozen=True)
class BallDetectionResult:
    """Everything one frame offered, ranked, or the reason it offered nothing.

    `candidates` is ordered best-first by contrast, with overlapping regions
    already merged -- two thresholded components a pixel apart are one object
    found twice, and leaving both in would make the runner-up a copy of the winner
    and the margin check meaningless in exactly the frames it exists for.

    `refusal` is set precisely when `candidates` is empty, and it distinguishes
    the two things a detector alone can see: a region that fell outside the
    picture, and a region that held nothing of a ball's size and shape. Those are
    a framing problem and a contrast problem respectively. Everything else --
    ambiguity, drift, a departure -- needs more than one frame and is the
    tracker's to name.
    """

    candidates: tuple[BallCandidate, ...]
    refusal: BallRefusal | None
    region_pixels: int

    def __post_init__(self) -> None:
        if bool(self.candidates) == (self.refusal is not None):
            raise BallDetectionError(
                "A detection result carries candidates or a refusal, never both and "
                f"never neither; got {len(self.candidates)} candidates and {self.refusal}."
            )


class BallDetector(Protocol):
    """Finds the regions in one frame that could be a golf ball.

    Stateless and order-independent: `detect` may be called on the frames of a
    clip in any order, repeatedly, and must return the same result each time. See
    the module docstring for why that is a requirement rather than an accident.
    """

    @property
    def info(self) -> BallDetectorInfo:
        """What actually ran, for provenance on the result."""
        ...

    def detect(self, frame: VideoFrame, anchor: BallAnchor) -> BallDetectionResult:
        """Every candidate ball in `frame`, searched around `anchor`."""
        ...
