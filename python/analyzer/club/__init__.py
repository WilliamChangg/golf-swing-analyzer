"""Club tracking: the first thing here that measures an object rather than a body.

Every layer below reads landmarks a model was trained to find, which arrive with
the model's own opinion of how well it saw them. A golf shaft has no model, no
landmark index and no reported visibility. It has edges, and a position that is
known for certain about one of its ends -- somebody is holding it.

    detector   the seam: one frame in, every candidate line out, no decision
    hough      the classical implementation: ROI, Canny, probabilistic Hough
    geometry   the frame change and the two angle conventions it must not mix
    track      the decision, seeded where the evidence is best and grown outward
    extract    video plus filtered poses in, a report out

The phase's argument is in `analyzer/contracts/club.py`, and it is the same shape
Phases 8 and 9 found one and two layers down. The number a line detector reports
about itself -- how much edge evidence backs its answer -- is blind to the failure
that matters, and blind in a nameable direction: through the downswing the club
is the *blurriest* thing in the frame and the background is the sharpest, so
support ranks a door frame above a club exactly where the club matters most. What
catches it is what the frame was compared against, in space and in time.

`club` sits with `phases`, `sync` and `biomechanics` on the golf-specific side of
`docs/architecture.md`'s line, and it is the second marginal member of that set
for the reason `sync` is the first. Its mechanism would follow any rigid rod held
in a pair of hands; only the assumption that there is one is about golf.
"""

from analyzer.club.detector import (
    ClubDetectionError,
    ClubDetector,
    DetectionResult,
    GripAnchor,
    ShaftCandidate,
)
from analyzer.club.extract import frame_evidence, track_club
from analyzer.club.geometry import (
    angle_deg,
    angle_difference_deg,
    angle_to_frame_widths,
    point_segment_distance_px,
    unwrap_deg,
)
from analyzer.club.hough import HoughShaftDetector
from analyzer.club.track import CandidateView, FrameEvidence, TrackResult, track_shafts

__all__ = [
    "CandidateView",
    "ClubDetectionError",
    "ClubDetector",
    "DetectionResult",
    "FrameEvidence",
    "GripAnchor",
    "HoughShaftDetector",
    "ShaftCandidate",
    "TrackResult",
    "angle_deg",
    "angle_difference_deg",
    "angle_to_frame_widths",
    "frame_evidence",
    "point_segment_distance_px",
    "track_club",
    "track_shafts",
    "unwrap_deg",
]
