"""Ball detection: the first thing here that measures by watching something stop.

Every layer below measures a presence -- where a landmark is, where a shaft
points, how deep a joint sits -- and the reading is taken off a thing that is in
the picture. A golf ball sits in plain view doing nothing for hundreds of frames
and then is not there, and the instant this package exists to find is the boundary
between those two states.

    detector   the seam: one frame in, every candidate region out, no decision
    blob       the classical implementation: ROI, top-hat, components, geometry
    track      the decision: establish a position, watch it, locate the departure
    extract    video plus filtered poses in, a report out

The phase's argument is in `analyzer/contracts/ball.py`. The short form is that
tracking the ball in flight is the obvious thing to ask for and is unanswerable
on consumer footage -- a ball leaving at 70 m/s is a metre-long smear and then it
is out of frame -- while the ball at rest is the easiest object in the clip. So
this measures the ball where the ball is easy, and reads impact off the edge of
that interval. **The uncertainty is one frame interval and it is a bracket**,
which no other impact estimate in this project can say.

The fusion of that instant with Phase 4's and Phase 10's is deliberately not here:
it belongs to none of the three phases that produce an estimate, so it lives in
`analyzer/impact.py` with its contract in `analyzer/contracts/impact.py`.

`ball` sits with `phases`, `sync`, `club` and `biomechanics` on the golf-specific
side of `docs/architecture.md`'s line, and it is the most marginal member of that
set -- nothing in the mechanism knows what a golf ball is beyond its size against
a body. What makes it about golf is the assumption that the small round thing
disappearing near the player's feet disappeared because it was struck.
"""

from analyzer.ball.blob import ContrastBlobDetector
from analyzer.ball.detector import (
    BallAnchor,
    BallCandidate,
    BallDetectionError,
    BallDetectionResult,
    BallDetector,
)
from analyzer.ball.extract import detect_ball, frame_evidence
from analyzer.ball.track import (
    BallCandidateView,
    BallEvidence,
    BallTrackResult,
    track_ball,
)

__all__ = [
    "BallAnchor",
    "BallCandidate",
    "BallCandidateView",
    "BallDetectionError",
    "BallDetectionResult",
    "BallDetector",
    "BallEvidence",
    "BallTrackResult",
    "ContrastBlobDetector",
    "detect_ball",
    "frame_evidence",
    "track_ball",
]
