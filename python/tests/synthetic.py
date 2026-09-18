"""A synthetic swing whose events are known by construction.

Shared by the phase-detection tests and the synchronisation tests, which need
the same thing for different reasons. Phase 4 needs a signal whose four instants
are inputs rather than readings off a plot. Phase 7 needs **one** swing sampled
by two cameras with different clocks, which is only possible if the swing exists
as a function of time independent of any clip -- so `swing_sequence` takes a
`world_start_s` saying where in that swing a camera started rolling, and returns
a clip whose own timestamps begin at zero.

Two cameras are then built by sampling the same function twice:

    reference = swing_sequence(world_start_s=0.0,      fps=120)
    target    = swing_sequence(world_start_s=-0.4,     fps=60)

and the target clip's clock reads 0.4 s ahead of the reference's at every instant
both saw, which is the ground truth the sync tests measure against.

It is a swing only in the shape of its signal -- one speed minimum at the highest
point, one speed maximum after it -- which is exactly and only what the rules
key on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    Landmark,
    LandmarkPoint,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from tests.conftest import SQUARE_FRAME

FPS = 120.0

# Event times the synthetic signal is built around, on the swing's own clock.
TAKEAWAY_S = 0.50
TOP_S = 1.30
IMPACT_S = 1.70
# The descent and the follow-through are mirror images in the arc model below,
# so the hands come to rest as long after impact as the top was before it.
FINISH_S = TOP_S + 2 * (IMPACT_S - TOP_S)
DURATION_S = 2.60

# Radius and lowest point of the arc the hands travel, in frame widths.
ARC_RADIUS = 0.22
ARC_LOW_HEIGHT = 0.33
ARC_TOP_ANGLE = 2.2

# A static body for the synthetic subject, in image coordinates (y downward).
# The torso has to be real: detection judges hand travel in torso lengths, so a
# fixture with every landmark stacked on one point has no scale to measure
# against and every ratio it produces is meaningless.
#
# The ankles are on the ground, and they are here for Phase 11: a teed ball rests
# on the ground, so the ball search region is anchored at the ankle midpoint
# rather than at the hands. Without them every landmark this fixture does not
# name falls back to a single point near the top of the frame, and a region
# anchored there would search the sky -- which is precisely the failure the real
# footage produced and the anchor exists to avoid.
TORSO_LENGTH = 0.25
GROUND_Y = 0.95
BODY: dict[int, tuple[float, float]] = {
    int(Landmark.LEFT_SHOULDER): (0.45, 0.25),
    int(Landmark.RIGHT_SHOULDER): (0.55, 0.25),
    int(Landmark.LEFT_HIP): (0.46, 0.25 + TORSO_LENGTH),
    int(Landmark.RIGHT_HIP): (0.54, 0.25 + TORSO_LENGTH),
    int(Landmark.LEFT_ANKLE): (0.46, GROUND_Y),
    int(Landmark.RIGHT_ANKLE): (0.54, GROUND_Y),
}


@dataclass(frozen=True)
class SwingShape:
    """The parameters of one swing, so that more than one swing can exist.

    The module's constants are the defaults, so every caller that predates this
    gets exactly the signal it was written against. Phase 12 needs a corpus of
    swings that differ from each other -- different tempos, different arcs,
    different people -- because a learned detector trained on one repeated swing
    learns that swing, and a split that holds out players needs players to hold
    out. Varying the parameters is the only honest way this fixture can supply
    that, and `tests/synthetic_labels.py` says plainly what it still cannot.
    """

    takeaway_s: float = TAKEAWAY_S
    top_s: float = TOP_S
    impact_s: float = IMPACT_S
    arc_radius: float = ARC_RADIUS
    arc_low_height: float = ARC_LOW_HEIGHT
    arc_top_angle: float = ARC_TOP_ANGLE

    @property
    def finish_s(self) -> float:
        """The descent and the follow-through are mirror images in this model."""
        return self.top_s + 2 * (self.impact_s - self.top_s)


DEFAULT_SHAPE = SwingShape()


def arc_angle(t: NDArray[np.float64], shape: SwingShape = DEFAULT_SHAPE) -> NDArray[np.float64]:
    """Angle of the hands along their arc, zero at the bottom.

    The hands are modelled on a circle rather than on a vertical line, because
    the vertical model gets impact wrong in a way that matters. On a real swing
    the hands are at the bottom of their arc at impact and moving horizontally,
    so their *vertical* speed there is near zero while their total speed peaks.
    A purely vertical fixture puts peak speed and lowest position at different
    instants, and would make the corroboration check meaningless.

    Backswing eases in and out, so speed is zero at the takeaway and again at
    the top. The descent and follow-through are one cosine sweep through the
    bottom, which puts maximum angular speed exactly where the angle crosses
    zero -- the lowest point of the arc, and the instant impact is defined to be.
    """
    angle = np.zeros_like(t)
    finish_s = shape.finish_s

    rising = (t >= shape.takeaway_s) & (t < shape.top_s)
    u = (t[rising] - shape.takeaway_s) / (shape.top_s - shape.takeaway_s)
    angle[rising] = shape.arc_top_angle * np.sin(np.pi / 2 * u) ** 2

    sweeping = (t >= shape.top_s) & (t <= finish_s)
    u = (t[sweeping] - shape.top_s) / (finish_s - shape.top_s)
    angle[sweeping] = shape.arc_top_angle * np.cos(np.pi * u)

    after = t > finish_s
    angle[after] = -shape.arc_top_angle
    return angle


def hand_path(
    t: NDArray[np.float64], shape: SwingShape = DEFAULT_SHAPE
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The arc as an image-space path.

    Returns x and y with y in image convention (increasing downward), so the
    detector's conversion to an upward height is genuinely exercised rather than
    bypassed by handing it a signal that already points the right way.
    """
    angle = arc_angle(t, shape)
    height = shape.arc_low_height + shape.arc_radius * (1.0 - np.cos(angle))
    return 0.5 + shape.arc_radius * np.sin(angle), 1.0 - height


def pose_sequence(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    t: NDArray[np.float64],
    *,
    visibility: NDArray[np.float64] | None = None,
    detected: NDArray[np.bool_] | None = None,
    per_landmark_visibility: dict[int, NDArray[np.float64]] | None = None,
    path: str = "/data/synthetic.mov",
    body: dict[int, tuple[float, float]] | None = None,
) -> PoseSequence:
    """A pose sequence whose wrists follow the given path on a still body.

    `per_landmark_visibility` overrides the shared value for named landmarks,
    which is how down-the-line footage behaves: one wrist is hidden behind the
    other, and which one changes through the swing.

    `body` replaces the static landmarks, which is how one fixture becomes more
    than one subject: everything this engine measures is scaled by torso length,
    so two bodies of different proportions are the cheapest test that the scaling
    is real rather than a constant that happens to cancel.
    """
    vis = np.full(t.size, 0.95) if visibility is None else visibility
    overrides = per_landmark_visibility or {}
    seen = np.ones(t.size, dtype=bool) if detected is None else detected
    skeleton = body if body is not None else BODY
    wrists = {int(Landmark.LEFT_WRIST), int(Landmark.RIGHT_WRIST)}

    frames = []
    for index in range(t.size):
        if not seen[index]:
            frames.append(PoseFrame(frame_index=index, timestamp_s=float(t[index]), detected=False))
            continue
        points = []
        for landmark in range(LANDMARK_COUNT):
            if landmark in wrists:
                # Left and right a little apart, so the midpoint is a distinct
                # point from either and the hand-source choice is exercised.
                offset = 0.01 if landmark == int(Landmark.LEFT_WRIST) else -0.01
                px, py = float(x[index]) + offset, float(y[index])
            else:
                px, py = skeleton.get(landmark, (0.5, 0.2))
            channel = overrides.get(landmark)
            points.append(
                LandmarkPoint(
                    x=px,
                    y=py,
                    z=0.0,
                    visibility=float(vis[index] if channel is None else channel[index]),
                    presence=0.99,
                )
            )
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=float(t[index]),
                detected=True,
                image=points,
                hip_local=points,
            )
        )

    found = int(np.count_nonzero(seen))
    return PoseSequence(
        video_path=path,
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="f" * 64, size_bytes=1
        ),
        geometry=SQUARE_FRAME,
        model=PoseModelInfo(
            name="fake",
            variant="fake",
            precision="float32",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 16, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=t.size,
            frames_detected=found,
            detection_rate=found / t.size if t.size else 0.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
        ),
        frames=frames,
    )


def swing_sequence(
    duration_s: float = DURATION_S,
    fps: float = FPS,
    *,
    world_start_s: float = 0.0,
    time_scale: float = 1.0,
    shape: SwingShape = DEFAULT_SHAPE,
    **kwargs: object,
) -> PoseSequence:
    """One camera's recording of the swing.

    `world_start_s` is where in the swing this camera started rolling; the clip's
    own timestamps always begin at zero, which is what a container reports. A
    camera that started 0.4 s *before* the swing's origin has
    `world_start_s = -0.4`, and its clock therefore reads 0.4 s ahead of a camera
    that started at the origin.

    `time_scale` stretches the swing itself, not the clock: 1.1 is a swing that
    took 10% longer to make. It is how a test builds two clips that are *not* the
    same swing, which no offset and no clock rate can reconcile -- the case the
    sync layer can only report as residual.
    """
    clock = np.arange(0.0, duration_s, 1.0 / fps)
    x, y = hand_path(world_start_s + clock / time_scale, shape)
    return pose_sequence(x, y, clock, **kwargs)  # type: ignore[arg-type]
