"""A ball that vanishes at a frame which is an input.

The harness Phase 11 is measured against, and the fifth of its kind here:
`synthetic.py` builds a swing whose four instants are inputs,
`synthetic_board.py` a board photographed by a camera whose parameters are
inputs, `synthetic_body3d.py` a body whose 3D positions are inputs,
`synthetic_club.py` a club whose angle is an input. This builds a ball whose
**departure frame** is an input, so that an error in the located instant is a
subtraction rather than a reading off a picture.

It is built on `synthetic_club.py` rather than beside it, and that is the whole
reason it can measure anything. The ball is placed where that fixture's club head
arrives at the bottom of its arc, and it is removed at the frame the same
fixture's hands reach the lowest point of the same arc. So the club, the hands
and the ball describe **one** event, and the question this phase exists to ask --
does the frame the ball vanishes at agree with the frame the club arrives, and
with the frame the hands peak -- is a question about one scene rather than about
three that resemble each other.

## The departure is the truth, and it is exact

`departure_frame` is the first frame with no ball in it. The ball is drawn in
every frame before it and in none after, so a detector's answer is right or wrong
by an integer, with no tolerance to argue about. That is the property that makes
the measurement in `scripts/benchmark_ball.py` worth anything, and it is the
reason the fixture draws presence per frame rather than moving a ball out of
shot: a ball that flies away is a ball that is smeared across some frames and
partly visible in others, and then the truth is a matter of opinion.

## What it is not

It is not a photograph, in exactly the senses `synthetic_club.py` lists: no
defocus, no rolling shutter, no compression, no grass that is a thousand
individually bright blades, no ball that is half in shadow. Every rate measured
against it is a **floor**.

Two departures from real footage are worth naming because they flatter the
result. The background is flat, so the top-hat response has nothing in it but the
objects deliberately drawn; a real range offers daisies, ball marks, sprinkler
heads and old divots, which is what `distractors` exists to put back. And the
ball is drawn as a uniform disc, where a real one is a lit sphere with a shadowed
underside and a bright specular cap -- which is rounder in outline than a real
ball measures, so the circularity scores here are optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analyzer.contracts.pose import PoseSequence
from tests import synthetic_club as club
from tests.synthetic import DURATION_S, IMPACT_S, swing_sequence

# Where the ball sits: the club head's position at the bottom of the arc, which is
# the instant `synthetic.py` defines impact to be. Derived rather than chosen, so
# that moving the arc moves the ball with it.
_IMPACT_ANGLE_DEG = float(club.shaft_angle_deg(np.array([IMPACT_S]))[0])
_IMPACT_GRIP_PX = club.grip_px(np.array([IMPACT_S]))[0]

# A golf ball is 42.7 mm and an adult's shoulder-to-hip span about 450 mm, so the
# ball is 0.047 torso lengths in radius -- which is `BallConfig.ball_radius_torso`,
# and it is the same number here because both are computed from the same two
# published dimensions rather than fitted to each other.
BALL_RADIUS_PX = 0.047 * club.TORSO_PX

# How far past the end of the shaft the ball's centre sits, in its own radii.
#
# Not a fudge: `synthetic_club.py` draws the club as a line ending at the club
# head, and a club head is a lump on the end of that line which rests *behind* the
# ball rather than on top of it. Placing the ball's centre at the line's end would
# draw a three-pixel shaft straight through the middle of it, splitting it into two
# crescents that no circularity filter should accept -- and the fixture would then
# be measuring its own stick figure rather than the detector. Just past the end,
# the shaft stops at the ball's near edge, which is what address and contact both
# actually look like.
_BALL_STANDOFF_RADII = 1.5

BALL_PX = (
    float(
        _IMPACT_GRIP_PX[0]
        + (club.SHAFT_PX + _BALL_STANDOFF_RADII * BALL_RADIUS_PX)
        * math.cos(math.radians(_IMPACT_ANGLE_DEG))
    ),
    float(
        _IMPACT_GRIP_PX[1]
        + (club.SHAFT_PX + _BALL_STANDOFF_RADII * BALL_RADIUS_PX)
        * math.sin(math.radians(_IMPACT_ANGLE_DEG))
    ),
)
"""Where the ball is, in pixels: just past where the club head arrives at the
bottom of its arc, so the head reaches the ball's near edge at impact."""

BALL_GREY = 250.0
"""A white ball. Bright against the turf band below, and the sweep in
`scripts/benchmark_ball.py` drives it towards the background to find where
contrast stops being enough."""

GROUND_GREY = 95
"""Turf. Dark enough to give a white ball real contrast and to put a long
straight horizontal edge within a ball's width of it, which is the shape the
circularity filter has to survive."""


def ground_y() -> float:
    """Where the turf line sits: just above the ball, as it does on a tee shot.

    Deliberately close. A horizon drawn far from the ball is not a test of
    anything -- the whole difficulty is that the edge the ball sits on is a
    high-contrast straight line a few pixels away from the object being found.
    """
    return BALL_PX[1] - 1.5 * BALL_RADIUS_PX


def departure_frame(fps: float) -> int:
    """The first frame with no ball in it, for a given frame rate.

    Rounded to the frame grid the clip is sampled on, so the truth is an integer
    on that grid rather than a time that falls between two frames.
    """
    return round(IMPACT_S * fps)


@dataclass(frozen=True)
class BallTruth:
    """What a rendered clip was rendered from, for a result to be scored against."""

    departure_frame: int
    """First frame with no ball. A detector's `first_absent_frame` should equal
    this exactly."""

    last_seen_frame: int
    """Final frame carrying a ball. `departure_frame - 1`, by construction."""

    x: float
    y: float
    radius_px: float
    fps: float

    @property
    def departure_s(self) -> float:
        return self.departure_frame / self.fps


def _visible(index: int, departure: int, fade_frames: int) -> float:
    """How bright the ball is at one frame, as a fraction of its full intensity.

    `fade_frames` dims it over the frames immediately before it goes, which is
    what a club head crossing the line of sight does on a down-the-line view. It
    is the case `DepartureConfidence.abruptness` exists to notice, and without it
    in the fixture that factor would be scored at 1.0 on every clip and would
    never be shown to work.
    """
    if index >= departure:
        return 0.0
    if fade_frames <= 0:
        return 1.0
    remaining = departure - index
    if remaining > fade_frames:
        return 1.0
    return remaining / (fade_frames + 1)


def swing_with_ball(
    *,
    fps: float = 120.0,
    shutter_fraction: float = 0.25,
    duration_s: float = DURATION_S,
    departure: int | None = None,
    ball_grey: float = BALL_GREY,
    fade_frames: int = 0,
    distractors: tuple[tuple[float, float], ...] = (),
    distractor_departure: int | None = None,
    noise: float = 3.0,
    seed: int = 0,
) -> tuple[list[club.RenderedFrame], PoseSequence, BallTruth]:
    """A swing with a ball in it, the matching poses, and the truth of both.

    `distractors` are further stationary ball-sized objects, in pixels. A tee
    marker, a second range ball, a white shoe. They do not depart unless
    `distractor_departure` says so, which is the case that tells this phase it
    cannot know which object was struck.
    """
    gone = departure_frame(fps) if departure is None else departure
    truth = BallTruth(
        departure_frame=gone,
        last_seen_frame=gone - 1,
        x=BALL_PX[0],
        y=BALL_PX[1],
        radius_px=BALL_RADIUS_PX,
        fps=fps,
    )

    def balls_at(index: int) -> tuple[club.RenderedBall, ...]:
        drawn: list[club.RenderedBall] = []
        alpha = _visible(index, gone, fade_frames)
        if alpha > 0.0:
            # Dimmed towards the turf it sits on rather than towards black, which
            # is what being covered by something actually looks like: the ball's
            # own contrast against its surround falls, and the surround is the
            # thing it is measured against.
            drawn.append(
                club.RenderedBall(
                    x=truth.x,
                    y=truth.y,
                    radius_px=truth.radius_px,
                    grey=GROUND_GREY + alpha * (ball_grey - GROUND_GREY),
                )
            )
        for x, y in distractors:
            if distractor_departure is not None and index >= distractor_departure:
                continue
            drawn.append(club.RenderedBall(x=x, y=y, radius_px=truth.radius_px, grey=ball_grey))
        return tuple(drawn)

    frames = club.swing(
        fps=fps,
        shutter_fraction=shutter_fraction,
        duration_s=duration_s,
        noise=noise,
        seed=seed,
        balls_at=balls_at,
        ground_y=ground_y(),
        ground_grey=GROUND_GREY,
    )
    sequence = swing_sequence(duration_s=duration_s, fps=fps)
    if len(sequence.frames) != len(frames):  # pragma: no cover - guards a silent drift
        raise AssertionError(
            f"The rendered clip has {len(frames)} frames and the pose sequence "
            f"{len(sequence.frames)}; they must be sampled on one grid."
        )
    return frames, sequence, truth
