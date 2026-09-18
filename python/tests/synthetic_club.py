"""A shaft whose direction is an input, filmed under blur, clutter and occlusion.

The harness Phase 10 is measured against, and the fourth of its kind here:
`synthetic.py` builds a swing whose four instants are inputs,
`synthetic_board.py` a board photographed by a camera whose parameters are
inputs, `synthetic_body3d.py` a body whose 3D positions are inputs. This builds a
club whose **angle** is an input, so that an angular error is a number rather
than a reading off a picture.

It is deliberately built on `synthetic.py`'s arc rather than beside it. The shaft
pivots about that fixture's own hand path and its direction is driven by the same
`arc_angle`, so a clip rendered here and a pose sequence built there describe
**one** motion -- which is what makes it possible to detect the swing's phases
from the poses and then ask what fraction of each phase the club was tracked in.
That question is this phase's gate, and it cannot be asked of two fixtures that
merely resemble each other.

## What it renders, and why each part is there

A dark line of known length and direction, pivoting about the hands, over a torso
of known height. Three things can be added, and each corresponds to one line of
the capture protocol in `data/README.md` -- which is the point: those lines were
advice, and this is what turns them into measurements.

**Motion blur**, from a declared shutter. An exposure is a temporal average of
the scene, so the shaft is drawn at every angle it passed through during the
exposure and the results averaged. The sweep is `angular_rate * exposure` and the
exposure is `shutter_fraction * frame_interval` -- so 1.0 is a 360-degree
shutter, the worst any camera does, 0.5 is the cinema default, and 0.03 is about
1/1000 s at 30 fps, which is what the protocol asks for.

**Clutter**: sharp vertical lines at fixed positions in the frame. Door frames,
fence posts, window mullions. Fixed in the frame rather than relative to the
hands, which is the distinction that decides whether the fixture measures
anything -- see `_draw_clutter`. Placed so that the hands pass near them during
the swing, because a background line that never comes near the grip is rejected
on distance before anything else is asked of it.

**Occlusion**: an opaque block over part of the shaft, which is what the player's
own body does to the club at the top of the backswing on a face-on view.

## What it is not

It is not a photograph. There is no defocus, no rolling shutter, no compression,
no shaft that reflects the sky at one end and the grass at the other, and the
background is flat where a driving range is not. Every rate measured here is
therefore a **floor**, in the sense Phase 8's rendered board and Phase 9's
synthetic body are floors.

One departure from a real swing is worth naming because it flatters the result.
The club is 1.1 torso lengths long where a real one is about 2.5, so that its tip
stays inside the frame at every angle the arc reaches. A longer lever moves its
tip proportionally faster and smears proportionally more, so a real club at the
same frame rate and shutter carries about twice the blur measured here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.club.detector import GripAnchor
from analyzer.contracts.pose import Landmark, PoseSequence
from analyzer.ingestion.reader import VideoFrame
from tests.conftest import SQUARE_FRAME
from tests.synthetic import BODY, DURATION_S, TORSO_LENGTH, arc_angle, hand_path, swing_sequence

# Square, like every other synthetic fixture here, so one pixel is 1/1000 of a
# frame width on both axes and a coordinate in the fixture means the same number
# in the frame the measurements are taken in.
FRAME = SQUARE_FRAME
SCALE = float(FRAME.width)

TORSO_LENGTH_FW = TORSO_LENGTH
"""Shoulder-to-hip span in frame widths, re-exported so a caller building
evidence does not have to reach into two fixtures to find one number."""

TORSO_PX = TORSO_LENGTH * SCALE
"""The same span in pixels. The scale every bound in `ClubConfig` is stated in,
so the fixture has to have one and it has to be the one `synthetic.py` builds its
body from."""

SHAFT_TORSO = 1.1
SHAFT_PX = SHAFT_TORSO * TORSO_PX

SHAFT_THICKNESS_PX = 3
BACKGROUND = 205
SHAFT_GREY = 35
BODY_GREY = 120
CLUTTER_GREY = 60

# How much further the shaft turns than the hands do. A club is a lever: the
# hands sweep an arc and the shaft rotates further about them, which is why the
# club head is the fastest thing in the picture and the reason this phase is hard.
CLUB_GAIN = 1.4

ADDRESS_GRIP_PX = (
    float(hand_path(np.array([0.0]))[0][0] * SCALE),
    float(hand_path(np.array([0.0]))[1][0] * SCALE),
)
"""Where the hands sit at address, in pixels. The default pivot for a single
rendered frame, so that a test that does not care about the swing still renders
the club somewhere a club could be."""


@dataclass(frozen=True)
class RenderedBall:
    """A ball drawn into a frame, and the truth it was drawn from.

    Added for Phase 11 and kept here rather than in `synthetic_ball.py` for the
    reason the shaft is driven by `synthetic.py`'s arc: a ball and a club that are
    rendered by two different functions are two scenes that resemble each other,
    and the question Phase 11 asks -- does the frame the ball vanishes at agree
    with the frame the club arrives at -- cannot be asked of two such scenes.

    `grey` carries the intensity so a sweep can drive the ball towards its
    background, which is the capture variable this phase has that Phase 10's
    shutter is: contrast against what it sits on.
    """

    x: float
    y: float
    radius_px: float
    grey: float


@dataclass(frozen=True)
class RenderedFrame:
    """One rendered frame and the truth it was rendered from."""

    frame: VideoFrame
    anchor: GripAnchor
    angle_deg: float
    """True shaft direction in **pixel convention** -- degrees from +x with y
    increasing downward, which is the frame `ShaftCandidate` reports in. A test
    comparing against a `ShaftObservation` must negate it; `true_angle_fw` does
    that, and exists so that no test has to remember to."""

    blur_px: float
    """Length of the smear at the club head, in pixels. The quantity every sweep
    in `scripts/benchmark_club.py` is against."""

    @property
    def true_angle_fw(self) -> float:
        """The same direction in the frame every measurement is reported in."""
        return -self.angle_deg


def shaft_angle_deg(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shaft direction through the swing, in pixel convention.

    Ninety degrees is straight down the screen, which is roughly where a club
    points at address and again at impact; the backswing carries it up and past
    the vertical and the follow-through carries it up the other side.

    The total sweep is about 350 degrees and it crosses the +-180 discontinuity
    exactly once, in the downswing. That is deliberate. It is where an angle
    series that has not been unwrapped reports a rate of tens of thousands of
    degrees per second, and it is the only part of a swing anyone cares about.
    """
    return 90.0 - np.degrees(arc_angle(np.asarray(t, dtype=np.float64))) * CLUB_GAIN


def grip_px(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Where the hands are through the swing, in pixels. `(n, 2)`.

    `synthetic.py`'s own hand path, scaled. Using it rather than a fixed pivot
    means the blur rendered here is rotation *and* translation, as a real swing's
    is, and that the club is always drawn where this fixture's pose sequence says
    the hands were.
    """
    x, y = hand_path(np.asarray(t, dtype=np.float64))
    return np.stack((x * SCALE, y * SCALE), axis=-1)


def _blank(rng: np.random.Generator, noise: float) -> NDArray[np.uint8]:
    """A flat background with a little sensor noise.

    Noise is not decoration. Canny's thresholds are derived from the region's
    median intensity, and a perfectly flat patch has a degenerate histogram -- so
    a noiseless fixture would exercise a code path no camera produces.
    """
    if noise <= 0.0:
        return np.full((FRAME.height, FRAME.width, 3), BACKGROUND, dtype=np.uint8)
    # One channel's worth of noise, broadcast across the three. Sensor noise is
    # per-channel in a real camera, and drawing it that way costs three times as
    # many samples for a fixture whose detector converts to grey before it looks
    # at anything.
    field = rng.standard_normal((FRAME.height, FRAME.width), dtype=np.float32) * noise
    grey = np.clip(field + BACKGROUND, 0.0, 255.0).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def _draw_body(canvas: NDArray[np.uint8]) -> None:
    """The torso `synthetic.py` describes, drawn at exactly `TORSO_PX`."""
    corners = np.array(
        [
            BODY[int(Landmark.LEFT_SHOULDER)],
            BODY[int(Landmark.RIGHT_SHOULDER)],
            BODY[int(Landmark.RIGHT_HIP)],
            BODY[int(Landmark.LEFT_HIP)],
        ],
        dtype=np.float64,
    )
    cv2.fillPoly(canvas, [np.rint(corners * SCALE).astype(np.int32)], (BODY_GREY,) * 3)
    head = np.rint(np.array(BODY[int(Landmark.LEFT_SHOULDER)]) * SCALE).astype(int)
    cv2.circle(canvas, (int(head[0]) + 50, int(head[1]) - 60), 45, (BODY_GREY,) * 3, -1)


def _draw_clutter(canvas: NDArray[np.uint8], columns: tuple[float, ...]) -> None:
    """Sharp vertical lines at fixed positions in the frame.

    **Fixed in the frame, not relative to the hands**, which is the whole point
    and is easy to get wrong: clutter drawn through the moving grip travels with
    the player, and a line that travels with the player has perfect temporal
    continuity with itself. Rendered that way it is not a door frame, it is a
    second club -- and the tracker follows it happily, which is what the first
    version of this fixture measured.

    A real door frame stands still. It therefore competes only while the hands
    happen to pass near it, which is a few frames of a swing rather than all of
    them, and that is exactly the case the margin check exists for.
    """
    for column in columns:
        x = round(column)
        cv2.line(
            canvas, (x, 0), (x, FRAME.height), (CLUTTER_GREY,) * 3, SHAFT_THICKNESS_PX, cv2.LINE_AA
        )


def _draw_shaft(
    canvas: NDArray[np.uint8],
    pivot: tuple[float, float],
    angle_deg: float,
    sweep_deg: float,
    length_px: float,
) -> None:
    """The shaft, smeared over the angles it passed through during the exposure.

    An exposure integrates the scene over its own duration, so this averages the
    *line* over the sweep rather than blurring the finished image. The difference
    is the whole measurement: a Gaussian blur of a drawn line leaves a line with
    soft edges and a perfectly well-defined direction, while a real smear leaves
    a fan whose density falls off with the sweep and whose Hough response
    collapses. The second is what a club does; the first is what a fixture that
    cheated would do.
    """
    samples = max(1, round(math.radians(abs(sweep_deg)) * length_px / 2.0))

    # Everything the fan can reach, and nothing else. The alternative -- a
    # full-frame layer per sample -- is correct and spends most of its time
    # compositing background onto background; a heavily smeared frame draws
    # dozens of samples, so the crop is the difference between a fixture that can
    # be swept and one that cannot.
    margin = length_px + SHAFT_THICKNESS_PX + 2
    x0 = max(0, math.floor(pivot[0] - margin))
    y0 = max(0, math.floor(pivot[1] - margin))
    x1 = min(FRAME.width, math.ceil(pivot[0] + margin))
    y1 = min(FRAME.height, math.ceil(pivot[1] + margin))
    if x1 <= x0 or y1 <= y0:
        return

    accumulator = np.zeros((y1 - y0, x1 - x0), dtype=np.uint16)
    layer = np.empty_like(accumulator, dtype=np.uint8)
    origin = (round(pivot[0]) - x0, round(pivot[1]) - y0)

    for index in range(samples):
        offset = 0.0 if samples == 1 else sweep_deg * (index / (samples - 1) - 0.5)
        radians = math.radians(angle_deg + offset)
        tip = (
            round(pivot[0] + length_px * math.cos(radians)) - x0,
            round(pivot[1] + length_px * math.sin(radians)) - y0,
        )
        layer.fill(0)
        cv2.line(layer, origin, tip, 255, SHAFT_THICKNESS_PX, cv2.LINE_AA)
        accumulator += layer

    alpha = np.clip(accumulator.astype(np.float32) / (255.0 * samples), 0.0, 1.0)[:, :, None]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(region * (1.0 - alpha) + SHAFT_GREY * alpha, 0, 255).astype(
        np.uint8
    )


def _draw_ground(canvas: NDArray[np.uint8], y: float, grey: int) -> None:
    """A darker band below a horizon, which is what a ball actually sits on.

    Off by default, so Phase 10's measurements are taken against the background
    they were taken against. It exists because a ball floating on a flat field is
    not the test that matters: the mat or turf under it puts a long straight
    high-contrast edge within a ball's width of the thing being detected, and
    whether the shape filter survives that is the question.
    """
    cv2.rectangle(canvas, (0, round(y)), (FRAME.width, FRAME.height), (grey,) * 3, -1)


def _draw_ball(canvas: NDArray[np.uint8], ball: RenderedBall) -> None:
    """The ball, antialiased, drawn before the shaft so the club can cover it.

    Order matters and this is the realistic one: at impact the club head is
    between the camera and the ball on a face-on view, so a ball drawn last would
    show through a club that is physically in front of it -- and this phase's
    central caveat is precisely that a covered ball and a departed one are the
    same picture. A fixture that could not produce that picture could not test it.
    """
    cv2.circle(
        canvas,
        (round(ball.x), round(ball.y)),
        max(1, round(ball.radius_px)),
        (round(ball.grey),) * 3,
        -1,
        cv2.LINE_AA,
    )


def render(
    *,
    index: int = 0,
    timestamp_s: float = 0.0,
    angle_deg: float = 90.0,
    sweep_deg: float = 0.0,
    grip: tuple[float, float] | None = None,
    length_px: float = SHAFT_PX,
    clutter_x: tuple[float, ...] = (),
    occlusion: tuple[int, int, int, int] | None = None,
    noise: float = 3.0,
    seed: int = 0,
    balls: tuple[RenderedBall, ...] = (),
    ground_y: float | None = None,
    ground_grey: int = 90,
) -> RenderedFrame:
    """One frame, and the truth it was rendered from."""
    pivot = grip if grip is not None else ADDRESS_GRIP_PX
    rng = np.random.default_rng(seed + index)
    canvas = _blank(rng, noise)
    if ground_y is not None:
        _draw_ground(canvas, ground_y, ground_grey)
    _draw_body(canvas)
    if clutter_x:
        _draw_clutter(canvas, clutter_x)
    for ball in balls:
        _draw_ball(canvas, ball)
    _draw_shaft(canvas, pivot, angle_deg, sweep_deg, length_px)
    if occlusion is not None:
        cv2.rectangle(
            canvas, (occlusion[0], occlusion[1]), (occlusion[2], occlusion[3]), (BODY_GREY,) * 3, -1
        )

    return RenderedFrame(
        frame=VideoFrame(index=index, timestamp_s=timestamp_s, image=canvas),
        anchor=GripAnchor(x=pivot[0], y=pivot[1], torso_px=TORSO_PX),
        angle_deg=angle_deg,
        blur_px=math.radians(abs(sweep_deg)) * length_px,
    )


def swing(
    *,
    fps: float = 120.0,
    shutter_fraction: float = 0.25,
    duration_s: float = DURATION_S,
    clutter_x: tuple[float, ...] = (),
    occlusion: tuple[int, int, int, int] | None = None,
    noise: float = 3.0,
    seed: int = 0,
    balls_at: Callable[[int], tuple[RenderedBall, ...]] | None = None,
    ground_y: float | None = None,
    ground_grey: int = 90,
) -> list[RenderedFrame]:
    """A swing filmed at a declared frame rate and shutter.

    `shutter_fraction` is the exposure as a fraction of the frame interval, which
    is the form in which it decides anything: 1.0 is a 360-degree shutter, 0.5 is
    180 degrees -- the cinema default, and what a phone does in poor light -- and
    0.03 is about 1/1000 s at 30 fps. The smear at the club head follows from it
    and from how fast the shaft is turning at that instant, which is why the
    downswing is the only part of the clip where it matters.

    The frame grid is `swing_sequence`'s, exactly, so frame `i` here and frame
    `i` of the pose sequence are the same instant.

    `balls_at` is asked, per frame, which balls exist in it. A function rather
    than a list because the whole of Phase 11 is about a ball that exists in some
    frames and not others, and the frame it stops existing at is the thing being
    measured -- so the fixture has to express presence per frame or it cannot
    express the truth a result is scored against.
    """
    times = np.arange(0.0, duration_s, 1.0 / fps)
    angles = shaft_angle_deg(times)
    grips = grip_px(times)
    # Centred difference on the angle series: the rate the exposure integrates
    # over. The ends fall back to one-sided, and they are at address and in the
    # follow-through where the club is slow and the difference is immaterial.
    rates = np.gradient(angles, times)

    return [
        render(
            index=index,
            timestamp_s=float(times[index]),
            angle_deg=float(angles[index]),
            sweep_deg=float(rates[index]) * shutter_fraction / fps,
            grip=(float(grips[index, 0]), float(grips[index, 1])),
            clutter_x=clutter_x,
            occlusion=occlusion,
            noise=noise,
            seed=seed,
            balls=() if balls_at is None else balls_at(index),
            ground_y=ground_y,
            ground_grey=ground_grey,
        )
        for index in range(times.size)
    ]


def swing_with_poses(
    *, fps: float = 120.0, duration_s: float = DURATION_S, **kwargs: object
) -> tuple[list[RenderedFrame], PoseSequence]:
    """The same swing as rendered frames and as a pose sequence, frame for frame.

    The pair is what makes per-phase coverage measurable: the poses give a
    detectable swing with real event frames, and the frames give a club to track
    through it. Nothing reconciles the two afterwards -- they are the same
    `arc_angle` sampled on the same clock, so frame `i` is one instant.
    """
    frames = swing(fps=fps, duration_s=duration_s, **kwargs)  # type: ignore[arg-type]
    sequence = swing_sequence(duration_s=duration_s, fps=fps)
    if len(sequence.frames) != len(frames):  # pragma: no cover - guards a silent drift
        raise AssertionError(
            f"The rendered clip has {len(frames)} frames and the pose sequence "
            f"{len(sequence.frames)}; they must be sampled on one grid."
        )
    return frames, sequence


def peak_rate_deg_s(fps: float = 120.0, duration_s: float = DURATION_S) -> float:
    """The fastest the fixture's shaft turns. For sanity-checking a bound."""
    times = np.arange(0.0, duration_s, 1.0 / fps)
    return float(np.max(np.abs(np.gradient(shaft_angle_deg(times), times))))
