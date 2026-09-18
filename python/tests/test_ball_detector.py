"""The classical detector, against a ball rendered at a known place and size.

Where `test_ball_track.py` constructs evidence to exercise the decisions, these
run on actual pixels, because the things asserted here are properties of the
image processing and cannot be stated in candidates: that the operator finds a
ball on a dark surface and a dark ball on a light one, that it is not fooled by
the straight turf edge a ball sits on, and that it refuses a framing in which the
ball has no shape left.

The end-to-end case -- render a swing, detect every frame, track it, and check
the located instant against the frame the ball was removed at -- is in
`test_ball_pipeline.py`, which is the only test in this phase that exercises all
of it at once.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from analyzer.ball import ContrastBlobDetector
from analyzer.ball.detector import BallAnchor, BallDetectionError
from analyzer.contracts.ball import BallConfig, BallRefusal
from analyzer.ingestion.reader import VideoFrame
from tests import synthetic_ball as fixture
from tests import synthetic_club as club

TORSO_PX = club.TORSO_PX
RADIUS_PX = fixture.BALL_RADIUS_PX


def anchor(*, torso_px: float = TORSO_PX) -> BallAnchor:
    return BallAnchor(x=club.ADDRESS_GRIP_PX[0], y=club.ADDRESS_GRIP_PX[1], torso_px=torso_px)


def plain(
    *,
    background: int = 205,
    ball: int | None = 250,
    radius: float = RADIUS_PX,
    at: tuple[float, float] | None = None,
    noise: float = 3.0,
    seed: int = 0,
) -> VideoFrame:
    """A flat field with at most one disc on it, and nothing else.

    Deliberately simpler than the swing fixture: these tests are about the
    operator, and a body and a club in the frame would make a failure ambiguous
    between "did not find the ball" and "found something else first".
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((club.FRAME.height, club.FRAME.width), dtype=np.float32) * noise
    canvas = np.repeat(np.clip(field + background, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if ball is not None:
        centre = at if at is not None else fixture.BALL_PX
        cv2.circle(
            canvas,
            (round(centre[0]), round(centre[1])),
            max(1, round(radius)),
            (ball,) * 3,
            -1,
            cv2.LINE_AA,
        )
    return VideoFrame(index=0, timestamp_s=0.0, image=canvas)


def best(frame: VideoFrame, *, config: BallConfig | None = None, **kwargs: object):
    detector = ContrastBlobDetector(config or BallConfig())
    return detector.detect(frame, anchor(**kwargs))  # type: ignore[arg-type]


# --- the operator ---------------------------------------------------------


def test_finds_a_bright_ball_where_it_was_drawn() -> None:
    result = best(plain())

    assert result.candidates
    found = result.candidates[0]
    assert found.x == pytest.approx(fixture.BALL_PX[0], abs=2.0)
    assert found.y == pytest.approx(fixture.BALL_PX[1], abs=2.0)
    assert found.radius_px == pytest.approx(RADIUS_PX, rel=0.25)


def test_finds_a_dark_ball_on_a_light_surface_just_as_well() -> None:
    """The sign of the step is a fact about the mat, not about the ball.

    A detector that only looked for bright regions would fail here by returning
    `NO_CANDIDATE`, which reads as a framing problem and would send someone to
    move their camera.
    """
    bright = best(plain(background=205, ball=250))
    dark = best(plain(background=205, ball=140))

    assert bright.candidates and dark.candidates
    assert dark.candidates[0].x == pytest.approx(fixture.BALL_PX[0], abs=2.0)
    assert dark.candidates[0].contrast > 0.3


def test_an_empty_region_offers_nothing_and_says_why() -> None:
    result = best(plain(ball=None))

    assert not result.candidates
    assert result.refusal is BallRefusal.NO_CANDIDATE


def test_a_region_outside_the_picture_is_a_framing_refusal() -> None:
    """Distinct from finding nothing: one is fixed by moving the camera."""
    detector = ContrastBlobDetector(BallConfig())
    result = detector.detect(plain(), BallAnchor(x=-5000.0, y=-5000.0, torso_px=TORSO_PX))

    assert result.refusal is BallRefusal.OUT_OF_FRAME


def test_a_ball_too_small_to_have_a_shape_is_refused_as_framing() -> None:
    """Below about three pixels a disc's roundness is rasterisation, not shape.

    Driven by shrinking the subject rather than the ball, because that is the
    real case: a clip framed wide enough to show the whole range puts the ball
    under the bound while everything else about the capture is fine.
    """
    tiny = TORSO_PX / 12.0
    result = best(plain(radius=0.047 * tiny), torso_px=tiny)

    assert result.refusal is BallRefusal.OUT_OF_FRAME


# --- what it must not be fooled by ----------------------------------------


def test_the_turf_edge_under_the_ball_is_not_a_candidate() -> None:
    """A long straight high-contrast line a ball's width away is the hard case.

    It has excellent contrast and no roundness, which is exactly the split
    `circularity` exists to make -- and the reason the fixture draws the turf line
    close to the ball rather than at the bottom of the frame.
    """
    rendered = fixture.swing_with_ball(fps=120.0, duration_s=0.05)[0]
    result = best(rendered[0].frame)

    assert result.candidates, "the ball itself is still found"
    near_edge = [
        candidate
        for candidate in result.candidates
        if abs(candidate.y - fixture.ground_y()) < 2.0
        and abs(candidate.x - fixture.BALL_PX[0]) > 4 * RADIUS_PX
    ]
    assert not near_edge, "no candidate sits along the turf line away from the ball"


def test_a_bar_of_the_right_area_is_rejected_on_shape() -> None:
    """Area alone is not enough, and this is the candidate that proves it."""
    canvas = plain(ball=None).image.copy()
    # A rectangle with the same area as the ball, and none of its roundness.
    half = round(RADIUS_PX * RADIUS_PX * np.pi / 4.0 / 2.0)
    x, y = round(fixture.BALL_PX[0]), round(fixture.BALL_PX[1])
    cv2.rectangle(canvas, (x - half, y - 2), (x + half, y + 2), (250,) * 3, -1)

    result = best(VideoFrame(index=0, timestamp_s=0.0, image=canvas))
    assert result.refusal is BallRefusal.NO_CANDIDATE


def test_two_balls_are_two_candidates_rather_than_one_merged_region() -> None:
    """The margin check needs a runner-up that is a different object."""
    canvas = plain().image.copy()
    cv2.circle(
        canvas,
        (round(fixture.BALL_PX[0] + 80), round(fixture.BALL_PX[1])),
        round(RADIUS_PX),
        (250,) * 3,
        -1,
        cv2.LINE_AA,
    )

    result = best(VideoFrame(index=0, timestamp_s=0.0, image=canvas))
    assert len(result.candidates) >= 2
    positions = sorted(candidate.x for candidate in result.candidates[:2])
    assert positions[1] - positions[0] == pytest.approx(80.0, abs=3.0)


# --- the seam -------------------------------------------------------------


def test_detection_is_stateless_and_order_independent() -> None:
    """The Protocol requires it, and the tracker's whole design rests on it."""
    detector = ContrastBlobDetector(BallConfig())
    frame = plain()

    first = detector.detect(frame, anchor())
    detector.detect(plain(ball=None), anchor())
    second = detector.detect(frame, anchor())

    assert [(c.x, c.y, c.contrast) for c in first.candidates] == [
        (c.x, c.y, c.contrast) for c in second.candidates
    ]


def test_a_detection_result_carries_candidates_or_a_refusal_never_both() -> None:
    from analyzer.ball.detector import BallDetectionResult

    with pytest.raises(BallDetectionError):
        BallDetectionResult(candidates=(), refusal=None, region_pixels=0)


def test_an_anchor_needs_a_real_torso_scale() -> None:
    """Every bound the detector applies is stated in torso lengths."""
    with pytest.raises(BallDetectionError):
        BallAnchor(x=0.0, y=0.0, torso_px=0.0)


def test_the_detector_reports_the_opencv_that_ran() -> None:
    """The pin says what should be installed; a result records what was."""
    info = ContrastBlobDetector(BallConfig()).info

    assert info.opencv_version == cv2.__version__
    assert info.name == "contrast_blob"
