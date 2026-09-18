"""The whole path, on rendered pixels, against a departure frame that is an input.

The only tests in this phase that run everything at once: a swing is rendered
with a ball in it, every frame goes through the real detector, the tracker
identifies the ball and locates the instant, and the answer is compared with the
frame the ball was actually removed at. Nothing is constructed and nothing is
stubbed.

They are slower than the rest of the suite for that reason, and they earn it by
being the only place where a mistake in the units, the frame conversion or the
detector's own geometry would show up. A tracker test cannot catch a detector
that reports positions in the wrong frame, because it is handed positions.

`scripts/benchmark_ball.py` sweeps the same path across shutter, contrast,
clutter and frame rate; these assert the handful of properties that must hold on
every run rather than measuring a curve.
"""

from __future__ import annotations

import math

import pytest

from analyzer.ball import ContrastBlobDetector, track_ball
from analyzer.ball.detector import BallAnchor
from analyzer.ball.extract import frame_evidence
from analyzer.ball.track import BallEvidence, BallTrackResult
from analyzer.contracts.ball import BallConfig
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.impact import ImpactSource
from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.impact import fuse_impact
from analyzer.phases import detect_phases
from analyzer.phases.signals import swing_signals
from tests import synthetic
from tests import synthetic_ball as fixture
from tests import synthetic_club as club

# 60 fps rather than 120: the same clip in half the frames, which halves the
# runtime of every test in this file and still leaves the swing detectable and
# the ball unambiguous. The frame-rate-dependent claim -- that the bracket is one
# interval whatever the rate -- is asserted in `test_ball_track.py`, where it
# costs nothing.
FPS = 60.0


def run(**kwargs: object) -> tuple[BallTrackResult, SwingPhases, fixture.BallTruth]:
    """Render a clip, detect every frame, and track the ball through it."""
    rendered, poses, truth = fixture.swing_with_ball(fps=FPS, **kwargs)  # type: ignore[arg-type]
    config = BallConfig()
    detector = ContrastBlobDetector(config)
    anchor = BallAnchor(
        x=club.ADDRESS_GRIP_PX[0], y=club.ADDRESS_GRIP_PX[1], torso_px=club.TORSO_PX
    )

    evidence: list[BallEvidence] = []
    for item in rendered:
        detection = detector.detect(item.frame, anchor)
        evidence.append(
            frame_evidence(
                item.frame.index,
                item.frame.timestamp_s,
                detection.candidates,
                geometry=club.FRAME,
                torso_length=club.TORSO_LENGTH_FW,
                refusal=detection.refusal,
            )
        )

    filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)
    phases = detect_phases(filtered)
    result = track_ball(evidence, config, phases=phases, torso_length=club.TORSO_LENGTH_FW)
    return result, phases, truth


@pytest.fixture(scope="module")
def clean() -> tuple[BallTrackResult, SwingPhases, fixture.BallTruth]:
    """One clean rendered swing, shared by the tests that only read it."""
    return run()


def test_the_located_instant_is_the_frame_the_ball_was_removed_at(
    clean: tuple[BallTrackResult, SwingPhases, fixture.BallTruth],
) -> None:
    """The exit criterion for this phase, on pixels, with no tolerance.

    The truth is an integer and so is the answer, so this is an equality rather
    than an approximation -- which is the property that made the fixture draw
    presence per frame instead of flying a ball out of shot.
    """
    result, _, truth = clean

    assert result.departure is not None
    assert result.departure.first_absent_frame == truth.departure_frame
    assert result.departure.last_seen_frame == truth.last_seen_frame


def test_the_ball_is_found_where_it_was_drawn_and_at_the_size_it_was_drawn(
    clean: tuple[BallTrackResult, SwingPhases, fixture.BallTruth],
) -> None:
    """Catches a frame conversion that is wrong by a flip, a scale or an aspect.

    A tracker handed positions cannot catch this; only a test that renders at a
    known pixel and reads back a frame-widths coordinate can.
    """
    result, _, truth = clean

    established = result.established
    assert established is not None
    assert established.x == pytest.approx(truth.x / club.FRAME.width, abs=0.005)
    assert established.y == pytest.approx(
        (club.FRAME.height - truth.y) / club.FRAME.width, abs=0.005
    )
    # A golf ball is 0.047 torso lengths in radius. The measured radius runs a
    # little large because a thresholded antialiased disc is a little larger than
    # the disc, which is why this is a band rather than a value.
    assert 0.03 < established.radius_torso < 0.075


def test_the_ball_is_seen_in_every_frame_before_it_goes(
    clean: tuple[BallTrackResult, SwingPhases, fixture.BallTruth],
) -> None:
    """On a clean render there is no excuse for a hole, and a hole would matter.

    A missed frame immediately before the departure moves the reported instant,
    which is the one thing this phase reports.
    """
    result, _, _ = clean

    assert result.pre_departure_coverage == pytest.approx(1.0)


def test_the_identification_is_unambiguous_on_a_clean_clip(
    clean: tuple[BallTrackResult, SwingPhases, fixture.BallTruth],
) -> None:
    result, _, _ = clean

    established = result.established
    assert established is not None
    assert established.departed is True
    assert established.margin == pytest.approx(1.0)
    assert result.departure is not None
    assert result.departure.confidence.overall > 0.9


def test_a_stationary_rival_beside_the_ball_does_not_take_the_identification() -> None:
    """A tee marker within a ball's width of the drift bound is the real case.

    The units bug this guards against -- comparing a frame-widths distance
    against a torso-lengths bound -- merged the two into one object, which then
    never departed, and the clip reported no impact at all while the ball was in
    plain view.
    """
    marker = (fixture.BALL_PX[0] + 70.0, fixture.BALL_PX[1] + 4.0)
    result, _, truth = run(distractors=(marker,))

    assert result.departure is not None
    assert result.departure.first_absent_frame == truth.departure_frame
    established = result.established
    assert established is not None
    assert established.x == pytest.approx(truth.x / club.FRAME.width, abs=0.005)


def test_a_ball_that_is_never_removed_yields_no_instant() -> None:
    """A practice swing over real pixels. The ball is found; nothing is claimed."""
    result, _, _ = run(departure=10_000)

    assert result.established is not None
    assert result.departure is None
    assert any("can be struck at" in note for note in result.warnings)


def test_a_ball_covered_before_it_goes_reads_early_and_says_so() -> None:
    """The central caveat, end to end: a covered ball is not a struck ball.

    This is the phase's documented failure rather than a bug, so the assertion is
    about the reporting and not about the answer. The ball dims until it falls
    under the confidence bound, the run ends a couple of frames before the ball
    is actually removed, and **the reported instant is early by exactly that
    much**. What has to hold is that the report says so: the confidence falls and
    a warning names the cause.
    """
    result, _, truth = run(fade_frames=8)

    departure = result.departure
    assert departure is not None
    assert departure.frame_index < truth.departure_frame, "the documented early bias"

    assert departure.confidence.abruptness < 0.85
    assert departure.confidence.permanence < 1.0, "the dim ball is still a candidate"
    assert departure.confidence.overall < 0.7
    assert any("usual strength" in note for note in result.warnings)
    assert any("absence was verified" in note for note in result.warnings)


def test_a_clean_departure_scores_exactly_one_on_both_covering_factors() -> None:
    """The other half of the previous test: the factors do not fire spuriously.

    A warning that also appears on clean footage is a warning nobody reads, and
    both of these are compared against a bound rather than a distribution -- so
    the clean case has to sit at the top of the range rather than merely near it.
    """
    result, _, truth = run()

    departure = result.departure
    assert departure is not None
    assert departure.frame_index == truth.departure_frame
    assert departure.confidence.abruptness == pytest.approx(1.0)
    assert departure.confidence.permanence == pytest.approx(1.0)
    assert result.warnings == []


def test_the_fusion_prefers_the_observation_over_the_hand_speed_peak() -> None:
    """The whole point of Phase 11, assembled from the real pieces.

    The fixture puts every estimate at one instant by construction, so this
    asserts that the fusion is wired correctly and **not** that the methods agree
    on real footage -- which is a question no fixture can answer, because this one
    has no bias in it to find.
    """
    result, phases, truth = run()

    fused = fuse_impact(
        phases,
        ball=_report(result),
        club=None,
        smoothing_window_s=0.05,
        frame_interval_s=1.0 / FPS,
    )
    assert fused is not None
    assert fused.source is ImpactSource.BALL_DEPARTURE
    assert fused.observed is True
    assert fused.frame_index == truth.departure_frame
    assert fused.uncertainty_is_bracket is True
    assert fused.uncertainty_s == pytest.approx(1.0 / FPS, rel=0.05)

    hand_speed = fused.candidate(ImpactSource.HAND_SPEED)
    assert hand_speed is not None, "Phase 4's estimate is kept beside the observation"
    assert hand_speed.delta_frames is not None


def test_the_search_region_is_anchored_on_the_ground_not_the_hands() -> None:
    """The anchor that decided the reference footage, asserted on the fixture.

    A hands-anchored region is a disc of about three torso lengths centred
    chest-high, which on a tall frame contains the sky -- and a patch of cloud
    between two branches is rounder, better resolved and stiller than a golf ball
    a few pixels across. That failure is invisible in every per-frame number, so
    it is worth pinning where the anchor actually ends up rather than only that
    the clip comes out right.
    """
    from analyzer.ball.extract import _address_anchor
    from analyzer.contracts.phases import SwingEvent

    _, poses, truth = fixture.swing_with_ball(fps=FPS)
    filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)
    phases = detect_phases(filtered)
    anchor = _address_anchor(filtered, swing_signals(filtered), club.FRAME, phases)

    assert anchor is not None
    # On the ground, where the fixture puts the ankles -- not up at the hands.
    assert anchor[1] == pytest.approx(synthetic.GROUND_Y * club.FRAME.height, abs=5.0)
    assert anchor[1] > club.ADDRESS_GRIP_PX[1] + 100, "well below the hands"

    # And the ball is comfortably inside the region that anchor defines.
    radius = BallConfig().search_radius_torso * club.TORSO_PX
    assert math.hypot(truth.x - anchor[0], truth.y - anchor[1]) < radius

    takeaway = phases.event(SwingEvent.TAKEAWAY)
    assert takeaway is not None and takeaway.frame_index > 0


def test_the_anchor_falls_back_to_the_hands_without_ankles() -> None:
    """A clip framed from the waist up has no ground in it, and still gets a region.

    Worse than the ankles for the reason above, and better than refusing: the
    identification's own bounds are what decide whether anything is reported.
    """
    from analyzer.ball.extract import _address_anchor
    from analyzer.contracts.pose import Landmark

    _, poses, _ = fixture.swing_with_ball(fps=FPS)
    filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)
    for ankle in (Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE):
        filtered.landmarks.pop(ankle, None)

    anchor = _address_anchor(filtered, swing_signals(filtered), club.FRAME, None)
    assert anchor is not None
    assert anchor[1] == pytest.approx(club.ADDRESS_GRIP_PX[1], abs=40.0)


def _report(result: BallTrackResult) -> object:
    """The tracker's result as the contract the fusion reads.

    The fusion takes what crosses the engine boundary rather than the internal
    result, which is right, and this is the seam that lets a test avoid encoding
    a video in order to reach it.
    """
    from analyzer.contracts.ball import BallDetectorInfo, BallTrackingReport

    return BallTrackingReport(
        detected=result.observed_frames > 0,
        video_path="<fixture>",
        detector=BallDetectorInfo(name="test", method="test", opencv_version="test"),
        geometry=club.FRAME,
        torso_length=club.TORSO_LENGTH_FW,
        frame_count=len(result.frames),
        observed_frames=result.observed_frames,
        pre_departure_coverage=result.pre_departure_coverage,
        established=result.established,
        departure=result.departure,
        quality=result.quality,
        config=BallConfig(),
    )
