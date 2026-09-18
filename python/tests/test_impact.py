"""Reconciling four impact estimates into one, and refusing to average them.

The fusion has no image processing in it and no numerics worth testing; what it
has is a policy, and the tests are about the policy. Every one of them is a
statement that could have gone the other way and would have looked reasonable:

* that an observation beats a proxy however confident the proxy is;
* that the reported instant is never a blend, so its provenance is nameable;
* that the estimates which lost are kept with their deltas, because those deltas
  are the measurement this phase exists to make possible;
* that a disagreement too large to be a bias is said out loud rather than
  resolved silently in favour of the higher-precedence source.
"""

from __future__ import annotations

import pytest

from analyzer.contracts.ball import (
    BallConfig,
    BallDeparture,
    BallDetectorInfo,
    BallTrackingReport,
    DepartureConfidence,
)
from analyzer.contracts.club import (
    ClubConfig,
    ClubDetectorInfo,
    ClubImpactEstimate,
    ClubTrackingReport,
)
from analyzer.contracts.impact import ImpactSource
from analyzer.contracts.phases import (
    DetectedEvent,
    DetectedPhase,
    EventConfidence,
    HandSignalInfo,
    HandSource,
    PhaseConfig,
    SwingEvent,
    SwingPhase,
    SwingPhases,
)
from analyzer.contracts.pose import FrameGeometry
from analyzer.impact import fuse_impact

FPS = 120.0
GEOMETRY = FrameGeometry(width=1000, height=1000)

# Frames the four estimates are placed at by default. Spread a few frames apart,
# in the order the physics predicts: the hands peak first, the hands reach their
# lowest point, the club head arrives, the ball goes.
HAND_SPEED_FRAME = 196
HAND_LOW_FRAME = 201
CLUB_HEAD_FRAME = 203
BALL_FRAME = 204


def phases(
    *,
    impact_frame: int = HAND_SPEED_FRAME,
    corroboration: int | None = HAND_LOW_FRAME,
    detected: bool = True,
) -> SwingPhases:
    confidence = EventConfidence(overall=0.8, margin=0.9, visibility=1.0, resolution=0.9)
    events = [
        DetectedEvent(
            event=SwingEvent.TAKEAWAY,
            frame_index=64,
            timestamp_s=64 / FPS,
            confidence=confidence,
            methodology="test",
        ),
        DetectedEvent(
            event=SwingEvent.TOP,
            frame_index=156,
            timestamp_s=156 / FPS,
            confidence=confidence,
            methodology="test",
        ),
        DetectedEvent(
            event=SwingEvent.IMPACT,
            frame_index=impact_frame,
            timestamp_s=impact_frame / FPS,
            confidence=confidence,
            methodology="Maximum hand speed. A kinematic estimate.",
            corroboration_frame=corroboration,
            corroboration_delta_s=(
                None if corroboration is None else (corroboration - impact_frame) / FPS
            ),
        ),
        DetectedEvent(
            event=SwingEvent.FINISH,
            frame_index=251,
            timestamp_s=251 / FPS,
            confidence=confidence,
            methodology="test",
        ),
    ]
    return SwingPhases(
        detected=detected,
        events=events if detected else [],
        phases=[
            DetectedPhase(
                phase=SwingPhase.DOWNSWING,
                start_frame=156,
                end_frame=impact_frame,
                start_s=156 / FPS,
                end_s=impact_frame / FPS,
                duration_s=(impact_frame - 156) / FPS,
                confidence=0.8,
            )
        ]
        if detected
        else [],
        hand=HandSignalInfo(
            source=HandSource.MIDPOINT,
            valid_frames=312,
            total_frames=312,
            peak_speed=1.0,
            travel=0.5,
            torso_length=0.25,
            travel_ratio=2.0,
        ),
        frames=312,
        config=PhaseConfig(),
    )


def ball(*, frame: int = BALL_FRAME, confidence: float = 1.0) -> BallTrackingReport:
    return BallTrackingReport(
        detected=True,
        video_path="<test>",
        detector=BallDetectorInfo(name="test", method="test", opencv_version="test"),
        geometry=GEOMETRY,
        torso_length=0.25,
        frame_count=312,
        observed_frames=frame,
        pre_departure_coverage=1.0,
        departure=BallDeparture(
            frame_index=frame,
            timestamp_s=frame / FPS,
            last_seen_frame=frame - 1,
            last_seen_s=(frame - 1) / FPS,
            first_absent_frame=frame,
            first_absent_s=frame / FPS,
            interval_s=1.0 / FPS,
            confidence=DepartureConfidence(
                overall=confidence, establishment=1.0, abruptness=1.0, permanence=confidence
            ),
            methodology="ball departure, per the test",
        ),
        config=BallConfig(),
    )


def club(*, frame: int = CLUB_HEAD_FRAME) -> ClubTrackingReport:
    return ClubTrackingReport(
        tracked=True,
        video_path="<test>",
        detector=ClubDetectorInfo(name="test", method="test", opencv_version="test"),
        geometry=GEOMETRY,
        torso_length=0.25,
        frame_count=312,
        tracked_frames=300,
        coverage=0.96,
        impact=ClubImpactEstimate(
            frame_index=frame,
            timestamp_s=frame / FPS,
            methodology="lowest observed club head, per the test",
            head_frames_searched=40,
        ),
        config=ClubConfig(),
    )


def fuse(**kwargs: object):
    kwargs.setdefault("smoothing_window_s", 0.05)
    kwargs.setdefault("frame_interval_s", 1.0 / FPS)
    return fuse_impact(**kwargs)  # type: ignore[arg-type]


# --- precedence -----------------------------------------------------------


def test_the_ball_wins_over_every_other_estimate() -> None:
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert result.source is ImpactSource.BALL_DEPARTURE
    assert result.frame_index == BALL_FRAME
    assert result.observed is True


def test_a_confident_proxy_does_not_beat_an_unconfident_observation() -> None:
    """Precedence is about what each source measures, not about how sure it is.

    A hand-speed peak on clean footage scores well and is still a proxy for an
    event it cannot see; a ball departure on a clip that ended early scores badly
    and is still a frame in which the ball was there and a frame in which it was
    not.
    """
    result = fuse(phases=phases(), ball=ball(confidence=0.1), club=club())

    assert result is not None
    assert result.source is ImpactSource.BALL_DEPARTURE
    assert result.confidence == pytest.approx(0.1)


def test_the_club_head_is_used_when_no_ball_was_seen() -> None:
    result = fuse(phases=phases(), ball=None, club=club())

    assert result is not None
    assert result.source is ImpactSource.CLUB_HEAD
    assert result.frame_index == CLUB_HEAD_FRAME
    assert result.observed is False


def test_the_hand_estimates_are_the_last_resort_in_order() -> None:
    """`HAND_LOW` above `HAND_SPEED`: the one with no known bias goes first."""
    with_low = fuse(phases=phases(), ball=None, club=None)
    assert with_low is not None
    assert with_low.source is ImpactSource.HAND_LOW

    without_low = fuse(phases=phases(corroboration=None), ball=None, club=None)
    assert without_low is not None
    assert without_low.source is ImpactSource.HAND_SPEED


def test_nothing_at_all_produces_no_instant() -> None:
    """A clip with no swing and no ball has no event to time."""
    assert fuse(phases=phases(detected=False), ball=None, club=None) is None
    assert fuse(phases=None, ball=None, club=None) is None


def test_a_ball_alone_is_enough() -> None:
    """A clip Phase 4 will not call a swing can still have an observed impact."""
    result = fuse(phases=None, ball=ball(), club=None)

    assert result is not None
    assert result.source is ImpactSource.BALL_DEPARTURE
    assert len(result.candidates) == 1


# --- the reported instant is never a blend --------------------------------


def test_the_instant_is_one_source_verbatim() -> None:
    """Averaging an unbiased observation with biased proxies is the thing not done.

    The mean of the four frames here is 201, which is not any of them. A fused
    result carrying 201 would be defensible-looking and would have no provenance
    a reader could act on.
    """
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert result.frame_index in {HAND_SPEED_FRAME, HAND_LOW_FRAME, CLUB_HEAD_FRAME, BALL_FRAME}
    assert result.frame_index == BALL_FRAME
    assert result.methodology == ball().departure.methodology  # type: ignore[union-attr]


# --- the deltas are the measurement ---------------------------------------


def test_every_source_that_answered_is_kept_with_its_delta() -> None:
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert {candidate.source for candidate in result.candidates} == set(ImpactSource)

    hand_speed = result.candidate(ImpactSource.HAND_SPEED)
    assert hand_speed is not None
    assert hand_speed.delta_frames == HAND_SPEED_FRAME - BALL_FRAME
    assert hand_speed.delta_s == pytest.approx((HAND_SPEED_FRAME - BALL_FRAME) / FPS)


def test_the_winner_is_in_the_list_with_a_zero_delta() -> None:
    """Dropping it would make the list mean "the ones that lost"."""
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    chosen = result.candidate(ImpactSource.BALL_DEPARTURE)
    assert chosen is not None
    assert chosen.delta_s == pytest.approx(0.0)
    assert chosen.delta_frames == 0


def test_the_hand_speed_bias_is_reported_as_a_number() -> None:
    """Phase 4's documented caveat, turned into a measurement by the observation."""
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert any("peak hand speed runs" in note and "early" in note for note in result.warnings)


def test_agreement_is_the_widest_disagreement() -> None:
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert result.agreement_s == pytest.approx((BALL_FRAME - HAND_SPEED_FRAME) / FPS)


# --- the error bars are not the same kind of claim -------------------------


def test_only_the_ball_carries_a_bracket() -> None:
    """One frame interval that impact is inside, against a scale it might have moved on."""
    observed = fuse(phases=phases(), ball=ball(), club=club())
    assert observed is not None
    assert observed.uncertainty_is_bracket is True
    assert observed.uncertainty_s == pytest.approx(1.0 / FPS)

    inferred = fuse(phases=phases(), ball=None, club=None)
    assert inferred is not None
    assert inferred.uncertainty_is_bracket is False
    assert inferred.uncertainty_s == pytest.approx(0.05)


def test_an_inferred_instant_says_it_was_not_seen() -> None:
    result = fuse(phases=phases(), ball=None, club=None)

    assert result is not None
    assert result.observed is False
    assert any("inferred from the player's motion" in note for note in result.warnings)


# --- disagreements too large to be a bias ---------------------------------


def test_an_impossible_disagreement_is_said_out_loud() -> None:
    """Half a downswing apart is not two readings of one event.

    Without this the fusion silently prefers the ball and reports a confident
    instant while holding evidence that something is badly wrong.
    """
    result = fuse(phases=phases(), ball=ball(), club=club(frame=260))

    assert result is not None
    assert result.source is ImpactSource.BALL_DEPARTURE
    assert any("not disagreeing about one event" in note for note in result.warnings)


def test_ordinary_disagreements_are_not_warned_about() -> None:
    """A few frames is the bias this phase exists to measure, not an anomaly."""
    result = fuse(phases=phases(), ball=ball(), club=club())

    assert result is not None
    assert not any("not disagreeing about one event" in note for note in result.warnings)
