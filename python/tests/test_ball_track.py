"""The temporal pass: identifying the ball, and locating the frame it left.

These run on **constructed evidence** rather than on rendered frames, which is
the point of the seam: `track_ball` takes candidate positions and timestamps, so
a test can state exactly what the image offered -- a ball and a tee marker of
equal contrast, a ball that comes back, a ball that goes during the backswing --
and assert what the tracker does with it. Driving those cases through a renderer
would mean tuning pixels until the detector produced the situation under test,
which tests the fixture.

The ones that matter most assert **refusals and gates**, because almost every way
this phase can be wrong produces a confident number rather than an error:

* that a ball which reappears was occluded and not struck;
* that a departure before the top of the backswing is not an impact;
* that a ball still in frame when the clip ends yields no instant at all;
* that a one-frame speck in the follow-through cannot move the answer;
* that the two confidences are computed from disjoint evidence, so neither can
  flatter the other.
"""

from __future__ import annotations

import pytest

from analyzer.ball import track_ball
from analyzer.ball.track import BallCandidateView, BallEvidence
from analyzer.contracts.ball import BallConfig, BallRefusal
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

FPS = 120.0
TORSO = 0.25

# Where the constructed ball sits, in frame widths. Far enough from the origin
# that a distractor can be placed either side of it without either going negative.
BALL = (0.50, 0.10)


def candidate(
    position: tuple[float, float], *, contrast: float = 0.9, radius: float = 0.012
) -> BallCandidateView:
    """One candidate at a given place, with the size a golf ball has."""
    return BallCandidateView(
        x=position[0],
        y=position[1],
        radius=radius,
        radius_px=radius * 1000.0,
        radius_torso=radius / TORSO,
        contrast=contrast,
        circularity=0.9,
    )


def evidence(
    frames: int,
    *,
    present: object = True,
    contrast: float = 0.9,
    position: tuple[float, float] = BALL,
    extra: list[tuple[float, float]] | None = None,
    extra_until: int | None = None,
    fps: float = FPS,
) -> list[BallEvidence]:
    """A clip's worth of evidence with the ball present on the named frames.

    `present` is either a bool for the whole clip or a predicate on the frame
    index, which is how every departure case in this file is written: the frame
    the ball stops being offered is the truth a result is scored against.
    """

    def visible(index: int) -> bool:
        return present(index) if callable(present) else bool(present)

    built: list[BallEvidence] = []
    for index in range(frames):
        candidates: list[BallCandidateView] = []
        if visible(index):
            candidates.append(candidate(position, contrast=contrast))
        for place in extra or []:
            if extra_until is None or index < extra_until:
                candidates.append(candidate(place, contrast=contrast * 0.9))
        built.append(
            BallEvidence(
                frame_index=index,
                timestamp_s=index / fps,
                candidates=tuple(candidates),
                refusal=None if candidates else BallRefusal.NO_CANDIDATE,
            )
        )
    return built


def phases_over(frames: int, *, fps: float = FPS) -> SwingPhases:
    """A detected swing tiling `frames`, for the gate tests.

    Built rather than detected, because these tests are about what the gate
    *does* with phase boundaries and not about where Phase 4 puts them.
    """
    quarters = [frames * n // 4 for n in range(5)]
    order = (
        SwingPhase.ADDRESS,
        SwingPhase.BACKSWING,
        SwingPhase.DOWNSWING,
        SwingPhase.FOLLOW_THROUGH,
    )
    confidence = EventConfidence(overall=0.9, margin=0.9, visibility=1.0, resolution=1.0)
    return SwingPhases(
        detected=True,
        events=[
            DetectedEvent(
                event=event,
                frame_index=quarters[index + 1],
                timestamp_s=quarters[index + 1] / fps,
                confidence=confidence,
                methodology="constructed by the test",
            )
            for index, event in enumerate(
                (SwingEvent.TAKEAWAY, SwingEvent.TOP, SwingEvent.IMPACT, SwingEvent.FINISH)
            )
        ],
        phases=[
            DetectedPhase(
                phase=phase,
                start_frame=quarters[index],
                end_frame=quarters[index + 1],
                start_s=quarters[index] / fps,
                end_s=quarters[index + 1] / fps,
                duration_s=(quarters[index + 1] - quarters[index]) / fps,
                confidence=0.9,
            )
            for index, phase in enumerate(order)
        ],
        hand=HandSignalInfo(
            source=HandSource.MIDPOINT,
            valid_frames=frames,
            total_frames=frames,
            peak_speed=1.0,
            travel=0.5,
            torso_length=TORSO,
            travel_ratio=2.0,
        ),
        frames=frames,
        config=PhaseConfig(),
    )


def track(frames: list[BallEvidence], **kwargs: object) -> object:
    config = kwargs.pop("config", None) or BallConfig()
    phases = kwargs.pop("phases", None)
    return track_ball(frames, config, phases=phases, torso_length=TORSO)  # type: ignore[arg-type]


# --- the measurement ------------------------------------------------------


def test_departure_is_the_frame_after_the_last_sighting() -> None:
    """The instant is a bracket between two frames, and both ends are reported."""
    result = track(evidence(200, present=lambda i: i < 150), phases=phases_over(200))

    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None
    assert departure.last_seen_frame == 149
    assert departure.first_absent_frame == 150
    assert departure.frame_index == 150


def test_the_bracket_is_one_frame_interval_wide() -> None:
    """The phase's central claim: the uncertainty does not depend on the footage.

    Asserted at two frame rates because the point is not that it is small but
    that it is exactly one interval at any rate -- nothing here smooths, so
    nothing here has a resolution that degrades.
    """
    for fps in (30.0, 240.0):
        result = track(
            evidence(200, present=lambda i: i < 150, fps=fps), phases=phases_over(200, fps=fps)
        )
        departure = result.departure  # type: ignore[attr-defined]
        assert departure is not None
        assert departure.interval_s == pytest.approx(1.0 / fps, rel=1e-6)


def test_a_ball_still_there_at_the_end_yields_no_instant() -> None:
    """A practice swing. The ball is found, nothing departs, and nothing is claimed."""
    result = track(evidence(200), phases=phases_over(200))

    assert result.established is not None  # type: ignore[attr-defined]
    assert result.departure is None  # type: ignore[attr-defined]
    assert any("can be struck at" in note for note in result.warnings)  # type: ignore[attr-defined]


def test_frames_after_the_departure_are_gone_not_missing() -> None:
    """`GONE` and `NO_CANDIDATE` are the same picture and opposite facts."""
    result = track(evidence(200, present=lambda i: i < 150), phases=phases_over(200))

    after = [frame for frame in result.frames if frame.frame_index >= 150]  # type: ignore[attr-defined]
    assert after
    assert all(frame.refusal is BallRefusal.GONE for frame in after)
    assert result.refusals[BallRefusal.GONE] == len(after)  # type: ignore[attr-defined]


# --- what it refuses ------------------------------------------------------


def test_a_ball_that_comes_back_was_occluded_not_struck() -> None:
    """Permanence is the only factor that can tell those apart, so it must.

    Something crosses the tee at 120 and moves on at 150; the ball is then struck
    at 200. The mid-clip gap looks exactly like a departure from its left-hand
    side, and taking it would put impact 80 frames early.
    """
    result = track(
        evidence(260, present=lambda i: i < 120 or 150 <= i < 200), phases=phases_over(260)
    )

    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None
    assert departure.first_absent_frame == 200, "the occlusion is not the departure"


def test_a_ball_that_comes_back_and_stays_yields_no_instant() -> None:
    """The same occlusion with no strike after it. Nothing left, so nothing is claimed."""
    result = track(evidence(260, present=lambda i: not (120 <= i < 150)), phases=phases_over(260))

    assert result.established is not None  # type: ignore[attr-defined]
    assert result.departure is None  # type: ignore[attr-defined]


def test_a_departure_before_the_top_is_not_an_impact() -> None:
    """A ball that rolls off a tee departs exactly like one that is hit."""
    frames = 200
    phases = phases_over(frames)
    top = phases.event(SwingEvent.TOP)
    assert top is not None

    result = track(evidence(frames, present=lambda i: i < top.frame_index - 10), phases=phases)

    assert result.departure is None  # type: ignore[attr-defined]
    assert any("before the top of the backswing" in note for note in result.warnings)  # type: ignore[attr-defined]


def test_a_single_late_speck_cannot_move_the_answer() -> None:
    """One false positive in the follow-through is not a run, so it is not a departure.

    Without the "long enough to be a track" rule this moves the reported instant
    by fifty frames, and nothing in the output would say so.
    """
    result = track(evidence(260, present=lambda i: i < 150 or i == 200), phases=phases_over(260))

    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None
    assert departure.first_absent_frame == 150


def test_an_isolated_flicker_is_not_a_ball() -> None:
    """Establishment needs frames, because a flicker seen from one side is a departure."""
    result = track(evidence(200, present=lambda i: i < 3), phases=phases_over(200))

    assert result.established is None  # type: ignore[attr-defined]
    assert result.departure is None  # type: ignore[attr-defined]
    assert not result.detected if hasattr(result, "detected") else True  # type: ignore[attr-defined]


# --- the identification ---------------------------------------------------


def test_the_object_that_departs_is_the_one_identified() -> None:
    """Two stationary candidates, one of which leaves. That one is the ball.

    The distractor is placed outside the drift bound -- `max_drift_torso` is in
    torso lengths and positions are in frame widths, so the bound here is
    0.10 * 0.25 = 0.025 -- and this test is the reason that conversion exists.
    """
    marker = (BALL[0] + 0.06, BALL[1])
    result = track(
        evidence(200, present=lambda i: i < 150, extra=[marker]), phases=phases_over(200)
    )

    established = result.established  # type: ignore[attr-defined]
    assert established is not None
    assert established.x == pytest.approx(BALL[0], abs=1e-6)
    assert established.departed is True

    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None
    assert departure.first_absent_frame == 150


def test_a_rival_that_also_departs_is_reported_rather_than_hidden() -> None:
    """Nothing in one view says which of two vanishing objects was struck."""
    marker = (BALL[0] + 0.06, BALL[1])
    result = track(
        evidence(200, present=lambda i: i < 150, extra=[marker], extra_until=150),
        phases=phases_over(200),
    )

    assert result.departure is not None  # type: ignore[attr-defined]
    assert any("emptied and stayed empty" in note for note in result.warnings)  # type: ignore[attr-defined]


def test_established_margin_falls_only_for_a_rival_that_also_departs() -> None:
    """The margin is measured over what the choice was actually made over.

    A tee marker that sits there all clip is agreed on by exactly as many frames
    as the ball, so a margin scored against every stationary candidate would call
    this a coin toss -- when the ball was in fact singled out unambiguously by
    being the only thing that left. It is the rival that *also* leaves which the
    clip cannot resolve, and only that one drives the margin down.

    Every frame's own margin is 1.0 in both cases, because the rival is nowhere
    near the ball and so never competed for any single frame's choice. Only the
    clip-level margin can see either situation, which is why the two exist
    separately.
    """
    marker = (BALL[0] + 0.06, BALL[1])
    staying = track(
        evidence(200, present=lambda i: i < 150, extra=[marker]), phases=phases_over(200)
    )
    departing = track(
        evidence(200, present=lambda i: i < 150, extra=[marker], extra_until=150),
        phases=phases_over(200),
    )

    assert staying.established.margin == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert staying.established.runner_up_distance_torso is None  # type: ignore[attr-defined]

    assert departing.established.margin < 0.5  # type: ignore[attr-defined]
    assert departing.established.runner_up_distance_torso == pytest.approx(  # type: ignore[attr-defined]
        0.06 / TORSO, rel=1e-6
    )

    for result in (staying, departing):
        observed = [f.ball for f in result.frames if f.ball is not None]  # type: ignore[attr-defined]
        assert observed
        assert all(ball.confidence.margin == pytest.approx(1.0) for ball in observed)


def test_a_candidate_beyond_the_drift_bound_is_not_the_ball() -> None:
    """A shoe entering the region is refused as MOVED rather than followed."""
    shifted = (BALL[0] + 0.05, BALL[1])
    frames = evidence(200)
    for index in range(100, 110):
        frames[index] = BallEvidence(
            frame_index=index,
            timestamp_s=index / FPS,
            candidates=(candidate(shifted),),
        )

    result = track(frames, phases=phases_over(200))
    moved = [f for f in result.frames if f.refusal is BallRefusal.MOVED]  # type: ignore[attr-defined]
    assert len(moved) == 10


# --- the two confidences, and their independence --------------------------


def test_departure_confidence_ignores_how_good_each_frame_looked() -> None:
    """Halving every frame's contrast must not move the departure's confidence.

    The circularity guard, asserted directly: `DepartureConfidence` is scored on
    presence, tail strength *relative to the clip's own typical strength*, and
    permanence -- none of which a uniform change in contrast touches. If this
    fails, the two confidences have been wired together and the identification is
    arguing with itself.
    """
    strong = track(evidence(200, present=lambda i: i < 150, contrast=0.9), phases=phases_over(200))
    weak = track(evidence(200, present=lambda i: i < 150, contrast=0.4), phases=phases_over(200))

    assert strong.departure is not None and weak.departure is not None  # type: ignore[attr-defined]
    assert weak.departure.confidence.overall == pytest.approx(  # type: ignore[attr-defined]
        strong.departure.confidence.overall  # type: ignore[attr-defined]
    )
    assert weak.quality.median_contrast < strong.quality.median_contrast  # type: ignore[attr-defined]


def test_abruptness_falls_when_the_ball_dims_before_it_goes() -> None:
    """A ball being covered is not a ball being struck, and the score says which."""
    fading = evidence(200, present=lambda i: i < 150)
    for index in range(147, 150):
        fading[index] = BallEvidence(
            frame_index=index,
            timestamp_s=index / FPS,
            candidates=(candidate(BALL, contrast=0.3),),
        )

    result = track(fading, phases=phases_over(200))
    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None
    assert departure.confidence.abruptness < 0.5
    assert any("usual strength" in note for note in result.warnings)  # type: ignore[attr-defined]


def test_permanence_falls_when_the_clip_ends_before_the_absence_is_verified() -> None:
    """A check that could not be made scores as not made, rather than as passed."""
    short = track(evidence(154, present=lambda i: i < 150), phases=phases_over(154))
    long = track(evidence(260, present=lambda i: i < 150), phases=phases_over(260))

    assert short.departure is not None and long.departure is not None  # type: ignore[attr-defined]
    assert short.departure.confidence.permanence < 0.5  # type: ignore[attr-defined]
    assert long.departure.confidence.permanence == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert any("absence was verified" in note for note in short.warnings)  # type: ignore[attr-defined]


# --- coverage and reporting ------------------------------------------------


def test_coverage_is_measured_before_the_departure_only() -> None:
    """After it the ball is correctly absent, so counting those frames is wrong.

    A clip-wide rate would report this perfect track at 75%, and a longer
    follow-through would report the same track lower still.
    """
    result = track(evidence(200, present=lambda i: i < 150), phases=phases_over(200))

    assert result.pre_departure_coverage == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert result.observed_frames == 150  # type: ignore[attr-defined]


def test_an_ungated_departure_says_it_was_ungated() -> None:
    """With no swing detected there is nothing to check the instant against."""
    result = track(evidence(200, present=lambda i: i < 150), phases=None)

    departure = result.departure  # type: ignore[attr-defined]
    assert departure is not None, "the observation still stands without a swing around it"
    assert departure.phase is None
    assert any("could not be checked" in note for note in result.warnings)  # type: ignore[attr-defined]


def test_every_frame_is_kept_whether_or_not_it_carried_a_ball() -> None:
    """Frame index and position in the sequence must stay the same number."""
    result = track(evidence(200, present=lambda i: i < 150), phases=phases_over(200))

    assert len(result.frames) == 200  # type: ignore[attr-defined]
    assert [frame.frame_index for frame in result.frames] == list(range(200))  # type: ignore[attr-defined]
    assert all(  # type: ignore[attr-defined]
        (frame.ball is None) != (frame.refusal is None) for frame in result.frames
    )
