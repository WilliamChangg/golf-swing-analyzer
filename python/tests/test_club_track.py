"""The temporal pass: what the tracker adds, and what it refuses to add.

Most of these run on **constructed evidence** rather than on rendered frames.
That is the point of the seam: `track_shafts` takes candidates and timestamps, so
a test can state exactly what the image offered -- a club and a door frame of
equal support, a candidate implying an impossible rotation, a gap in the middle
of a swing -- and assert what the tracker does with it. Driving those cases
through a renderer would mean tuning pixels until the detector produced the
situation under test, which tests the fixture.

The ones that matter most assert **refusals**, because the exit criterion for
this phase is that low confidence emits nothing:

* that a prediction scores candidates and never supplies one, so a frame with
  nothing acceptable leaves a hole rather than a smooth interpolation;
* that two equally good readings of the same pixels are refused rather than
  resolved by a coin toss the confidence would then describe as a measurement;
* that a single accepted frame with nothing agreeing around it is discarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.club import track_shafts, unwrap_deg
from analyzer.club.track import CandidateView, FrameEvidence
from analyzer.contracts.club import ClubConfig, ShaftRefusal
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


def candidate(angle_deg: float, *, support: float = 1.0, length: float = 0.3) -> CandidateView:
    """A candidate at a given direction, with the tip placed to match it."""
    radians = np.radians(angle_deg)
    tip = (0.5 + length * np.cos(radians), 0.5 + length * np.sin(radians))
    return CandidateView(
        tip=tip,
        tip_px=(tip[0] * 1000.0, (1.0 - tip[1]) * 1000.0),
        angle_deg=angle_deg,
        length=length,
        length_torso=length / TORSO,
        support=support,
    )


def evidence(
    angles: list[list[float]] | list[float],
    *,
    supports: list[list[float]] | None = None,
    grip: bool | list[bool] = True,
    fps: float = FPS,
) -> list[FrameEvidence]:
    """A clip's worth of evidence from a list of per-frame candidate directions.

    A bare list of floats is one candidate per frame; a list of lists is several.
    """
    frames: list[FrameEvidence] = []
    grips = grip if isinstance(grip, list) else [grip] * len(angles)
    for index, item in enumerate(angles):
        directions = item if isinstance(item, list) else [item]
        strengths = supports[index] if supports else [1.0] * len(directions)
        frames.append(
            FrameEvidence(
                frame_index=index,
                timestamp_s=index / fps,
                grip=(0.5, 0.5) if grips[index] else None,
                candidates=tuple(
                    candidate(angle, support=strength)
                    for angle, strength in zip(directions, strengths, strict=True)
                ),
                refusal=(
                    None
                    if directions
                    else (ShaftRefusal.NO_CANDIDATE if grips[index] else ShaftRefusal.NO_GRIP)
                ),
            )
        )
    return frames


def phases_over(frames: int, *, fps: float = FPS) -> SwingPhases:
    """A detected swing tiling `frames`, for the coverage tests.

    Built rather than detected, because these tests are about what coverage
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


def tracked(result: object) -> list[int]:
    return [frame.frame_index for frame in result.frames if frame.detected]  # type: ignore[attr-defined]


class TestFollowingAClub:
    def test_a_steadily_turning_shaft_is_tracked_throughout(self) -> None:
        angles = [float(index * 2) for index in range(40)]
        result = track_shafts(evidence(angles), ClubConfig())
        assert result.tracked_frames == 40
        assert result.unanchored_frames == 1, "one seed, and everything else grown from it"

    def test_the_reported_direction_is_the_candidate_that_was_chosen(self) -> None:
        result = track_shafts(evidence([10.0, 12.0, 14.0]), ClubConfig(min_track_frames=1))
        assert [frame.shaft.angle_deg for frame in result.frames] == [10.0, 12.0, 14.0]

    def test_an_angular_rate_is_measured_from_the_unwrapped_series(self) -> None:
        """A swing carries the shaft past 180 degrees, once, in the downswing.

        Differencing the reported angles there gives a jump of 360 and a rate of
        tens of thousands of degrees per second -- inside the only phase anyone
        cares about. The rate therefore comes from the unwrapped series, and this
        pins it: the fixture turns at a constant 240 deg/s straight through the
        discontinuity.
        """
        step = 2.0
        angles = [((170.0 + index * step + 180.0) % 360.0) - 180.0 for index in range(20)]
        assert any(abs(angles[i + 1] - angles[i]) > 180.0 for i in range(len(angles) - 1))

        result = track_shafts(evidence(angles), ClubConfig())
        rates = [
            frame.shaft.angular_rate_deg_s
            for frame in result.frames
            if frame.shaft and frame.shaft.angular_rate_deg_s is not None
        ]
        assert rates
        assert all(rate == pytest.approx(step * FPS, rel=0.05) for rate in rates)

    def test_a_rate_is_not_measured_across_a_gap(self) -> None:
        """Because nothing says how many half-turns happened while unobserved.

        Measured on the synthetic swing before this rule existed: the nearest
        tracked neighbour across a three-frame hole reported 8,050 deg/s on a
        club turning at 700.
        """
        angles: list[list[float]] = [[float(index * 2)] for index in range(20)]
        for index in (9, 10, 11):
            angles[index] = []
        result = track_shafts(evidence(angles), ClubConfig())
        by_index = {frame.frame_index: frame for frame in result.frames}
        assert by_index[8].shaft is not None
        assert by_index[8].shaft.angular_rate_deg_s is not None
        assert by_index[12].shaft is not None
        # Frame 12's left neighbour is missing, so the rate is one-sided rather
        # than taken across the hole -- and it must still be the real rate.
        assert by_index[12].shaft.angular_rate_deg_s == pytest.approx(2.0 * FPS, rel=0.05)


class TestTheThreeFactors:
    def test_two_equally_good_readings_are_refused_rather_than_guessed(self) -> None:
        """**The refusal this phase exists for.**

        A door frame through the hands and a club pointing the other way are the
        same pixels. With no temporal information neither is preferable, so the
        frame emits nothing -- rather than picking one and reporting a confidence
        that would describe the coin toss as a measurement.
        """
        result = track_shafts(evidence([[45.0, -135.0]] * 10), ClubConfig())
        assert result.tracked_frames == 0
        assert result.refusals == {ShaftRefusal.AMBIGUOUS: 10}

    def test_a_prediction_separates_what_evidence_alone_cannot(self) -> None:
        """The same ambiguity, with tracked neighbours either side of it.

        This is the tracker earning what the detector cannot supply: the club has
        been turning steadily, so one of the two readings continues that motion
        and the other reverses it.
        """
        angles: list[list[float]] = [[float(index * 2)] for index in range(20)]
        angles[10] = [20.0, -160.0]  # the true direction, and its opposite
        result = track_shafts(evidence(angles), ClubConfig())
        by_index = {frame.frame_index: frame for frame in result.frames}
        assert by_index[10].shaft is not None
        assert by_index[10].shaft.angle_deg == pytest.approx(20.0)
        # And there is no runner-up, because the opposite reading was not merely
        # outranked: turning 180 degrees in one frame at 120 fps is past what a
        # club can do, so it is excluded from the pool before anything is scored.
        # That is a harder rejection than a low continuity and it is the right
        # one -- a candidate implying an impossible rotation is not a worse
        # explanation of the same object, it is a different object.
        assert by_index[10].shaft.runner_up_angle_deg is None

    def test_a_stronger_but_stationary_rival_does_not_win(self) -> None:
        """A door frame with better edges than a smeared club still loses.

        And the margin stays high while it does, which is why margin is measured
        on the same score the choice was made on. Scoring it on evidence alone
        would report this frame as ambiguous and refuse a correct detection at
        exactly the point in the swing where this phase is hardest.
        """
        angles: list[list[float]] = [[float(index * 3)] for index in range(20)]
        supports: list[list[float]] = [[1.0] for _ in range(20)]
        angles[10] = [30.0, 90.0]  # the club, and a stationary vertical
        supports[10] = [0.6, 1.0]  # and the vertical has the better edges

        result = track_shafts(evidence(angles, supports=supports), ClubConfig())
        chosen = result.frames[10].shaft
        assert chosen is not None
        assert chosen.angle_deg == pytest.approx(30.0)
        assert chosen.confidence.support == pytest.approx(0.6)
        assert chosen.confidence.margin > 0.5

    def test_an_impossible_rotation_is_rejected_as_a_different_object(self) -> None:
        angles: list[list[float]] = [[float(index * 2)] for index in range(20)]
        angles[10] = [175.0]  # a quarter-turn away in one frame at 120 fps
        result = track_shafts(evidence(angles), ClubConfig())
        assert result.frames[10].refusal is ShaftRefusal.DISCONTINUOUS

    def test_weak_edges_alone_emit_nothing(self) -> None:
        config = ClubConfig()
        result = track_shafts(
            evidence([0.0] * 10, supports=[[config.min_support - 0.1]] * 10), config
        )
        assert result.tracked_frames == 0
        assert result.refusals == {ShaftRefusal.LOW_CONFIDENCE: 10}

    def test_confidence_is_the_product_of_the_three_factors(self) -> None:
        result = track_shafts(evidence([float(i * 2) for i in range(10)]), ClubConfig())
        for frame in result.frames:
            assert frame.shaft is not None
            confidence = frame.shaft.confidence
            assert confidence.overall == pytest.approx(
                confidence.support * confidence.margin * confidence.continuity
            )


class TestRefusingRatherThanFilling:
    def test_a_gap_stays_a_gap(self) -> None:
        """A prediction scores candidates. It never supplies one.

        The rule Phase 3 states about its gap policy, one layer up: a value
        produced where there is no observation is an invention however smooth it
        looks. A tracker that wrote its prediction into the output would produce
        a continuous track through a club nobody could see, and nothing
        downstream could tell which frames were measured.
        """
        angles: list[list[float]] = [[float(index * 2)] for index in range(30)]
        for index in range(12, 18):
            angles[index] = []
        result = track_shafts(evidence(angles), ClubConfig())
        assert tracked(result) == [*range(12), *range(18, 30)]
        assert result.refusals[ShaftRefusal.NO_CANDIDATE] == 6

    def test_a_frame_with_no_hands_is_refused_by_name(self) -> None:
        """Nothing about the club: the pose layer had no anchor to search from."""
        grips = [True] * 30
        for index in range(10, 15):
            grips[index] = False
        result = track_shafts(
            evidence([float(index * 2) for index in range(30)], grip=grips), ClubConfig()
        )
        assert result.refusals[ShaftRefusal.NO_GRIP] == 5

    def test_an_isolated_frame_is_discarded(self) -> None:
        """One line near the hands is not evidence that anything was followed."""
        angles: list[list[float]] = [[] for _ in range(20)]
        angles[7] = [45.0]
        result = track_shafts(evidence(angles), ClubConfig())
        assert result.tracked_frames == 0
        assert result.refusals[ShaftRefusal.ISOLATED] == 1

    def test_a_run_is_kept_once_it_is_long_enough(self) -> None:
        angles: list[list[float]] = [[] for _ in range(20)]
        for index in (7, 8, 9):
            angles[index] = [45.0 + index]
        result = track_shafts(evidence(angles), ClubConfig())
        assert tracked(result) == [7, 8, 9]

    def test_every_frame_is_present_whether_tracked_or_not(self) -> None:
        """`PoseFrame`'s decision, for `PoseFrame`'s reasons."""
        angles: list[list[float]] = [[float(index)] for index in range(15)]
        angles[3] = []
        result = track_shafts(evidence(angles), ClubConfig())
        assert [frame.frame_index for frame in result.frames] == list(range(15))
        assert all((frame.shaft is None) != (frame.refusal is None) for frame in result.frames)


class TestSeeding:
    def test_the_track_is_seeded_at_the_strongest_evidence_not_at_frame_zero(self) -> None:
        """Which is what lets the downswing be reached in good condition.

        Frame zero here is ambiguous -- two equally supported readings -- so a
        tracker that started there would start by refusing, and a tracker that
        started there and committed would be choosing between them blind.
        Seeding at the unambiguous middle and growing outward resolves frame
        zero from a neighbour instead.
        """
        angles: list[list[float]] = [
            [float(index * 2), float(index * 2) + 180.0] for index in range(3)
        ]
        angles += [[float(index * 2)] for index in range(3, 20)]
        result = track_shafts(evidence(angles), ClubConfig())
        assert result.tracked_frames == 20
        assert result.frames[0].shaft is not None
        assert result.frames[0].shaft.angle_deg == pytest.approx(0.0)

    def test_two_runs_separated_by_a_long_gap_are_seeded_separately(self) -> None:
        angles: list[list[float]] = [[float(index * 2)] for index in range(40)]
        for index in range(15, 32):  # far longer than max_prediction_window_s
            angles[index] = []
        result = track_shafts(evidence(angles), ClubConfig())
        assert result.unanchored_frames == 2, "one seed per run"


class TestDerivedQuantities:
    def test_the_club_head_is_reported_only_where_the_evidence_reached_it(self) -> None:
        """Two things shorten a segment and one view cannot separate them."""
        angles: list[list[float]] = [[float(index * 2)] for index in range(20)]
        frames = evidence(angles)
        short = frames[10]
        frames[10] = FrameEvidence(
            frame_index=short.frame_index,
            timestamp_s=short.timestamp_s,
            grip=short.grip,
            candidates=(candidate(20.0, length=0.15),),
        )
        result = track_shafts(frames, ClubConfig())
        assert result.frames[9].shaft is not None
        assert result.frames[9].shaft.reaches_head is True
        assert result.frames[10].shaft is not None
        assert result.frames[10].shaft.reaches_head is False

    def test_coverage_is_reported_per_phase(self) -> None:
        """**The gate.** A high clip-wide rate can contain no downswing at all."""
        angles: list[list[float]] = [[float(index * 2)] for index in range(40)]
        for index in range(20, 30):  # the whole downswing of `phases_over(40)`
            angles[index] = []
        result = track_shafts(evidence(angles), ClubConfig(), phases=phases_over(40))

        overall = result.tracked_frames / 40
        downswing = next(
            entry for entry in result.phase_coverage if entry.phase is SwingPhase.DOWNSWING
        )
        assert overall >= 0.7
        assert downswing.coverage == 0.0
        assert any("downswing" in note for note in result.warnings)

    def test_blur_is_bounded_by_the_frame_interval(self) -> None:
        """The most that can be said about smear from a file that records no shutter."""
        result = track_shafts(
            evidence([float(index * 2) for index in range(20)]),
            ClubConfig(),
            frame_interval_s=1.0 / FPS,
        )
        quality = result.quality
        assert quality is not None
        assert quality.max_blur_px == pytest.approx(quality.max_tip_speed_px_s / FPS)

    def test_no_track_reports_nothing_rather_than_zero(self) -> None:
        """Zero coverage and 'nothing was attempted' are different facts."""
        result = track_shafts(evidence([[] for _ in range(10)]), ClubConfig())
        assert result.quality is None
        assert result.tracked_frames == 0


class TestImpact:
    def _falling_then_rising(self, frames: int = 40) -> list[FrameEvidence]:
        """A club head that descends to a minimum and rises again, like a swing."""
        built: list[FrameEvidence] = []
        for index in range(frames):
            height = abs(index - 25) * 0.01
            tip = (0.5 + index * 0.001, 0.2 + height)
            built.append(
                FrameEvidence(
                    frame_index=index,
                    timestamp_s=index / FPS,
                    grip=(0.5, 0.5),
                    candidates=(
                        CandidateView(
                            tip=tip,
                            tip_px=(tip[0] * 1000.0, (1.0 - tip[1]) * 1000.0),
                            angle_deg=float(index * 2),
                            length=0.3,
                            length_torso=0.3 / TORSO,
                            support=1.0,
                        ),
                    ),
                )
            )
        return built

    def test_the_club_head_locates_impact_independently(self) -> None:
        result = track_shafts(self._falling_then_rising(), ClubConfig(), phases=phases_over(40))
        assert result.impact is not None
        assert result.impact.frame_index == 25
        assert result.impact.kinematic_frame == 30
        assert result.impact.delta_s == pytest.approx((25 - 30) / FPS)

    def test_it_is_refused_when_the_downswing_was_not_tracked(self) -> None:
        """The phase's own gate, applied to its own derived quantity.

        Without it, a clip whose club head blurred away through impact reports
        the lowest point of whatever survived -- which the synthetic swing put 38
        frames late, with both of the local checks passing.
        """
        built = self._falling_then_rising()
        for index in range(20, 30):
            built[index] = FrameEvidence(
                frame_index=index, timestamp_s=index / FPS, grip=(0.5, 0.5)
            )
        result = track_shafts(built, ClubConfig(), phases=phases_over(40))
        assert result.impact is None

    def test_it_is_refused_without_a_detected_swing(self) -> None:
        assert track_shafts(self._falling_then_rising(), ClubConfig()).impact is None


class TestUnwrap:
    def test_a_continuous_turn_is_unwrapped(self) -> None:
        wrapped = np.array([170.0, 175.0, -180.0, -175.0, -170.0])
        assert np.allclose(unwrap_deg(wrapped), [170.0, 175.0, 180.0, 185.0, 190.0])

    def test_phase_is_not_carried_across_a_gap(self) -> None:
        """A gap in the series is a gap in the evidence.

        Carrying the unwrap across it would invent a number of turns the clip
        never showed -- and it would do so silently, since the result still looks
        like a smooth angle series.
        """
        wrapped = np.array([170.0, 175.0, np.nan, -175.0, -170.0])
        result = unwrap_deg(wrapped)
        assert np.allclose(result[:2], [170.0, 175.0])
        assert np.isnan(result[2])
        assert np.allclose(result[3:], [-175.0, -170.0])

    def test_an_all_nan_series_survives(self) -> None:
        assert np.all(np.isnan(unwrap_deg(np.array([np.nan, np.nan]))))
