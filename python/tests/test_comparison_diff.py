"""The gates a difference passes before it may be called one.

Built on metric sets constructed by hand, for the reason `synthetic_coaching`
gives: what these tests need is the ability to put two values an exact distance
apart, and that is a property of the fixture rather than of the engine. Driving
two synthetic bodies through pose estimation to arrive at a difference chosen at
the top would test the chain again and make a gate's behaviour depend on whether
a polynomial fit reproduced a constructed angle to three decimal places.
"""

from __future__ import annotations

import math

import pytest

from analyzer.comparison.diff import camera_agreement, camera_term_deg, compare_metrics
from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.comparison import (
    ComparisonConfig,
    DifferenceRefusal,
    Direction,
    MetricDifference,
)
from analyzer.contracts.metrics import (
    CameraView,
    MetricBasis,
    MetricName,
    MetricSet,
    MetricUnit,
    RefusedMetric,
)
from analyzer.contracts.phases import SwingEvent
from tests.synthetic_coaching import metric, metric_set, tempo_metrics

INTERVAL = 1.0 / 240.0
CONFIG = ComparisonConfig()


def rotation_set(
    value: float,
    *,
    uncertainty: float = 1.0,
    view: CameraView = CameraView.FACE_ON,
    confidence: float = 0.9,
    openness: float = 1.0,
    shoulder_span_ratio: float = 0.83,
    calibration: CalibrationStatus = CalibrationStatus.NONE,
) -> MetricSet:
    """One foreshortened rotation at the top, with an uncertainty to bracket with."""
    return metric_set(
        [
            metric(
                MetricName.SHOULDER_TURN,
                value,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.TOP,
                uncertainty=uncertainty,
                confidence=confidence,
                view=view,
            )
        ],
        view=view,
        openness=openness,
        shoulder_span_ratio=shoulder_span_ratio,
        calibration=calibration,
    )


def run(reference: MetricSet, target: MetricSet, config: ComparisonConfig = CONFIG):
    camera = camera_agreement(reference, target, config)
    return compare_metrics(
        reference,
        target,
        camera,
        config,
        reference_interval_s=INTERVAL,
        target_interval_s=INTERVAL,
    )


# --- the camera term ------------------------------------------------------


def test_two_square_cameras_contribute_nothing() -> None:
    assert camera_term_deg(50.0, 0.0, 0.0) == pytest.approx(0.0)


def test_an_unknown_azimuth_contributes_nothing_rather_than_a_guess() -> None:
    assert camera_term_deg(50.0, None, 8.0) == 0.0
    assert camera_term_deg(50.0, 8.0, None) == 0.0


def test_moving_the_camera_round_the_player_moves_the_reported_rotation() -> None:
    """The measurement this phase exists for, as arithmetic.

    A fifty-degree turn seen from twenty degrees off broadside does not read as
    fifty. The term says how far it can read out, and it is large enough to
    swamp most differences anybody would want to read from this system.
    """
    assert camera_term_deg(50.0, 0.0, 10.0) == pytest.approx(11.07, abs=0.05)
    assert camera_term_deg(50.0, 0.0, 20.0) == pytest.approx(27.16, abs=0.05)


def test_equal_azimuths_do_not_cancel_because_the_sign_is_unknown() -> None:
    """Two cameras five degrees off may be together or ten degrees apart.

    Foreshortening is identical from either side of broadside, so nothing here
    separates them and the term has to assume the worse of the two. A term that
    cancelled would be assuming the answer.
    """
    assert camera_term_deg(50.0, 5.0, 5.0) > 5.0


def test_the_term_grows_as_the_cameras_separate() -> None:
    growing = [camera_term_deg(50.0, 0.0, angle) for angle in (0.0, 5.0, 10.0, 15.0, 20.0)]
    assert growing == sorted(growing)


# --- camera agreement -----------------------------------------------------


def test_matching_spans_from_one_view_are_consistent() -> None:
    agreement = camera_agreement(rotation_set(50.0), rotation_set(55.0), CONFIG)

    assert agreement.consistent
    assert agreement.span_disagreement == pytest.approx(0.0)


def test_a_moved_camera_is_measured_from_the_address_span() -> None:
    agreement = camera_agreement(
        rotation_set(50.0, shoulder_span_ratio=0.83),
        rotation_set(50.0, shoulder_span_ratio=0.60),
        CONFIG,
    )

    assert not agreement.consistent
    assert agreement.span_disagreement == pytest.approx(0.23 / 0.715, abs=1e-6)


def test_the_azimuth_is_reported_but_does_not_decide() -> None:
    """Openness near one is where the arccosine is worst, and it is not gated on."""
    agreement = camera_agreement(
        rotation_set(50.0, openness=0.99), rotation_set(50.0, openness=1.0), CONFIG
    )

    assert agreement.consistent
    assert agreement.reference_azimuth_deg is not None
    assert agreement.reference_azimuth_deg > 5.0


def test_azimuth_separation_is_the_sum_not_the_difference() -> None:
    agreement = camera_agreement(
        rotation_set(50.0, openness=0.98), rotation_set(50.0, openness=0.98), CONFIG
    )

    assert agreement.azimuth_separation_deg == pytest.approx(
        2 * (agreement.reference_azimuth_deg or 0.0)
    )


def test_an_unmeasurable_openness_is_null_rather_than_clamped() -> None:
    agreement = camera_agreement(
        rotation_set(50.0, openness=1.12), rotation_set(50.0, openness=1.0), CONFIG
    )

    assert agreement.reference_azimuth_deg is None
    assert agreement.azimuth_separation_deg is None


def test_two_unknown_views_are_not_a_match() -> None:
    agreement = camera_agreement(
        rotation_set(50.0, view=CameraView.UNKNOWN),
        rotation_set(50.0, view=CameraView.UNKNOWN),
        CONFIG,
    )

    assert not agreement.consistent


def test_a_clip_with_no_view_estimate_reports_nulls() -> None:
    blank = metric_set([], computed=False)
    blank.view = None

    agreement = camera_agreement(blank, rotation_set(50.0), CONFIG)

    assert not agreement.consistent
    assert agreement.reference_span is None
    assert agreement.span_disagreement is None


# --- the gates, in order --------------------------------------------------


def test_a_difference_that_clears_its_bracket_is_reported() -> None:
    differences, refused, considered = run(
        rotation_set(40.0, uncertainty=1.0), rotation_set(60.0, uncertainty=1.0)
    )

    assert considered == 1
    assert refused == []
    (entry,) = differences
    assert entry.name is MetricName.SHOULDER_TURN
    assert entry.difference == pytest.approx(20.0)
    assert entry.direction is Direction.HIGHER
    assert entry.bracket == pytest.approx(2.0)
    assert entry.margin == pytest.approx(18.0)


def test_direction_is_lower_when_the_target_measures_less() -> None:
    differences, _, _ = run(rotation_set(60.0), rotation_set(40.0))

    assert differences[0].direction is Direction.LOWER
    assert differences[0].difference == pytest.approx(-20.0)


def test_the_bracket_is_the_sum_of_the_two_and_not_their_quadrature() -> None:
    """Two recordings share no systematic error, so a sum is the bound and a
    quadrature is not. Three and four give seven here, never five."""
    differences, _, _ = run(rotation_set(0.0, uncertainty=3.0), rotation_set(40.0, uncertainty=4.0))

    assert differences[0].bracket == pytest.approx(7.0)
    assert [term.value for term in differences[0].terms] == pytest.approx([3.0, 4.0])


def test_a_difference_inside_the_bracket_is_unresolved_not_agreement() -> None:
    _, refused, _ = run(rotation_set(50.0, uncertainty=3.0), rotation_set(52.0, uncertainty=3.0))

    (entry,) = refused
    assert entry.refusal is DifferenceRefusal.UNRESOLVED
    assert entry.reference_value == pytest.approx(50.0)
    assert entry.target_value == pytest.approx(52.0)
    assert entry.bracket == pytest.approx(6.0)
    assert "cannot tell them apart" in entry.reason


def test_an_exactly_equal_pair_is_unresolved_rather_than_a_zero_difference() -> None:
    _, refused, _ = run(rotation_set(50.0), rotation_set(50.0))

    assert refused[0].refusal is DifferenceRefusal.UNRESOLVED


def test_the_camera_term_enters_a_foreshortened_bracket() -> None:
    clean = run(rotation_set(30.0), rotation_set(70.0))[0][0]
    tilted = run(
        rotation_set(30.0, openness=0.997),
        rotation_set(70.0, openness=0.997, shoulder_span_ratio=0.80),
    )[0][0]

    assert len(clean.terms) == 2
    assert len(tilted.terms) == 3
    assert tilted.terms[2].source == "camera azimuth"
    assert tilted.bracket > clean.bracket


def test_the_camera_term_can_refuse_a_rotation_difference_on_its_own() -> None:
    """Two cameras twenty degrees apart wipe out a forty-degree difference.

    The whole argument for the term being computed rather than assumed away: the
    two measurements are fine, the arithmetic is fine, and the difference is not
    attributable to the swing.
    """
    _, refused, _ = run(rotation_set(30.0, openness=0.94), rotation_set(70.0, openness=0.94))

    assert refused[0].refusal is DifferenceRefusal.UNRESOLVED
    assert "cameras not having stood in the same place" in refused[0].reason


def test_different_views_refuse_a_projected_quantity() -> None:
    _, refused, _ = run(
        rotation_set(50.0, view=CameraView.FACE_ON),
        rotation_set(20.0, view=CameraView.DOWN_THE_LINE),
    )

    assert refused[0].refusal is DifferenceRefusal.VIEW_MISMATCH


def test_different_views_do_not_refuse_a_timing_quantity() -> None:
    """A stopwatch does not care where the camera stood, so the view gate skips it."""
    differences, refused, _ = run(
        tempo_metrics(0.8, 0.25, view=CameraView.FACE_ON),
        tempo_metrics(0.8, 0.40, view=CameraView.DOWN_THE_LINE),
    )

    assert {entry.name for entry in differences} == {
        MetricName.DOWNSWING_DURATION,
        MetricName.TEMPO_RATIO,
    }
    assert not any(entry.refusal is DifferenceRefusal.VIEW_MISMATCH for entry in refused)


def test_a_moved_camera_refuses_before_a_missing_metric_does() -> None:
    """Gate order, and the reason for it.

    A reader told "one clip did not measure this" re-films the missing clip; if
    the cameras were in different places, the re-filmed clip is then refused for
    the camera and they have shot footage to learn what the first answer could
    have told them.
    """
    reference = rotation_set(50.0, shoulder_span_ratio=0.83)
    target = metric_set([], shoulder_span_ratio=0.55)

    _, refused, _ = run(reference, target)

    assert refused[0].refusal is DifferenceRefusal.CAMERA_MOVED


def test_a_mismatched_calibration_refuses() -> None:
    _, refused, _ = run(
        rotation_set(50.0, calibration=CalibrationStatus.NONE),
        rotation_set(70.0, calibration=CalibrationStatus.INTRINSICS),
    )

    assert refused[0].refusal is DifferenceRefusal.CALIBRATION_MISMATCH


def test_a_supplied_timebase_refuses_seconds_and_keeps_the_ratio() -> None:
    differences, refused, _ = run(
        tempo_metrics(0.8, 0.25, slow_motion_factor=7.0),
        tempo_metrics(1.2, 0.30, slow_motion_factor=7.0),
    )

    assert {entry.refusal for entry in refused} == {DifferenceRefusal.SUPPLIED_TIMEBASE}
    assert {entry.name for entry in refused} == {
        MetricName.BACKSWING_DURATION,
        MetricName.DOWNSWING_DURATION,
    }
    assert [entry.name for entry in differences] == [MetricName.TEMPO_RATIO]


def test_a_quantity_only_one_clip_produced_cites_the_other_clips_reason() -> None:
    reference = rotation_set(50.0)
    target = metric_set(
        [],
        refused=[
            RefusedMetric(
                name=MetricName.SHOULDER_TURN,
                event=SwingEvent.TOP,
                reason="The shoulders were never tracked through the address phase.",
            )
        ],
    )

    _, refused, _ = run(reference, target)

    assert refused[0].refusal is DifferenceRefusal.MISSING
    assert "never tracked" in refused[0].reason


def test_a_weak_measurement_refuses_before_the_arithmetic_runs() -> None:
    _, refused, _ = run(rotation_set(10.0, confidence=0.1), rotation_set(80.0))

    assert refused[0].refusal is DifferenceRefusal.LOW_CONFIDENCE
    assert refused[0].bracket is None


def test_a_quantity_with_no_uncertainty_anywhere_refuses() -> None:
    """Phase 6 quantifies uncertainty for the foreshortened rotations and nothing
    else, so this is the ordinary outcome for a projected angle."""
    _, refused, _ = run(rotation_set(50.0, uncertainty=None), rotation_set(80.0, uncertainty=None))

    assert refused[0].refusal is DifferenceRefusal.NO_BRACKET
    assert refused[0].reference_value == pytest.approx(50.0)


def test_a_duration_gets_its_bracket_from_the_clock() -> None:
    differences, _, _ = run(tempo_metrics(0.8, 0.25), tempo_metrics(1.4, 0.25))

    backswing = next(entry for entry in differences if entry.name is MetricName.BACKSWING_DURATION)
    assert backswing.bracket == pytest.approx(2 * INTERVAL)
    assert backswing.difference == pytest.approx(0.6)


def test_a_finer_clock_on_one_clip_narrows_the_bracket() -> None:
    reference, target = tempo_metrics(0.8, 0.25), tempo_metrics(0.82, 0.25)
    camera = camera_agreement(reference, target, CONFIG)

    coarse = compare_metrics(
        reference,
        target,
        camera,
        CONFIG,
        reference_interval_s=1 / 30,
        target_interval_s=1 / 30,
    )
    fine = compare_metrics(
        reference,
        target,
        camera,
        CONFIG,
        reference_interval_s=1 / 240,
        target_interval_s=1 / 240,
    )

    assert not any(entry.name is MetricName.BACKSWING_DURATION for entry in coarse[0])
    assert any(entry.name is MetricName.BACKSWING_DURATION for entry in fine[0])


def test_every_difference_carries_the_frames_both_values_came_from() -> None:
    differences, _, _ = run(rotation_set(30.0), rotation_set(70.0))

    assert differences[0].reference_frames
    assert differences[0].target_frames


def test_nothing_is_compared_that_neither_clip_produced() -> None:
    """A quantity neither recording measured is refused one layer down, where the
    reason is specific to that recording. Restating it here would be the same fact
    twice, worse worded the second time."""
    _, _, considered = run(metric_set([]), metric_set([]))

    assert considered == 0


def test_a_margin_is_always_positive() -> None:
    differences, _, _ = run(rotation_set(30.0), rotation_set(70.0))

    assert all(entry.margin > 0.0 for entry in differences)
    assert all(math.isfinite(entry.margin) for entry in differences)


def test_no_difference_carries_a_verdict_about_which_value_is_better() -> None:
    """The absence Phase 13 built into `Finding`, asserted rather than assumed.

    A score would need a scale relating degrees of shoulder turn to seconds of
    tempo, and a "better" would need to know which direction of each quantity is
    desirable -- which is exactly what Phase 13 found no source supplies.
    """
    forbidden = {"score", "severity", "better", "worse", "grade", "rank", "rating", "quality"}
    fields = set(MetricDifference.model_fields)

    assert not (fields & forbidden)
    assert {member.value for member in Direction} == {"higher", "lower"}
