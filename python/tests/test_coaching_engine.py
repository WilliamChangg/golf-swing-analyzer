"""Every gate, in both directions, and the cross-check that ties the phase together.

Two kinds of test here, and the second kind is the one that matters.

The first kind drives each gate: a rule refuses for this reason and produces this
refusal code. Those are ordinary.

The second kind checks the properties that hold across the whole engine whatever
the rules are -- that every finding's own sentence survives the guard, that every
cited frame exists in the clip, that a gate which refuses everything today would
still let something through if the data changed. A gate that has never been shown
to pass anything is not a gate, it is a wall with a docstring.
"""

from __future__ import annotations

import dataclasses

import pytest

from analyzer.coaching import guard
from analyzer.coaching.engine import coach
from analyzer.coaching.registry import REGISTRY, RuleKind
from analyzer.contracts.coaching import (
    CoachingConfig,
    Comparison,
    FindingRefusal,
    ThresholdMethod,
    ThresholdSource,
)
from analyzer.contracts.metrics import (
    CameraView,
    MetricBasis,
    MetricName,
    MetricUnit,
    RefusedMetric,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase
from tests.synthetic_coaching import metric, metric_set, phases, tempo_metrics

# A 240 fps capture: fine enough that a tempo comparison resolves, which is what
# makes the firing tests possible at all. The 30 fps default cannot, and that is
# itself asserted below rather than worked around.
FAST = 1.0 / 240.0


def test_no_swing_refuses_every_rule_and_says_so_once() -> None:
    report = coach(metric_set([], computed=False), phases(detected=False))
    assert not report.computed
    assert not report.findings
    assert len(report.refused) == report.rules_considered == len(REGISTRY)
    assert {entry.refusal for entry in report.refused} == {FindingRefusal.NO_SWING}


def test_a_tempo_outside_the_band_is_a_finding_with_the_direction_on_it() -> None:
    """The one rule that fires on real footage, fired deliberately.

    0.60 over 0.30 is a tempo of 2:1, below the band's 2.43 floor by enough that
    a 240 fps clock cannot be the explanation.
    """
    report = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST))
    finding = report.finding("tempo.ratio")
    assert finding is not None
    assert finding.comparison is Comparison.BELOW
    assert finding.value == pytest.approx(2.0)
    assert finding.margin == pytest.approx(2.43 - 2.0)
    assert finding.source.citation.startswith("John Novosel")


def test_a_tempo_inside_the_band_is_also_a_finding() -> None:
    """Silence when nothing is wrong would make the report a selection.

    A reader who sees no finding about tempo cannot tell whether it was measured
    and found ordinary or refused for one of seven reasons.
    """
    finding = coach(tempo_metrics(0.70, 0.23), phases(interval_s=FAST)).finding("tempo.ratio")
    assert finding is not None
    assert finding.comparison is Comparison.WITHIN
    assert finding.margin == 0.0


def test_thirty_frames_a_second_cannot_resolve_the_tempo_band() -> None:
    """The measured result on this project's own 30 fps footage, as a test.

    One frame of ambiguity at the top moves this ratio by about 0.74, and the
    whole published band is 1.37 wide. The comparison is refused rather than
    reported on whichever side it happened to land.
    """
    report = coach(tempo_metrics(0.800, 7.0 / 30.0), phases(interval_s=1.0 / 30.0))
    refusal = report.refusal("tempo.ratio")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.UNRESOLVED
    assert report.finding("tempo.ratio") is None


def test_the_same_swing_resolves_once_the_camera_is_fast_enough() -> None:
    """The other half of the previous test, and the reason it is a gate.

    Identical durations, a finer clock, and the comparison comes back. A gate
    that refused this as well would be refusing the swing rather than the
    resolution.
    """
    report = coach(tempo_metrics(0.800, 7.0 / 30.0), phases(interval_s=FAST))
    finding = report.finding("tempo.ratio")
    assert finding is not None
    assert finding.comparison is Comparison.WITHIN


def test_a_supplied_slow_motion_factor_refuses_durations_and_spares_the_ratio() -> None:
    """The asymmetry this gate exists for.

    Nothing in a conformed slow-motion file records its playback factor, so every
    duration measured from one is a measured number multiplied by a supplied one.
    A ratio of two of those durations divides the supplied number back out.
    """
    report = coach(tempo_metrics(0.60, 0.30, slow_motion_factor=7.0), phases(interval_s=FAST))
    for rule_id in ("tempo.backswing", "tempo.downswing"):
        refusal = report.refusal(rule_id)
        assert refusal is not None
        assert refusal.refusal is FindingRefusal.SUPPLIED_TIMEBASE
    assert report.finding("tempo.ratio") is not None


def test_a_convention_is_refused_on_the_basis_gate_not_on_the_measurement() -> None:
    """A shoulder turn measured perfectly is still not comparable to folklore.

    The metric here is present, confident and squarely inside the 80-100 band the
    convention is usually quoted with. It is refused anyway, before the value is
    looked at, and the reason names the source rather than the swing.
    """
    metrics = metric_set(
        [
            metric(
                MetricName.SHOULDER_TURN,
                90.0,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.TOP,
                frames=(40,),
                uncertainty=2.0,
            )
        ]
    )
    refusal = coach(metrics, phases()).refusal("rotation.shoulder_turn_top")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.BASIS_NOT_PERMITTED
    assert "no measurement protocol" in refusal.reason


def _coach_with(rule, metrics, monkeypatch):  # type: ignore[no-untyped-def]
    """Run the engine over one substituted rule, leaving the registry alone.

    The shipped registry is the product's opinion about what may be said; a test
    that edited it would be testing a different product. Substituting one rule
    for the length of one test exercises the machinery without changing what the
    engine ships believing.
    """
    from analyzer.coaching import engine

    monkeypatch.setattr(engine, "known_rules", lambda: (rule,))
    return coach(metrics, phases())


def test_the_basis_gate_lets_a_matching_measurement_through(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Proof the gate is a gate and not a wall.

    Nothing in the shipped registry pairs a three-dimensional source with a
    triangulated measurement, because this project has no figure it can cite for
    one. So the pairing is constructed here: same engine, same gate, a source
    whose method permits `SPATIAL` and a metric that is `SPATIAL`. The finding
    comes out.
    """
    sourced = dataclasses.replace(
        REGISTRY["rotation.x_factor_top_3d"],
        source=ThresholdSource(
            citation="A fixture standing in for a citable study",
            population="Fictional",
            method=ThresholdMethod.THREE_D_MOTION_CAPTURE,
            measures="Torso-pelvis separation, tracked in three dimensions.",
            permitted_bases=[MetricBasis.SPATIAL],
        ),
        band_low=40.0,
        band_high=50.0,
        band_derivation="Invented by a test.",
        no_threshold="",
    )
    metrics = metric_set(
        [
            metric(
                MetricName.X_FACTOR_3D,
                62.0,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.SPATIAL,
                event=SwingEvent.TOP,
                frames=(40,),
                uncertainty=1.5,
            )
        ]
    )
    report = _coach_with(sourced, metrics, monkeypatch)
    finding = report.finding("rotation.x_factor_top_3d")
    assert finding is not None
    assert finding.comparison is Comparison.ABOVE
    assert finding.margin == pytest.approx(12.0)


def test_a_rule_needing_another_camera_is_refused_by_name() -> None:
    report = coach(tempo_metrics(0.60, 0.30, view=CameraView.FACE_ON), phases(interval_s=FAST))
    refusal = report.refusal("posture.spine_tilt_impact")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.VIEW_MISMATCH
    assert "down-the-line" in refusal.reason


def test_a_metric_the_clip_did_not_produce_carries_the_metric_layer_s_own_reason(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    """Restating it here would be a second account of the same fact.

    And the two would drift. The metrics layer knows why it refused; this layer
    quotes it. The rule is given a source that permits the metric's own basis,
    because otherwise the basis gate refuses first -- correctly, and before the
    behaviour under test is reached.
    """
    sourced = dataclasses.replace(
        REGISTRY["rotation.shoulder_turn_top"],
        source=ThresholdSource(
            citation="A fixture",
            population="Fictional",
            method=ThresholdMethod.TWO_D_VIDEO,
            measures="A shoulder turn, filmed face-on.",
            permitted_bases=[MetricBasis.FORESHORTENED_ANGLE],
            filmed_from=CameraView.FACE_ON,
        ),
    )
    metrics = metric_set(
        [],
        refused=[
            RefusedMetric(
                name=MetricName.SHOULDER_TURN,
                event=SwingEvent.TOP,
                reason="The shoulders were never tracked in this clip.",
            )
        ],
    )
    refusal = _coach_with(sourced, metrics, monkeypatch).refusal("rotation.shoulder_turn_top")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.NO_METRIC
    assert refusal.reason == "The shoulders were never tracked in this clip."


def test_a_measurement_the_engine_does_not_stand_behind_supports_no_conclusion() -> None:
    report = coach(
        tempo_metrics(0.60, 0.30, confidence=0.10),
        phases(interval_s=FAST),
        CoachingConfig(min_confidence=0.25),
    )
    refusal = report.refusal("tempo.ratio")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.LOW_CONFIDENCE


def test_a_quantity_with_no_measured_uncertainty_supports_no_comparison() -> None:
    """Most projected angles carry none, so most same-clip rules refuse.

    Reported as its own code rather than folded into `UNRESOLVED`: "the value is
    too close to the edge to call" and "nothing measured how close it could be"
    are different facts, and only the first is fixed by a better camera.
    """
    metrics = metric_set(
        [
            metric(
                MetricName.SPINE_TILT,
                35.0,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.PROJECTED_ANGLE,
                phase=SwingPhase.ADDRESS,
                frames=(0, 1, 2),
            ),
            metric(
                MetricName.SPINE_TILT,
                28.0,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.PROJECTED_ANGLE,
                event=SwingEvent.IMPACT,
                frames=(50,),
            ),
        ],
        view=CameraView.DOWN_THE_LINE,
    )
    refusal = coach(metrics, phases()).refusal("posture.spine_tilt_impact")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.NO_UNCERTAINTY


def test_a_same_clip_change_fires_when_it_clears_both_uncertainties() -> None:
    metrics = _pelvis_turn(top=20.0, impact=60.0, uncertainty=3.0)
    finding = coach(metrics, phases()).finding("rotation.pelvis_turn_through_impact")
    assert finding is not None
    assert finding.comparison is Comparison.ABOVE
    assert finding.value == pytest.approx(40.0)
    assert finding.bracket == pytest.approx(3.0 * 2**0.5)
    assert "survives the projection" in finding.observation


def test_a_same_clip_change_inside_the_uncertainty_is_refused() -> None:
    """The real outcome on this project's tour footage, as a test.

    A pelvis turn of 27.9 degrees give or take 14.9 at the top, and 33.2 give or
    take 8.0 at impact. The change is 5.3 and the two measurements leave 16.9
    open between them, so whether it changed at all is not something the clip
    shows.
    """
    metrics = metric_set(
        [
            metric(
                MetricName.PELVIS_TURN,
                27.9,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.TOP,
                frames=(40,),
                uncertainty=14.9,
            ),
            metric(
                MetricName.PELVIS_TURN,
                33.2,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.IMPACT,
                frames=(50,),
                uncertainty=8.0,
            ),
        ]
    )
    refusal = coach(metrics, phases()).refusal("rotation.pelvis_turn_through_impact")
    assert refusal is not None
    assert refusal.refusal is FindingRefusal.UNRESOLVED


def _pelvis_turn(top: float, impact: float, uncertainty: float):  # type: ignore[no-untyped-def]
    return metric_set(
        [
            metric(
                MetricName.PELVIS_TURN,
                top,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.TOP,
                frames=(40,),
                uncertainty=uncertainty,
            ),
            metric(
                MetricName.PELVIS_TURN,
                impact,
                unit=MetricUnit.DEGREES,
                basis=MetricBasis.FORESHORTENED_ANGLE,
                event=SwingEvent.IMPACT,
                frames=(50,),
                uncertainty=uncertainty,
            ),
        ]
    )


# --- properties that hold across every finding the engine can produce -------


def _every_finding():  # type: ignore[no-untyped-def]
    """One finding of each shape the shipped rules can produce."""
    fast = phases(interval_s=FAST)
    return [
        *coach(tempo_metrics(0.60, 0.30), fast).findings,
        *coach(tempo_metrics(0.70, 0.23), fast).findings,
        *coach(_pelvis_turn(top=20.0, impact=60.0, uncertainty=3.0), phases()).findings,
    ]


def test_every_finding_s_own_sentence_survives_its_own_guard() -> None:
    """The cross-check that makes the guard a contract rather than a filter.

    A template able to emit a number absent from its own evidence is exactly the
    bug the guard exists to catch in a language model's output, and finding it
    here costs nothing.
    """
    produced = _every_finding()
    assert produced
    for finding in produced:
        offences = guard.inspect(finding.observation, guard.allowance_for(finding))
        assert not offences, f"{finding.rule_id}: {offences} in {finding.observation!r}"


def test_every_finding_cites_evidence_whose_frames_are_in_the_clip() -> None:
    """The exit criterion for the phase, asserted rather than asserted about."""
    for finding in _every_finding():
        assert finding.evidence
        for item in finding.evidence:
            assert item.frames
            assert all(0 <= frame < 400 for frame in item.frames)
            assert item.methodology


def test_timestamps_are_parallel_to_frames_or_absent_entirely() -> None:
    """A partial list is the outcome that silently mislabels a frame."""
    times = [index / 240.0 for index in range(400)]
    report = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST), times=times)
    for finding in report.findings:
        for item in finding.evidence:
            assert len(item.timestamps_s) == len(item.frames)
            assert item.timestamps_s == [times[frame] for frame in item.frames]

    without = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST))
    for finding in without.findings:
        for item in finding.evidence:
            assert item.timestamps_s == []


def test_a_tempo_finding_cites_the_durations_its_bracket_came_from() -> None:
    """Otherwise the report shows a number whose uncertainty came from off-stage."""
    finding = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST)).finding("tempo.ratio")
    assert finding is not None
    cited = {item.metric for item in finding.evidence}
    assert cited == {
        MetricName.TEMPO_RATIO,
        MetricName.BACKSWING_DURATION,
        MetricName.DOWNSWING_DURATION,
    }


def test_no_finding_carries_a_score() -> None:
    """There is no total, and nothing that could be read as one.

    Held as a test because a severity field is the single most requested addition
    to a system like this, and adding one is a one-line change that cannot be
    undone once a UI sorts by it.
    """
    for finding in _every_finding():
        fields = set(type(finding).model_fields)
        assert not fields & {"severity", "score", "grade", "rating", "priority"}


def test_every_rule_is_accounted_for_exactly_once() -> None:
    """A rule that neither fired nor refused would vanish from the report."""
    report = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST))
    seen = [entry.rule_id for entry in report.findings] + [
        entry.rule_id for entry in report.refused
    ]
    assert sorted(seen) == sorted(REGISTRY)


def test_the_report_says_how_many_refusals_no_recording_can_change() -> None:
    report = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST))
    unchangeable = [
        entry
        for entry in report.refused
        if entry.refusal in {FindingRefusal.NO_THRESHOLD, FindingRefusal.BASIS_NOT_PERMITTED}
    ]
    assert len(unchangeable) == 7
    assert any("A different clip changes neither count" in text for text in report.warnings)


@pytest.mark.parametrize("rule_id", sorted(REGISTRY), ids=lambda value: value)
def test_no_rule_can_fire_without_passing_the_gates(rule_id: str) -> None:
    """Every rule in the registry reaches a decision on an ordinary clip.

    Parametrised so a rule added later without thinking about the gates shows up
    as a named failure rather than as one absent row in a report nobody reads.
    """
    report = coach(tempo_metrics(0.60, 0.30), phases(interval_s=FAST))
    rule = REGISTRY[rule_id]
    outcome = report.finding(rule_id) or report.refusal(rule_id)
    assert outcome is not None
    if report.finding(rule_id) is not None:
        assert rule.kind is RuleKind.BAND
        assert rule.source.permitted_bases
