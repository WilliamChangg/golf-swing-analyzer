"""Evaluating every rule against one swing.

The entry point to the coaching engine, and the place where a measurement becomes
something said to a person. Each rule passes through the same seven gates in the
same order, and a rule that stops at any of them produces a `RefusedFinding`
naming which one -- because the seven are seven different things to do about it,
and a single "could not evaluate" would be useless to everybody.

## The order of the gates, and why it is that order

They run from facts about the **world**, which no recording can change, to facts
about **this recording**, which a person with a camera can:

1. `NO_SWING` -- nothing was detected, so there is nothing to reason about.
2. `NO_THRESHOLD` -- nobody has published a number for this quantity. No camera
   fixes that, and it is the most useful thing to say when it is true.
3. `BASIS_NOT_PERMITTED` -- a number exists, and it is a number about a
   different quantity from the one this engine measures. Checked **before**
   asking whether the clip produced the measurement, because a reader told "your
   clip could not measure your shoulder turn" will go and re-film, and the
   re-filmed clip will be refused for this reason instead.
4. `VIEW_MISMATCH` -- the comparison needs a camera position this clip was not
   shot from.
5. `NO_METRIC` -- this clip did not produce the measurement. The metric layer's
   own refusal is carried through verbatim, because it already said why.
6. `LOW_CONFIDENCE` -- it produced one, and does not stand behind it.
7. `NO_UNCERTAINTY` / `UNRESOLVED` -- how finely the value is known, against how
   far it sits from the edge it is being compared to. The last gate rather than
   the first because it is the only one that needs the number.

## What comes out

`Finding.observation` is assembled here, from the rule's declared words and the
measured value, and it is held to the same guard the optional phrasing layer is.
The test suite runs every rule's own sentence through `guard.inspect`; a template
that could emit a number absent from its evidence fails the suite rather than
shipping.

Nothing here scores a swing. There is no total, no grade and no ranking between
findings, and that absence is load-bearing: a single number summarising a swing
would be the most quoted output of this system and the one with the least behind
it -- it would need a scale relating degrees of turn to seconds of tempo, and
nobody has measured one.
"""

from __future__ import annotations

from collections.abc import Sequence

from analyzer.biomechanics.registry import definition
from analyzer.coaching.bracket import bracket_for, combined_bracket, frame_interval_s
from analyzer.coaching.evidence import evidence_from
from analyzer.coaching.registry import Rule, RuleKind, known_rules
from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.coaching import (
    CoachingConfig,
    CoachingReport,
    Comparison,
    Evidence,
    Finding,
    FindingRefusal,
    RefusedFinding,
)
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricBasis,
    MetricName,
    MetricSet,
    MetricUnit,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase, SwingPhases

# How finely each unit is written into an observation. Chosen so that the printed
# number is always within half a unit in the last place of the measured one,
# which is what the guard's tolerance is derived from -- print a value more
# coarsely than this and the sentence stops matching its own evidence.
_PRECISION: dict[MetricUnit, int] = {
    MetricUnit.DEGREES: 1,
    MetricUnit.SECONDS: 3,
    MetricUnit.RATIO: 2,
    MetricUnit.TORSO_LENGTHS: 3,
    MetricUnit.TORSO_LENGTHS_PER_S: 2,
    MetricUnit.METRES_PER_S: 2,
}

# The unit as it is written in a sentence. "/s" rather than "per second"
# throughout: "second" is a unit word, and a finding with no duration in its
# evidence must not use one.
_SUFFIX: dict[MetricUnit, str] = {
    MetricUnit.DEGREES: " degrees",
    MetricUnit.SECONDS: " s",
    MetricUnit.RATIO: "",
    MetricUnit.TORSO_LENGTHS: " torso lengths",
    MetricUnit.TORSO_LENGTHS_PER_S: " torso lengths/s",
    MetricUnit.METRES_PER_S: " m/s",
}


def _number(value: float, unit: MetricUnit) -> str:
    return f"{value:.{_PRECISION.get(unit, 3)}f}"


def _quantity(value: float, unit: MetricUnit) -> str:
    """A value with its unit, written so the guard can match it.

    A ratio is written `1.83:1` rather than `1.83 to 1`, because the guard reads
    a colon ratio as its quotient. Spelled out, the trailing "1" would be a bare
    number that has to appear in the evidence, and it never does.
    """
    if unit is MetricUnit.RATIO:
        return f"{_number(value, unit)}:1"
    return f"{_number(value, unit)}{_SUFFIX.get(unit, '')}"


def _difference(value: float, unit: MetricUnit) -> str:
    """A gap between two values of a unit, which is not always that unit.

    The distance between two ratios is not a ratio: 3.43:1 sits 0.37 from 3.80:1
    and writing that as "0.37:1" would name a tempo no part of this is talking
    about. Every other unit's difference is in the unit, and is written that way.
    """
    if unit is MetricUnit.RATIO:
        return _number(value, unit)
    return _quantity(value, unit)


def _lookup(
    metrics: MetricSet, name: MetricName, event: SwingEvent | None, phase: SwingPhase | None
) -> Metric | None:
    """The one metric a rule is about, matched on all three of its coordinates."""
    return next(
        (
            entry
            for entry in metrics.metrics
            if entry.name is name and entry.event is event and entry.phase is phase
        ),
        None,
    )


def _metric_refusal(metrics: MetricSet, name: MetricName, event: SwingEvent | None) -> str:
    """Why the metric layer did not produce this, in its own words.

    Carried through rather than restated. The layer that refused knows why it
    did, and a coaching-layer paraphrase would be a second account of the same
    fact that can drift from the first.
    """
    exact = next(
        (entry for entry in metrics.refused if entry.name is name and entry.event is event),
        None,
    )
    if exact is not None:
        return exact.reason
    any_anchor = next((entry for entry in metrics.refused if entry.name is name), None)
    if any_anchor is not None:
        return any_anchor.reason
    return (
        f"{definition(name).label} was not produced for this clip, and the metrics "
        "layer recorded no reason -- which is itself worth reporting."
    )


def _band_words(rule: Rule, unit: MetricUnit) -> str:
    low, high = rule.band_low, rule.band_high
    assert low is not None and high is not None  # noqa: S101 - callers gate on this
    return f"{_number(low, unit)}-{_quantity(high, unit)}"


def _observe_band(
    rule: Rule, metric: Metric, comparison: Comparison, margin: float, bracket: float
) -> str:
    """The sentence for a band comparison, built from the numbers behind it.

    Every quantity in it is in the finding's own evidence, which is what lets the
    same guard run over this and over a language model's rewrite of it.
    """
    unit = metric.unit
    measured = f"{rule.subject.capitalize()} measured {_quantity(metric.value, unit)}"
    band = _band_words(rule, unit)
    clears = (
        f"The bracket on this clip's own measurement is {_difference(bracket, unit)}, "
        "so the gap is wider than what its frame rate leaves open."
    )
    if comparison is Comparison.ABOVE:
        return (
            f"{measured}, which is {_difference(margin, unit)} above the top of the "
            f"published {band} range. {clears}"
        )
    if comparison is Comparison.BELOW:
        return (
            f"{measured}, which is {_difference(margin, unit)} below the bottom of the "
            f"published {band} range. {clears}"
        )
    return (
        f"{measured}, inside the published {band} range and clear of both ends by "
        f"more than the {_difference(bracket, unit)} bracket on this clip's own measurement."
    )


def _observe_change(rule: Rule, first: Metric, second: Metric, bracket: float) -> str:
    """The sentence for a same-clip comparison.

    States both endpoints and the change, and then states what a projection does
    to it. That last sentence is not decoration: the direction of a change
    survives a foreshortening and the size of it does not, and a reader given the
    number without the caveat will read the size.
    """
    unit = first.unit
    change = second.value - first.value
    direction = "increased" if change > 0 else "decreased"
    return (
        f"{rule.subject.capitalize()} measured {_quantity(first.value, unit)} at "
        f"{_anchor_words(first)} and {_quantity(second.value, unit)} at "
        f"{_anchor_words(second)}: it {direction} by {_difference(abs(change), unit)}, "
        f"which is more than the {_difference(bracket, unit)} those measurements "
        "leave open between them. The direction of a change of this kind survives "
        "the projection; its size is a statement about the picture."
    )
    # "those measurements" rather than "the two measurements" on purpose: the
    # guard reads a spelled-out "two" as the number 2, which is not in this
    # finding's evidence, and rejects the sentence. The first thing the guard
    # ever caught was this template. Leaving the wording as it was and teaching
    # the guard to ignore small counting words would have widened the hole that
    # the whole check exists to close.


def _anchor_words(metric: Metric) -> str:
    if metric.event is not None:
        return {
            SwingEvent.TAKEAWAY: "the takeaway",
            SwingEvent.TOP: "the top",
            SwingEvent.IMPACT: "impact",
            SwingEvent.FINISH: "the finish",
        }[metric.event]
    if metric.phase is not None:
        return {
            SwingPhase.ADDRESS: "address",
            SwingPhase.BACKSWING: "the backswing",
            SwingPhase.DOWNSWING: "the downswing",
            SwingPhase.FOLLOW_THROUGH: "the follow-through",
        }[metric.phase]
    return "the swing"


def _refuse(rule: Rule, refusal: FindingRefusal, reason: str) -> RefusedFinding:
    return RefusedFinding(
        rule_id=rule.rule_id,
        title=rule.title,
        refusal=refusal,
        reason=reason,
        source=rule.source,
    )


def _basis_reason(rule: Rule, basis: MetricBasis) -> str:
    """Why a number that exists is not a number about this measurement."""
    source = rule.source
    measured = definition(rule.metric).label.lower()
    named = basis.value.replace("_", " ")
    permitted = ", ".join(entry.value.replace("_", " ") for entry in source.permitted_bases)
    if not permitted:
        return (
            f"The source ({source.citation}) publishes no measurement protocol, so its "
            f"number is not a threshold on any measured quantity. This engine's "
            f"{measured} is a {named} measurement, and there is nothing here for it to "
            "be compared against that would mean anything."
        )
    return (
        f"The source ({source.citation}) measured a {permitted} quantity, and this "
        f"engine's {measured} is a {named} one. They share a name and are not the same "
        "thing: they disagree by more than the comparison would be measuring, so the "
        f"threshold does not transfer. What the source measured: {source.measures}"
    )


def _evaluate_band(
    rule: Rule,
    metrics: MetricSet,
    interval: float | None,
    times: Sequence[float] | None,
    config: CoachingConfig,
) -> Finding | RefusedFinding:
    metric = _lookup(metrics, rule.metric, rule.event, rule.phase)
    if metric is None:
        return _refuse(
            rule, FindingRefusal.NO_METRIC, _metric_refusal(metrics, rule.metric, rule.event)
        )
    if metric.confidence.overall < config.min_confidence:
        return _refuse(
            rule,
            FindingRefusal.LOW_CONFIDENCE,
            f"{metric.label} was measured at a confidence of "
            f"{metric.confidence.overall:.2f}, below the {config.min_confidence:.2f} "
            "this engine requires before a measurement supports a conclusion. "
            f"{metric.methodology}",
        )

    bracket, why = bracket_for(metric, metrics, interval)
    if bracket is None:
        return _refuse(rule, FindingRefusal.NO_UNCERTAINTY, why)

    assert rule.band_low is not None and rule.band_high is not None  # noqa: S101 - gated above
    low, high = rule.band_low, rule.band_high
    value = metric.value
    if value > high:
        comparison, margin = Comparison.ABOVE, value - high
    elif value < low:
        comparison, margin = Comparison.BELOW, low - value
    else:
        comparison, margin = Comparison.WITHIN, 0.0

    distance = (
        min(abs(value - low), abs(value - high)) if comparison is Comparison.WITHIN else margin
    )
    if distance <= bracket:
        return _refuse(
            rule,
            FindingRefusal.UNRESOLVED,
            f"{metric.label} measured {_quantity(value, metric.unit)} against a "
            f"{_band_words(rule, metric.unit)} range, which is "
            f"{_difference(distance, metric.unit)} from the nearer end of it -- inside "
            f"the {_difference(bracket, metric.unit)} this measurement leaves open. "
            f"{why} The clip cannot resolve which side of the range this swing is on.",
        )

    return Finding(
        rule_id=rule.rule_id,
        title=rule.title,
        observation=_observe_band(rule, metric, comparison, margin, bracket),
        comparison=comparison,
        value=value,
        unit=metric.unit,
        bracket=bracket,
        band_low=low,
        band_high=high,
        margin=margin,
        source=rule.source,
        evidence=_band_evidence(rule, metric, metrics, times),
        confidence=metric.confidence.overall,
    )


def _band_evidence(
    rule: Rule, metric: Metric, metrics: MetricSet, times: Sequence[float] | None
) -> list[Evidence]:
    """The measurement, plus anything its bracket was derived from.

    The tempo ratio is the case this exists for. Its bracket comes from the two
    durations it divides rather than from itself, so a reader checking the
    comparison needs those frames as well -- otherwise the finding cites a number
    whose uncertainty came from somewhere the report does not show.
    """
    items = [evidence_from(metric, times)]
    if metric.unit is not MetricUnit.RATIO:
        return items
    for name in (MetricName.BACKSWING_DURATION, MetricName.DOWNSWING_DURATION):
        supporting = metrics.get(name)
        if supporting is not None:
            items.append(evidence_from(supporting, times))
    return items


def _evaluate_change(
    rule: Rule,
    metrics: MetricSet,
    interval: float | None,
    times: Sequence[float] | None,
    config: CoachingConfig,
) -> Finding | RefusedFinding:
    first = _lookup(metrics, rule.metric, rule.event, rule.phase)
    second = _lookup(metrics, rule.metric, rule.compare_event, rule.compare_phase)
    if first is None or second is None:
        missing = rule.event if first is None else rule.compare_event
        return _refuse(
            rule, FindingRefusal.NO_METRIC, _metric_refusal(metrics, rule.metric, missing)
        )

    weakest = min(first.confidence.overall, second.confidence.overall)
    if weakest < config.min_confidence:
        return _refuse(
            rule,
            FindingRefusal.LOW_CONFIDENCE,
            f"{definition(rule.metric).label} was measured at a confidence of "
            f"{weakest:.2f} at one of the two anchors, below the "
            f"{config.min_confidence:.2f} this engine requires.",
        )

    first_bracket, first_why = bracket_for(first, metrics, interval)
    second_bracket, second_why = bracket_for(second, metrics, interval)
    if first_bracket is None or second_bracket is None:
        return _refuse(
            rule,
            FindingRefusal.NO_UNCERTAINTY,
            first_why if first_bracket is None else second_why,
        )

    bracket = combined_bracket(first_bracket, second_bracket)
    change = second.value - first.value
    if abs(change) <= bracket:
        return _refuse(
            rule,
            FindingRefusal.UNRESOLVED,
            f"{definition(rule.metric).label} moved {_difference(abs(change), first.unit)} "
            f"between {_anchor_words(first)} and {_anchor_words(second)}, which is "
            f"inside the {_difference(bracket, first.unit)} the two measurements leave "
            "open between them. Whether it changed at all is not something this clip shows.",
        )

    return Finding(
        rule_id=rule.rule_id,
        title=rule.title,
        observation=_observe_change(rule, first, second, bracket),
        comparison=Comparison.ABOVE if change > 0 else Comparison.BELOW,
        value=change,
        unit=first.unit,
        bracket=bracket,
        band_low=None,
        band_high=None,
        margin=abs(change) - bracket,
        source=rule.source,
        evidence=[evidence_from(first, times), evidence_from(second, times)],
        confidence=weakest,
    )


def _gate(rule: Rule, metrics: MetricSet, view: CameraView) -> RefusedFinding | None:
    """The gates that need no measured value, in the order of the docstring."""
    if rule.no_threshold:
        return _refuse(rule, FindingRefusal.NO_THRESHOLD, rule.no_threshold)

    basis = definition(rule.metric).basis
    if basis not in rule.source.permitted_bases:
        return _refuse(rule, FindingRefusal.BASIS_NOT_PERMITTED, _basis_reason(rule, basis))

    named = view.value.replace("_", "-")
    if rule.requires_view is not None and rule.requires_view is not view:
        wanted = rule.requires_view.value.replace("_", "-")
        return _refuse(
            rule,
            FindingRefusal.VIEW_MISMATCH,
            f"This comparison is about what a {wanted} camera sees, and this clip "
            f"was measured as {named}. The same numbers mean a different thing from "
            "the other position, so the rule is refused rather than applied to them.",
        )
    if (
        rule.source.filmed_from is not None
        and basis is not MetricBasis.TEMPORAL
        and rule.source.filmed_from is not view
    ):
        filmed = rule.source.filmed_from.value.replace("_", "-")
        return _refuse(
            rule,
            FindingRefusal.VIEW_MISMATCH,
            f"{rule.source.citation} measured from a {filmed} camera and this clip "
            f"is {named}. A projected quantity is a fact about a camera position as "
            "much as about a body, so the two numbers are not comparable.",
        )

    factor = metrics.slow_motion_factor
    if (
        rule.kind is RuleKind.BAND
        and definition(rule.metric).unit is MetricUnit.SECONDS
        and factor != 1.0
    ):
        return _refuse(
            rule,
            FindingRefusal.SUPPLIED_TIMEBASE,
            f"This clip was measured as {factor:g} times slower than real time, and "
            "that factor was supplied rather than measured -- nothing in a conformed "
            "slow-motion file records it. Every duration here is therefore a measured "
            "number multiplied by a guess, and comparing one against a band in seconds "
            "would be comparing a stopwatch against the guess. The tempo ratio is not "
            "refused for this: a factor that stretches both durations by the same "
            "amount divides out of their ratio.",
        )
    return None


def coach(
    metrics: MetricSet,
    phases: SwingPhases,
    config: CoachingConfig | None = None,
    times: Sequence[float] | None = None,
) -> CoachingReport:
    """Evaluate every rule against one measured swing.

    `times` is the clip's per-frame timestamp vector in real seconds, used only
    to give each piece of evidence an instant alongside its frame. Absent, the
    frames are still cited and the instants are left empty rather than derived
    from a frame rate this function would have to assume.
    """
    resolved = config or CoachingConfig()
    rules = known_rules()

    if not metrics.computed or not phases.detected:
        reason = (
            "No swing was detected in this clip, so there is nothing to reach a "
            "conclusion about. Every rule here compares a measurement anchored to "
            "a swing event, and a clip without events has none to anchor to."
        )
        return CoachingReport(
            computed=False,
            refused=[_refuse(rule, FindingRefusal.NO_SWING, reason) for rule in rules],
            rules_considered=len(rules),
            config=resolved,
            warnings=[reason],
        )

    view = metrics.view.view if metrics.view is not None else CameraView.UNKNOWN
    interval = frame_interval_s(phases)

    findings: list[Finding] = []
    refused: list[RefusedFinding] = []
    for rule in rules:
        blocked = _gate(rule, metrics, view)
        if blocked is not None:
            refused.append(blocked)
            continue
        evaluate = _evaluate_change if rule.kind is RuleKind.SAME_CLIP else _evaluate_band
        outcome = evaluate(rule, metrics, interval, times, resolved)
        if isinstance(outcome, Finding):
            findings.append(outcome)
        else:
            refused.append(outcome)

    report = CoachingReport(
        computed=True,
        findings=findings,
        refused=refused,
        rules_considered=len(rules),
        view=view,
        frame_interval_s=interval,
        config=resolved,
    )
    report.warnings = _warnings(report, metrics)
    return report


def _warnings(report: CoachingReport, metrics: MetricSet) -> list[str]:
    """What a reader should not have to derive from counting the refusals."""
    warnings: list[str] = []

    borrowed = [
        entry for entry in report.refused if entry.refusal is FindingRefusal.BASIS_NOT_PERMITTED
    ]
    absent = [entry for entry in report.refused if entry.refusal is FindingRefusal.NO_THRESHOLD]
    if borrowed or absent:
        warnings.append(
            f"{len(borrowed) + len(absent)} of {report.rules_considered} rules were "
            "refused before this clip was looked at. That is a statement about the "
            "numbers golf instruction publishes rather than about this recording: "
            f"{len(absent)} of them name a quantity nobody has published a threshold "
            f"for, and {len(borrowed)} rest on a number measured in a way that does "
            "not describe what a single camera records. A different clip changes "
            "neither count."
        )

    supplied = [
        entry for entry in report.refused if entry.refusal is FindingRefusal.SUPPLIED_TIMEBASE
    ]
    if supplied:
        warnings.append(
            f"{len(supplied)} comparison(s) against a band in seconds were refused "
            f"because this clip's playback factor of {metrics.slow_motion_factor:g} was "
            "supplied rather than measured. What survives is the ratio of two of its "
            "own durations, which that factor divides out of -- the same reason a "
            "length here is in torso lengths rather than in pixels."
        )

    unresolved = [entry for entry in report.refused if entry.refusal is FindingRefusal.UNRESOLVED]
    if unresolved and report.frame_interval_s is not None:
        rate = 1.0 / report.frame_interval_s
        warnings.append(
            f"{len(unresolved)} comparison(s) came out closer to their range than "
            f"this clip can resolve. Its events fall {report.frame_interval_s * 1000:.1f} ms "
            f"apart -- about {rate:.0f} samples per real second -- and a timing "
            "comparison cannot be finer than that. A faster capture, or a "
            "slow-motion recording with its factor declared, is what moves this."
        )

    if metrics.calibration is CalibrationStatus.NONE:
        geometric = [
            entry
            for entry in report.findings
            if any(item.basis is not MetricBasis.TEMPORAL for item in entry.evidence)
        ]
        # Deliberately not a restatement of the metrics layer's calibration
        # warning, which travels with the report already. The fact worth adding
        # here is which findings a lens could have touched, and on an
        # uncalibrated clip that is usually none of them -- not because the
        # engine is careful at this layer, but because the ones a lens touches
        # were all refused further up.
        warnings.append(
            "Every finding above rests on the clock alone, which an uncalibrated "
            "lens does not touch."
            if not geometric
            else (
                f"{len(geometric)} of the findings above rest on a length or an angle "
                "measured through an uncalibrated lens, so the distortion is in them."
            )
        )
    return warnings
