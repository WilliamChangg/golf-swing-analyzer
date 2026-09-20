"""What every rule must be true of, whoever adds the next one.

These are the tests that stop the registry rotting. A rule is cheap to add and a
threshold is cheap to type; what is expensive is a threshold that arrives without
its provenance, or one whose source measured a quantity nobody checked against
the quantity being compared. Each test here is a thing that would be noticed in
review on the day and not six months later.

The most important of them is `test_a_source_may_only_permit_bases_its_method_can_measure`.
It is the phase's whole argument, expressed as an invariant: a number from a
three-dimensional measurement may only ever be compared against a
three-dimensional one.
"""

from __future__ import annotations

import pytest

from analyzer.biomechanics.registry import REGISTRY as METRIC_REGISTRY
from analyzer.coaching.registry import REGISTRY, RuleKind, known_rules
from analyzer.contracts.coaching import ThresholdMethod
from analyzer.contracts.metrics import MetricBasis

RULES = known_rules()

# What each way of measuring the world entitles a threshold to be compared
# against. `SAME_CLIP` is absent because it borrows nothing and is checked
# separately; `CONVENTION` is present and maps to nothing at all.
_METHOD_ALLOWS: dict[ThresholdMethod, frozenset[MetricBasis]] = {
    ThresholdMethod.THREE_D_MOTION_CAPTURE: frozenset({MetricBasis.SPATIAL}),
    ThresholdMethod.INSTRUMENTED_SENSOR: frozenset({MetricBasis.SPATIAL}),
    ThresholdMethod.TWO_D_VIDEO: frozenset(
        {
            MetricBasis.TEMPORAL,
            MetricBasis.IMAGE_PLANE,
            MetricBasis.PROJECTED_ANGLE,
            MetricBasis.FORESHORTENED_ANGLE,
        }
    ),
    ThresholdMethod.CONVENTION: frozenset(),
}


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_a_source_may_only_permit_bases_its_method_can_measure(rule) -> None:  # type: ignore[no-untyped-def]
    """A threshold cannot be about a kind of quantity its instrument never saw.

    The invariant the whole phase rests on. Marker-based capture measures the
    body in three dimensions, so its numbers are thresholds on `SPATIAL`
    measurements and on nothing else -- not on an angle inferred from how much a
    line shortened in one photograph, which is a different quantity that shares
    a name with it.
    """
    source = rule.source
    if source.method is ThresholdMethod.SAME_CLIP:
        return
    allowed = _METHOD_ALLOWS[source.method]
    assert set(source.permitted_bases) <= allowed, (
        f"{rule.rule_id}: {source.method.value} cannot supply a threshold on "
        f"{set(source.permitted_bases) - allowed}"
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_a_video_source_declares_where_it_filmed_from_before_supplying_an_angle(rule) -> None:  # type: ignore[no-untyped-def]
    """A projected quantity is a fact about a camera position as much as a body.

    A duration measured off video transfers to any footage; an angle measured off
    video transfers only to footage shot from the same place. A source that
    permits a non-temporal basis without saying where it filmed from is claiming
    the first while supplying the second.
    """
    source = rule.source
    if source.method is not ThresholdMethod.TWO_D_VIDEO:
        return
    projected = set(source.permitted_bases) - {MetricBasis.TEMPORAL}
    if projected:
        assert source.filmed_from is not None, (
            f"{rule.rule_id}: {source.citation} permits {projected} without saying "
            "which camera position it measured from"
        )


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_a_convention_permits_nothing(rule) -> None:  # type: ignore[no-untyped-def]
    """A number with no protocol is not a threshold on anything.

    Kept as a test rather than as a comment because the tempting fix, when a rule
    refuses on real footage, is to add a basis to its source and watch the rule
    start firing.
    """
    if rule.source.method is ThresholdMethod.CONVENTION:
        assert rule.source.permitted_bases == []


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_every_band_states_how_it_was_derived(rule) -> None:  # type: ignore[no-untyped-def]
    """A range that appeared without a derivation is a range somebody chose.

    And a reader cannot tell the difference afterwards, which is why the
    derivation is required rather than encouraged.
    """
    has_band = rule.band_low is not None or rule.band_high is not None
    if has_band:
        assert rule.band_low is not None and rule.band_high is not None
        assert rule.band_low < rule.band_high
        assert rule.band_derivation, f"{rule.rule_id} has a band and no derivation"
        assert not rule.no_threshold


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_a_rule_without_a_band_says_why_there_is_none(rule) -> None:  # type: ignore[no-untyped-def]
    """Absence of a threshold is a finding about the literature, not a gap."""
    if rule.kind is RuleKind.SAME_CLIP:
        return
    if rule.band_low is None:
        assert rule.no_threshold, f"{rule.rule_id} has no band and no explanation"


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_every_rule_names_a_metric_the_engine_can_produce(rule) -> None:  # type: ignore[no-untyped-def]
    """A rule about a quantity nothing measures would refuse for the wrong reason.

    It would come out as `NO_METRIC` on every clip, which reads as a problem with
    the recording, and be indistinguishable from a rule that a better camera
    would satisfy.
    """
    assert rule.metric in METRIC_REGISTRY


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.rule_id)
def test_a_same_clip_rule_has_a_second_anchor(rule) -> None:  # type: ignore[no-untyped-def]
    """Comparing a measurement against itself needs two instants to compare."""
    if rule.kind is not RuleKind.SAME_CLIP:
        assert rule.compare_event is None and rule.compare_phase is None
        return
    assert rule.compare_event is not None or rule.compare_phase is not None
    assert (rule.event, rule.phase) != (rule.compare_event, rule.compare_phase)


def test_rule_ids_are_unique() -> None:
    """The registry is a dict, so a duplicate would silently drop a rule."""
    assert len(REGISTRY) == len(RULES)


def test_the_registry_is_mostly_refusals_and_that_is_recorded() -> None:
    """The count in the module docstring is a claim, so it is checked.

    Not pedantry: the docstring's argument -- that the numbers golf instruction
    publishes mostly cannot be used -- is only worth making while the numbers
    behind it are right. If a later phase adds a properly sourced threshold, this
    test fails and the docstring gets rewritten with it.
    """
    usable = [rule for rule in RULES if rule.source.permitted_bases and not rule.no_threshold]
    assert len(RULES) == 12
    assert len(usable) == 5  # three tempo rules, two same-clip comparisons
    assert sum(1 for rule in RULES if rule.source.method is ThresholdMethod.CONVENTION) == 6
