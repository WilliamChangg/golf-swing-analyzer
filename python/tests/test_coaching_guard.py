"""The guard, against the sentences it exists to stop.

Most of this file is a corpus of plausible wrong answers. They are written the
way a language model writes -- fluent, confident, and containing one thing nobody
measured -- because that is the failure mode. A guard tested only against
gibberish would pass everything that matters.

The accepted cases matter as much as the rejected ones. A guard that rejects
every rewording is not protecting anybody; it has just turned the phrasing layer
off in a way that looks like it is working.
"""

from __future__ import annotations

import pytest

from analyzer.coaching.guard import allowance_for, inspect, passes
from analyzer.contracts.coaching import (
    Comparison,
    Evidence,
    Finding,
    ThresholdMethod,
    ThresholdSource,
)
from analyzer.contracts.metrics import MetricBasis, MetricName, MetricUnit
from analyzer.contracts.phases import SwingPhase

SOURCE = ThresholdSource(
    citation="A fixture",
    year=2004,
    population="Fictional golfers",
    method=ThresholdMethod.TWO_D_VIDEO,
    measures="Frames.",
    permitted_bases=[MetricBasis.TEMPORAL],
)


def _finding() -> Finding:
    """A downswing 0.100 s above a 0.200-0.267 s band, measured on frames 40-49."""
    return Finding(
        rule_id="tempo.downswing",
        title="Downswing duration against the tour band",
        observation="The downswing measured 0.367 s, which is 0.100 s above the top.",
        comparison=Comparison.ABOVE,
        value=0.3667,
        unit=MetricUnit.SECONDS,
        bracket=0.0048,
        band_low=0.200,
        band_high=0.2667,
        margin=0.1000,
        confidence=0.87,
        source=SOURCE,
        evidence=[
            Evidence(
                metric=MetricName.DOWNSWING_DURATION,
                label="Downswing duration",
                value=0.3667,
                unit=MetricUnit.SECONDS,
                basis=MetricBasis.TEMPORAL,
                phase=SwingPhase.DOWNSWING,
                confidence=0.87,
                frames=[40, 90],
                methodology="Between the top and impact.",
            )
        ],
    )


def test_a_faithful_rewording_is_accepted() -> None:
    """The case that makes the rest of the file worth having."""
    assert passes(
        "Your downswing took 0.367 s. That is 0.100 s longer than the top of the "
        "range this source reports.",
        _finding(),
    )


def test_rounding_a_number_down_is_presentation_and_is_allowed() -> None:
    """A model writing 0.37 has claimed less precision, not a different value."""
    assert passes("The downswing took 0.37 s.", _finding())
    assert passes("The downswing took 0.4 s.", _finding())


def test_inventing_digits_is_not_presentation() -> None:
    """0.3670 claims four decimals. The measurement is 0.3667."""
    offences = inspect("The downswing took 0.3670 s.", allowance_for(_finding()))
    assert [entry.token for entry in offences] == ["0.3670"]


@pytest.mark.parametrize(
    "candidate,expected",
    [
        # The archetype: a real measurement, and one number beside it that is not.
        ("The downswing took 0.367 s, about 15% longer than tour average.", "15"),
        ("Your downswing took 0.367 s, compared with the 0.250 s tour standard.", "0.250"),
        # Both figures here are real. The invention is the multiplier.
        ("At 0.367 s this is roughly twice the 0.200 s floor.", "twice"),
    ],
)
def test_a_number_not_in_the_evidence_is_rejected(candidate: str, expected: str) -> None:
    offences = inspect(candidate, allowance_for(_finding()))
    assert any(expected in entry.token for entry in offences), offences


def test_a_spelled_out_number_does_not_walk_past_a_digit_scanner() -> None:
    """The hole a naive guard leaves, and the one this phase's rules live in.

    "Ninety degrees of shoulder turn" is the single most likely invention in golf
    coaching prose, and it contains no digits.
    """
    offences = inspect(
        "This is short of the ninety degrees a full turn is usually described as.",
        allowance_for(_finding()),
    )
    # The token carries the word it qualifies, so the report says what was claimed.
    assert "ninety degrees" in [entry.token for entry in offences], offences


def test_a_hyphenated_compound_is_read_as_one_number() -> None:
    offences = inspect("Forty-five degrees of separation.", allowance_for(_finding()))
    assert any(entry.token.startswith("forty") for entry in offences)


@pytest.mark.parametrize(
    "candidate,unit",
    [
        ("Your clubhead is moving at 0.367 mph through this.", "mph"),
        ("The hands travel about 0.367 yards.", "yards"),
        ("That is 0.367 inches of movement.", "inches"),
        ("It rotates at 0.367 rpm.", "rpm"),
    ],
)
def test_a_unit_this_system_cannot_measure_is_rejected(candidate: str, unit: str) -> None:
    """Every one of these carries a number that *is* in the evidence.

    Deliberately: the number check alone would pass all four, and a system that
    reported a torso length as a distance in yards would be inventing a scale
    rather than a value.
    """
    offences = inspect(candidate, allowance_for(_finding()))
    assert unit in [entry.token for entry in offences]


def test_a_unit_this_system_can_measure_elsewhere_is_still_rejected_here() -> None:
    """Metres exist downstream of a stereo triangulation and nowhere else.

    A finding built from one camera has no business reaching for them, however
    real the unit is in another part of the system.
    """
    offences = inspect("The hands moved 0.367 metres.", allowance_for(_finding()))
    assert "metres" in [entry.token for entry in offences]
    assert any("not in this finding's evidence" in entry.offence for entry in offences)


@pytest.mark.parametrize(
    "candidate",
    [
        "Your clubhead speed is limited by this.",
        "Expect a lower ball speed as a result.",
        "This will cost you a little carry distance.",
        "Your smash factor suffers here.",
    ],
)
def test_a_quantity_nothing_here_measures_is_rejected(candidate: str) -> None:
    """The check that catches a confident sentence containing no numbers at all."""
    assert not passes(candidate, _finding())


@pytest.mark.parametrize(
    "candidate",
    [
        "A downswing this long tends to produce a slice.",
        "This is why the ball fades away from the target.",
        "You will hook it from here.",
    ],
)
def test_a_claim_about_where_the_ball_went_is_rejected(candidate: str) -> None:
    """This system sees a ball at rest and the frame it stops being there.

    It never sees flight. Every sentence above is about an outcome nothing here
    observed, and three of them would be indistinguishable from analysis.
    """
    assert not passes(candidate, _finding())


@pytest.mark.parametrize(
    "candidate",
    [
        "The long downswing causes a loss of speed.",
        "This leads to an inconsistent strike.",
        "The extra time results in a weaker position.",
    ],
)
def test_a_causal_claim_is_rejected(candidate: str) -> None:
    """A finding is a measurement at an instant.

    That one measured quantity produced another is a claim about a mechanism
    nothing in this project tested, and it is the most natural sentence in the
    world for a model to write.
    """
    assert not passes(candidate, _finding())


def test_a_frame_index_is_quotable_only_as_a_frame() -> None:
    """The loophole a naive allowance opens, closed.

    A metric measured over the address phase cites scores of consecutive frame
    indices. Folding those into the number allowance would licence every small
    integer in the language.
    """
    finding = _finding()
    assert passes("Measured between frame 40 and frame 90.", finding)
    offences = inspect("The shoulders turned 90 degrees.", allowance_for(finding))
    assert "90" in [entry.token for entry in offences]


def test_a_ratio_is_read_as_its_quotient() -> None:
    """3.43:1 is one number, and checking the 3.43 and the 1 separately would

    demand that every finding's evidence contain the integer 1.
    """
    finding = _finding()
    finding.value = 1.831
    finding.unit = MetricUnit.RATIO
    assert passes("A tempo of 1.83:1.", finding)
    assert not passes("A tempo of 3.00:1.", finding)


def test_the_year_of_a_source_may_be_quoted() -> None:
    """A citation's date is evidence about the citation, not about the swing."""
    assert passes("The 2004 source puts this above its range.", _finding())


def test_an_empty_candidate_passes_and_that_is_the_caller_s_problem() -> None:
    """The guard checks what is said, not whether anything was.

    A provider returning an empty string is handled in `phrasing.py`, which knows
    the difference between a model that declined and a model that was never
    reached; the guard would have nothing to say about it either way.
    """
    assert passes("", _finding())
