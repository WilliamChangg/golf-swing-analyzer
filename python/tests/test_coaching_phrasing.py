"""The optional language layer: what it is sent, what it may return, and where it may live.

No language model runs in this suite, and none ran in this project. The providers
here are three lines each, and between them they cover every way the layer can
behave: a good rewording, an invented number, silence, and a server that is not
there. That is the whole state space, because the guard collapses everything a
real model could do into "passed" or "did not".

The tests that matter most are the two about what leaves the machine. A phrasing
layer is the first component in this system that could send a measurement of
somebody's body somewhere, and the checks on that are worth more than the ones on
whether the sentence reads nicely.
"""

from __future__ import annotations

import json

import pytest

from analyzer.coaching.local import LocalPhrasing, PhrasingError
from analyzer.coaching.phrasing import brief, phrase, prompt_for
from analyzer.contracts.coaching import (
    CoachingConfig,
    Comparison,
    Evidence,
    Finding,
    PhrasingMode,
    ThresholdMethod,
    ThresholdSource,
)
from analyzer.contracts.metrics import MetricBasis, MetricName, MetricUnit
from analyzer.contracts.phases import SwingPhase


def _finding() -> Finding:
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
        source=ThresholdSource(
            citation="A fixture",
            year=2004,
            population="Fictional golfers",
            method=ThresholdMethod.TWO_D_VIDEO,
            measures="Frames.",
            permitted_bases=[MetricBasis.TEMPORAL],
        ),
        evidence=[
            Evidence(
                metric=MetricName.DOWNSWING_DURATION,
                label="Downswing duration",
                value=0.3667,
                unit=MetricUnit.SECONDS,
                basis=MetricBasis.TEMPORAL,
                phase=SwingPhase.DOWNSWING,
                confidence=0.87,
                frames=[40, 49],
                timestamps_s=[1.333, 1.633],
                methodology="Between the top and impact.",
            )
        ],
    )


class Fake:
    """A provider that answers with whatever it was constructed with."""

    def __init__(self, answer: str | None, fail: Exception | None = None) -> None:
        self._answer = answer
        self._fail = fail
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def propose(self, finding: Finding, prompt: str) -> str | None:
        self.prompts.append(prompt)
        if self._fail is not None:
            raise self._fail
        return self._answer


def test_off_is_a_complete_configuration() -> None:
    """Nothing is attempted, nothing is reported as having failed."""
    findings = [_finding()]
    report = phrase(findings, CoachingConfig(), Fake("anything"))
    assert report.mode is PhrasingMode.OFF
    assert report.attempted == 0
    assert findings[0].phrased is None


def test_asking_for_a_model_without_naming_one_says_so() -> None:
    """Rather than silently producing the default and calling it a model's work."""
    report = phrase([_finding()], CoachingConfig(phrasing=PhrasingMode.LOCAL))
    assert report.attempted == 0
    assert report.unavailable is not None
    assert "endpoint" in report.unavailable


def test_a_faithful_rewording_is_kept_beside_the_measurement() -> None:
    findings = [_finding()]
    report = phrase(
        findings,
        CoachingConfig(phrasing=PhrasingMode.LOCAL),
        Fake("Your downswing took 0.367 s, which is 0.100 s longer than the range."),
    )
    assert report.accepted == 1
    assert report.rejected == 0
    assert findings[0].phrased is not None
    # The measurement survives regardless: `observation` is never replaced.
    assert findings[0].observation.startswith("The downswing measured")


def test_an_invented_number_is_discarded_whole() -> None:
    """Not repaired, not partially used. The rule's own sentence ships."""
    findings = [_finding()]
    report = phrase(
        findings,
        CoachingConfig(phrasing=PhrasingMode.LOCAL),
        Fake("Your downswing took 0.367 s against a tour average of 0.250 s."),
    )
    assert report.accepted == 0
    assert report.rejected == 1
    assert findings[0].phrased is None
    assert "0.250" in [entry.token for entry in report.rejections]
    # The candidate is kept so a person can see what the model actually said.
    assert "tour average" in report.rejections[0].candidate


def test_a_provider_that_declines_is_not_a_provider_that_failed() -> None:
    report = phrase([_finding()], CoachingConfig(phrasing=PhrasingMode.LOCAL), Fake(None))
    assert report.rejected == 1
    assert report.rejections[0].offence == "provider returned nothing"
    assert report.unavailable is not None


def test_a_provider_that_raises_stops_the_pass_and_is_recorded() -> None:
    """Continuing would produce a report whose rejection count meant nothing.

    Nine findings attempted against a server that is not there is one fact, not
    nine, and a rejection rate computed over it would be a measurement of an
    outage.
    """
    findings = [_finding(), _finding()]
    report = phrase(
        findings,
        CoachingConfig(phrasing=PhrasingMode.LOCAL),
        Fake(None, fail=PhrasingError("nothing is listening")),
    )
    assert report.attempted == 1
    assert report.unavailable is not None
    assert "nothing is listening" in report.unavailable
    assert all(entry.phrased is None for entry in findings)


def test_a_pass_that_kept_nothing_says_so_rather_than_looking_successful() -> None:
    report = phrase(
        [_finding()],
        CoachingConfig(phrasing=PhrasingMode.LOCAL),
        Fake("This will cost you 30 yards of carry."),
    )
    assert report.accepted == 0
    assert report.unavailable is not None
    assert "guard" in report.unavailable


# --- what leaves the machine ----------------------------------------------


def test_the_brief_contains_the_finding_and_nothing_else() -> None:
    """Asserted key by key, so a field added to `Finding` cannot leak silently.

    The brief is assembled by hand rather than dumped from the model for exactly
    this reason, and a test that only checked for the absence of a file path
    would not notice the next field that should not be sent.
    """
    payload = json.loads(brief(_finding()))
    assert set(payload) == {
        "title",
        "measured",
        "unit",
        "comparison",
        "band_low",
        "band_high",
        "margin",
        "bracket",
        "source",
        "population",
        "sentence",
        "evidence",
    }
    assert set(payload["evidence"][0]) == {"metric", "value", "unit", "frames_cited"}


def test_no_frame_index_or_timestamp_reaches_a_provider() -> None:
    """A provider has no use for frame 40, and every number it sees it may use.

    The evidence carries frame indices and real-clock instants so that a person
    can check the finding against the video. Sending them would hand a model
    small integers it could spend anywhere in a sentence.
    """
    text = prompt_for(_finding())
    assert "40" not in text
    assert "1.333" not in text
    assert "frames_cited" in text


def test_the_prompt_carries_the_engine_s_own_sentence_as_the_thing_to_rewrite() -> None:
    text = prompt_for(_finding())
    assert "The downswing measured 0.367 s" in text
    assert "Do not add a number" in text


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/api/generate",
        "http://192.168.1.50:11434/api/generate",
        "https://api.some-model-host.com/v1/complete",
        # A remote host with a reassuring path. Checked on the parsed hostname,
        # so this is refused rather than read as local.
        "http://evil.example.com/localhost/api/generate",
    ],
)
def test_an_endpoint_that_is_not_this_machine_is_refused(endpoint: str) -> None:
    """The one place in this system that opens a socket for a finding.

    If "local-first" is not enforced here it is a sentence in a README.
    """
    with pytest.raises(PhrasingError) as raised:
        LocalPhrasing(endpoint, model=None, timeout_s=1.0)
    assert "not on this machine" in str(raised.value)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:11434/api/generate",
        "http://127.0.0.1:11434/api/generate",
        "http://[::1]:11434/api/generate",
    ],
)
def test_loopback_is_accepted(endpoint: str) -> None:
    """Constructed only. Nothing here connects to anything."""
    assert LocalPhrasing(endpoint, model="a-model", timeout_s=1.0).name == "local:a-model"


def test_a_non_http_endpoint_is_refused() -> None:
    with pytest.raises(PhrasingError):
        LocalPhrasing("file:///etc/passwd", model=None, timeout_s=1.0)
