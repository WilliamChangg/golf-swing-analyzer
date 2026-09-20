"""The optional language layer, and everything it is not allowed to be.

A language model can make a finding read better. That is the whole of what it is
for here, and the design follows from taking the sentence seriously:

**It never sees the video.** Not a frame, not a landmark, not a file path. What
it is handed is the finding as structured data -- the value, the band, the
bracket, the unit and the rule's own sentence -- and a swing is a recording of a
person's body, so the decision to send one anywhere is not a default and not a
convenience.

**It never runs by default, and never remotely.** `PhrasingMode.OFF` is the
default and is a complete configuration: every finding already has a sentence.
`LOCAL` talks to an endpoint on this machine that the caller names. There is no
hosted fallback and no key to configure, because a fallback is how "local-first"
becomes "local unless it is inconvenient".

**It cannot add anything.** Every candidate goes through `guard.inspect`, and a
candidate with one unsupported number is discarded whole -- not repaired, not
partially used. The rule's own sentence is what ships when it fails.

**It cannot replace the measurement.** `Finding.observation` stays in the report
whatever happens. `phrased` sits beside it. A reader who wants to know what was
measured reads the field the model could not touch.

## What this module has and has not been run against

The protocol, the request shape, the guard integration, the fallback and the
failure paths are exercised by the test suite against fake providers, including
ones that return exactly the plausible-looking inventions this layer exists to
catch. **No language model has ever phrased a finding in this repository**, because
none is installed on the machine it was built on. The counts in `PhrasingReport`
are there so that a later run on a machine that has one produces evidence rather
than an impression, and so that "the model rephrased nothing" can never be
confused with "no model was asked".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from analyzer.coaching import guard
from analyzer.contracts.coaching import (
    CoachingConfig,
    Finding,
    GuardRejection,
    PhrasingMode,
    PhrasingReport,
)


class PhrasingError(RuntimeError):
    """A language model was asked for and could not be used.

    Defined here rather than beside the HTTP client, because it is the layer's
    error and not the transport's: a misconfigured endpoint, a server that is not
    running and a reply that is not JSON are all the same fact to a caller, which
    is that the findings will carry the engine's own sentences.

    Never fatal to a coaching run. A phrasing layer that could fail the analysis
    would make an optional component load-bearing.
    """

    def __init__(self, message: str, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


class PhrasingProvider(Protocol):
    """Something that can propose a sentence for a finding.

    Deliberately narrow: one finding in, one string or None out. A provider
    handed the whole report could phrase two findings into one paragraph, and
    then no single sentence could be checked against a single finding's evidence
    -- which is exactly the property the guard depends on.
    """

    @property
    def name(self) -> str:
        """What produced the candidate, for the report."""
        ...

    def propose(self, finding: Finding, prompt: str) -> str | None:
        """A candidate phrasing, or None when the provider could not answer."""
        ...


def brief(finding: Finding) -> str:
    """Everything a provider is given about a finding, as JSON.

    Assembled by hand rather than by dumping the model, so that adding a field to
    `Finding` cannot quietly start sending something new. Frames are included as
    counts rather than indices: a provider has no use for frame 137, and every
    number it is shown is a number it may try to use.
    """
    payload = {
        "title": finding.title,
        "measured": round(finding.value, 4),
        "unit": finding.unit.value,
        "comparison": finding.comparison.value,
        "band_low": finding.band_low,
        "band_high": finding.band_high,
        "margin": round(finding.margin, 4),
        "bracket": round(finding.bracket, 4),
        "source": finding.source.citation,
        "population": finding.source.population,
        "sentence": finding.observation,
        "evidence": [
            {
                "metric": item.label,
                "value": round(item.value, 4),
                "unit": item.unit.value,
                "frames_cited": len(item.frames),
            }
            for item in finding.evidence
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


INSTRUCTION = (
    "Rewrite the sentence below so a golfer can read it, in at most two "
    "sentences. Use only the numbers given. Do not add a number, a unit, a "
    "cause, an instruction or anything about where the ball went. If you cannot "
    "do this, repeat the sentence unchanged."
)


def prompt_for(finding: Finding) -> str:
    """The whole of what a provider is sent. Nothing else reaches it."""
    return f"{INSTRUCTION}\n\n{brief(finding)}"


def phrase(
    findings: Sequence[Finding], config: CoachingConfig, provider: PhrasingProvider | None = None
) -> PhrasingReport:
    """Ask a provider to reword each finding, and keep what survives the guard.

    Mutates each finding's `phrased` field in place on acceptance, which is the
    only mutation in this package and is confined to a field the rest of the
    engine never reads.
    """
    if config.phrasing is PhrasingMode.OFF:
        return PhrasingReport(mode=PhrasingMode.OFF)

    if provider is None:
        provider = _default_provider(config)
    if provider is None:
        return PhrasingReport(
            mode=config.phrasing,
            unavailable=(
                "Phrasing was asked for and no local endpoint was configured. Set "
                "the endpoint of a model running on this machine, or leave phrasing "
                "off -- every finding already has its own sentence."
            ),
        )

    report = PhrasingReport(mode=config.phrasing, provider=provider.name)
    for finding in findings:
        report.attempted += 1
        try:
            candidate = provider.propose(finding, prompt_for(finding))
        except PhrasingError as exc:
            # The layer's own error already reads as a sentence; prefixing it
            # with a class name would tell a reader which exception type was
            # raised, which is the one thing they cannot act on. The remediation
            # rides along for the same reason.
            report.unavailable = " ".join(filter(None, (str(exc), exc.remediation)))
            break
        except Exception as exc:  # noqa: BLE001 - boundary: a provider is foreign code
            report.unavailable = f"{type(exc).__name__}: {exc}"
            break
        if candidate is None:
            report.rejected += 1
            report.rejections.append(
                GuardRejection(
                    rule_id=finding.rule_id,
                    offence="provider returned nothing",
                    token="",
                    candidate="",
                )
            )
            continue

        offences = guard.inspect(candidate, guard.allowance_for(finding))
        if offences:
            report.rejected += 1
            report.rejections.extend(
                GuardRejection(
                    rule_id=finding.rule_id,
                    offence=offence.offence,
                    token=offence.token,
                    candidate=candidate,
                )
                for offence in offences
            )
            continue

        finding.phrased = candidate.strip()
        report.accepted += 1

    if report.attempted and not report.accepted and report.unavailable is None:
        report.unavailable = (
            "Every candidate was rejected by the guard, so all of the findings "
            "above are the engine's own sentences."
        )
    return report


def _default_provider(config: CoachingConfig) -> PhrasingProvider | None:
    """The local HTTP provider, when one has been configured.

    Imported here rather than at module level so that nothing in the coaching
    package pulls in a network client on a run that never phrases anything --
    which is every run in the default configuration.
    """
    if config.phrasing_endpoint is None:
        return None
    from analyzer.coaching.local import LocalPhrasing

    return LocalPhrasing(
        endpoint=config.phrasing_endpoint,
        model=config.phrasing_model,
        timeout_s=config.phrasing_timeout_s,
    )
