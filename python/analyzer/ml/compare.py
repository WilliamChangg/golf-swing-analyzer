"""The learned detector against the rules, on the same clips, with the same metric.

Phase 12.6, and the only sub-item that justifies the rest. A model's score means
nothing on its own: the question is whether it beats what the project already
had, which is two hundred lines of readable rules that never need training data
and never surprise anybody.

Three things make this a comparison rather than two numbers:

**One set.** Both detectors are scored on exactly the held-out clips, never on
sets assembled separately.

**One signal.** The rule-based result was computed in `dataset.py` from the same
filtered trajectories the features were built from, so the two detectors differ
in what they do with a signal rather than in which signal they got.

**One verdict word.** `ComparisonRow.verdict` is "indistinguishable" whenever the
gap between them is smaller than what the labels can resolve, which on a small
set is the usual and correct answer. A narrow win reported as a win is how a
project talks itself into believing a model earned its place.
"""

from __future__ import annotations

from collections.abc import Sequence

from analyzer.contracts.labels import LabelProvenance
from analyzer.contracts.ml import ComparisonRow, DetectorComparison
from analyzer.contracts.phases import SwingEvent
from analyzer.ml.dataset import Example
from analyzer.ml.evaluate import evaluate, noise_floor_ms, predict_classes, rule_classes
from analyzer.ml.tcn import SwingTCN

RULE_DETECTOR = "rule-based (phase 4)"


def compare_detectors(
    model: SwingTCN,
    examples: Sequence[Example],
    *,
    model_id: str,
    device: str = "cpu",
    label_digest: str = "",
) -> DetectorComparison:
    """Score both detectors on one held-out set and say what the difference means."""
    if not examples:
        raise ValueError("A comparison needs a held-out set; this one is empty.")

    rule_report = evaluate(
        RULE_DETECTOR,
        [rule_classes(example) for example in examples],
        examples,
        label_digest=label_digest,
    )
    model_report = evaluate(
        model_id,
        [predict_classes(model, example, device) for example in examples],
        examples,
        label_digest=label_digest,
    )

    floor = noise_floor_ms(examples)
    rows: list[ComparisonRow] = []
    for event in SwingEvent:
        rule_error = rule_report.event_error(event)
        model_error = model_report.event_error(event)
        rule_ms = rule_error.mae_ms if rule_error else None
        model_ms = model_error.mae_ms if model_error else None
        scored = min(
            rule_error.scored if rule_error else 0, model_error.scored if model_error else 0
        )

        if rule_ms is None or model_ms is None or scored == 0:
            rows.append(
                ComparisonRow(
                    event=event,
                    scored=scored,
                    rule_mae_ms=rule_ms,
                    model_mae_ms=model_ms,
                    noise_floor_ms=floor,
                    verdict="not scored",
                )
            )
            continue

        difference = rule_ms - model_ms
        if abs(difference) < floor:
            verdict = "indistinguishable"
        else:
            verdict = "model" if difference > 0 else "rule"
        rows.append(
            ComparisonRow(
                event=event,
                scored=scored,
                rule_mae_ms=rule_ms,
                model_mae_ms=model_ms,
                difference_ms=difference,
                noise_floor_ms=floor,
                verdict=verdict,
            )
        )

    provenance = (
        LabelProvenance.HUMAN
        if all(example.label.provenance is LabelProvenance.HUMAN for example in examples)
        else LabelProvenance.SYNTHETIC
    )
    permitted = rule_report.claims_permitted and model_report.claims_permitted

    warnings: list[str] = []
    if not permitted:
        warnings.append(
            "Neither score may be published: "
            + (model_report.claim_refusal or rule_report.claim_refusal or "")
        )
    missing = [row.event.value for row in rows if row.verdict == "not scored"]
    if missing:
        warnings.append(
            f"Not scored on {', '.join(missing)} -- one of the detectors found nothing to "
            "compare on any held-out clip, which is itself a difference between them."
        )

    return DetectorComparison(
        provenance=provenance,
        clips=len(examples),
        players=len({example.label.player_id for example in examples}),
        model_id=model_id,
        rule=rule_report,
        model=model_report,
        rows=rows,
        verdict=_verdict(rows, permitted),
        claims_permitted=permitted,
        warnings=warnings,
    )


def _verdict(rows: Sequence[ComparisonRow], permitted: bool) -> str:
    """One sentence, and it is allowed to say that nothing was established."""
    scored = [row for row in rows if row.verdict != "not scored"]
    if not scored:
        return "Neither detector produced comparable events on this set."

    wins = sum(1 for row in scored if row.verdict == "model")
    losses = sum(1 for row in scored if row.verdict == "rule")
    level = sum(1 for row in scored if row.verdict == "indistinguishable")
    summary = (
        f"Of {len(scored)} events scored, the model is closer on {wins}, the rules on "
        f"{losses}, and {level} are inside what the labels can resolve."
    )
    if not permitted:
        return summary + " None of it may be quoted as a result; see the refusal."
    return summary
