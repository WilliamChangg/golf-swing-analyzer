"""Scoring a detector, and deciding whether the score may be quoted.

Two kinds of number come out of here and they answer different questions.

**Per-frame** numbers say how well a detector classifies samples. They are
reported with a macro F1 rather than an accuracy, because a golf clip is mostly
address and mostly follow-through, and a detector that answers "address" to
everything can score over half on accuracy.

**Per-event** numbers say how far the detector put the takeaway, the top, impact
and the finish from where a person put them. This is the question Phase 12
exists to answer, and it is the one a per-frame score can pass while failing: a
model that gets 97% of samples right can still place impact ten frames out,
because impact is one boundary in a clip of several hundred samples.

## The noise floor

An error smaller than the labels' own uncertainty is not an achievement, it is a
measurement below the resolution of the instrument. The floor here has two parts
and they add:

    label uncertainty   what the labeller said they could see
    resample floor      half a sample of the grid the model works on

A detector whose mean error is under that sum cannot be distinguished from the
labels, and `claims_permitted` goes false with the reason recorded. So does a
score from a synthetic set, from too few players, or from too few held-out clips.
These are the project's standing honesty gates -- `docs/ROADMAP.md` names them --
placed where a caller cannot route around them by forgetting.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import torch
from numpy.typing import NDArray

from analyzer.contracts.labels import LabelProvenance
from analyzer.contracts.ml import (
    FRAME_CLASSES,
    MIN_PLAYERS_FOR_CLAIM,
    MIN_TEST_CLIPS_FOR_CLAIM,
    ClassMetrics,
    EvaluationReport,
    EventError,
    FrameClass,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase, SwingPhases
from analyzer.ml.dataset import CLASS_INDEX, Example
from analyzer.ml.features import FEATURE_SPEC
from analyzer.ml.tcn import SwingTCN

# Which class each event begins. The takeaway is the first sample of the
# backswing, the top the first of the downswing, impact the first of the
# follow-through; the finish is the last sample of the follow-through and is
# handled separately because it is an end rather than a beginning.
_EVENT_STARTS: dict[SwingEvent, FrameClass] = {
    SwingEvent.TAKEAWAY: FrameClass.BACKSWING,
    SwingEvent.TOP: FrameClass.DOWNSWING,
    SwingEvent.IMPACT: FrameClass.FOLLOW_THROUGH,
}

_PHASE_CLASS: dict[SwingPhase, FrameClass] = {
    SwingPhase.ADDRESS: FrameClass.ADDRESS,
    SwingPhase.BACKSWING: FrameClass.BACKSWING,
    SwingPhase.DOWNSWING: FrameClass.DOWNSWING,
    SwingPhase.FOLLOW_THROUGH: FrameClass.FOLLOW_THROUGH,
}


def predict_classes(model: SwingTCN, example: Example, device: str = "cpu") -> NDArray[np.int64]:
    """The model's class for every grid sample of one clip."""
    from analyzer.environment.hardware import select_torch_device

    device, _ = select_torch_device(device)
    model.to(torch.device(device))
    model.eval()
    values = np.ascontiguousarray(example.features.values.T)
    tensor = torch.from_numpy(values).unsqueeze(0).to(device=torch.device(device))
    with torch.no_grad():
        logits = model(tensor)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def rule_classes(example: Example) -> NDArray[np.int64]:
    """The rule-based detector's class for every grid sample of one clip.

    Read off `SwingPhases` through the same grid-to-frame mapping the model's
    predictions come back through, so the two detectors are scored in one
    coordinate system. A clip the rules refused is every sample NONE, which is
    the correct reading of a refusal rather than a penalty applied to it: the
    detector said there was no swing here.
    """
    phases: SwingPhases | None = example.rule
    classes = np.full(len(example.features), CLASS_INDEX[FrameClass.NONE], dtype=np.int64)
    if phases is None or not phases.detected:
        return classes
    for index, frame in enumerate(example.features.source_frames):
        phase = phases.phase_at(int(frame))
        if phase is not None:
            classes[index] = CLASS_INDEX[_PHASE_CLASS[phase]]
    return classes


def _longest_run(mask: NDArray[np.bool_]) -> tuple[int, int] | None:
    """Start and stop (exclusive) of the longest unbroken run of True, or None.

    The longest run rather than the first: a per-sample classifier flickers, and
    a single stray backswing sample early in the address would otherwise place
    the takeaway there. Flicker is real and this is the cheapest defence against
    it that does not smooth the prediction -- smoothing would move the boundary
    being measured, which is the one thing that must not be touched.
    """
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and (best is None or mask.size - start > best[1] - best[0]):
        best = (start, int(mask.size))
    return best


def events_from_classes(classes: NDArray[np.int64], example: Example) -> dict[SwingEvent, int]:
    """Frame numbers for whichever events a class sequence contains.

    Frames of the source clip, not samples of the grid, because that is the unit
    a labeller marked and the unit somebody can check by stepping the video.

    Events the sequence does not contain are absent from the result rather than
    guessed at. A prediction with no downswing samples has no top, and inventing
    one at the midpoint of what is there would convert a visible failure into a
    large error that looks like a small one.
    """
    frames = example.features.source_frames
    found: dict[SwingEvent, int] = {}

    for event, frame_class in _EVENT_STARTS.items():
        run = _longest_run(classes == CLASS_INDEX[frame_class])
        if run is not None:
            found[event] = int(frames[run[0]])

    follow = _longest_run(classes == CLASS_INDEX[FrameClass.FOLLOW_THROUGH])
    if follow is not None:
        found[SwingEvent.FINISH] = int(frames[follow[1] - 1])
    return found


def _confusion(
    predictions: Sequence[NDArray[np.int64]], examples: Sequence[Example]
) -> tuple[NDArray[np.int64], int]:
    """Counts of true class against predicted, over tracked samples only.

    Samples where the subject was not tracked are excluded from both the loss and
    this matrix, so a detector is not credited or penalised for what it said
    about a stretch of zeros. The excluded count is returned and reported.
    """
    size = len(FRAME_CLASSES)
    matrix = np.zeros((size, size), dtype=np.int64)
    scored = 0
    for predicted, example in zip(predictions, examples, strict=True):
        mask = example.features.valid
        truth = example.targets[mask]
        guess = predicted[mask]
        scored += int(truth.size)
        for true_index, predicted_index in zip(truth, guess, strict=True):
            matrix[true_index, predicted_index] += 1
    return matrix, scored


def _class_metrics(matrix: NDArray[np.int64]) -> list[ClassMetrics]:
    metrics: list[ClassMetrics] = []
    for index, frame_class in enumerate(FRAME_CLASSES):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        hit = int(matrix[index, index])
        precision = hit / predicted if predicted else 0.0
        recall = hit / support if support else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics.append(
            ClassMetrics(
                frame_class=frame_class,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return metrics


def _event_errors(
    predictions: Sequence[NDArray[np.int64]], examples: Sequence[Example]
) -> tuple[list[EventError], int]:
    """Per-event localisation, and how many clips came back out of order."""
    deltas: dict[SwingEvent, list[tuple[float, float, float]]] = {event: [] for event in SwingEvent}
    missed = dict.fromkeys(SwingEvent, 0)
    spurious = dict.fromkeys(SwingEvent, 0)
    out_of_order = 0

    for predicted, example in zip(predictions, examples, strict=True):
        found = events_from_classes(predicted, example)
        order = [found[event] for event in SwingEvent if event in found]
        if any(later <= earlier for earlier, later in pairwise(order)):
            out_of_order += 1

        per_frame_ms = 1000.0 * example.label.real_seconds_per_frame
        for event in SwingEvent:
            truth = example.label.event(event)
            guess = found.get(event)
            if truth is None and guess is not None:
                spurious[event] += 1
                continue
            if truth is None:
                continue
            if guess is None:
                missed[event] += 1
                continue
            delta_frames = float(guess - truth.frame_index)
            bracket_ms = truth.uncertainty_frames * per_frame_ms
            deltas[event].append((delta_frames, delta_frames * per_frame_ms, bracket_ms))

    errors: list[EventError] = []
    for event in SwingEvent:
        rows = deltas[event]
        if not rows:
            errors.append(
                EventError(event=event, scored=0, missed=missed[event], spurious=spurious[event])
            )
            continue
        frames = np.array([row[0] for row in rows])
        milliseconds = np.array([row[1] for row in rows])
        brackets = np.array([row[2] for row in rows])
        errors.append(
            EventError(
                event=event,
                scored=len(rows),
                missed=missed[event],
                spurious=spurious[event],
                median_frames=float(np.median(np.abs(frames))),
                mae_frames=float(np.mean(np.abs(frames))),
                mae_ms=float(np.mean(np.abs(milliseconds))),
                p90_ms=float(np.percentile(np.abs(milliseconds), 90)),
                bias_ms=float(np.mean(milliseconds)),
                within_label_uncertainty=float(
                    np.mean(np.abs(milliseconds) <= np.maximum(brackets, 1e-9))
                ),
            )
        )
    return errors, out_of_order


def noise_floor_ms(examples: Sequence[Example]) -> float:
    """The smallest error this set could distinguish, in real milliseconds.

    The labels' median bracket plus half a sample of the feature grid. Added
    rather than maxed: they are independent sources of the same kind of
    displacement -- the labeller's frame may be off by one, and the grid rounds
    to the nearest frame again -- and a floor that ignored either would license a
    claim the set cannot support.
    """
    brackets = [
        1000.0 * seconds
        for example in examples
        for event in (entry.event for entry in example.label.events)
        if (seconds := example.label.uncertainty_s(event)) is not None
    ]
    label_part = float(np.median(brackets)) if brackets else 0.0
    return label_part + 1000.0 * FEATURE_SPEC.resample_floor_s


def evaluate(
    detector: str,
    predictions: Sequence[NDArray[np.int64]],
    examples: Sequence[Example],
    *,
    label_digest: str = "",
) -> EvaluationReport:
    """Score one detector on one held-out set, and decide what may be said about it."""
    if not examples:
        raise ValueError("There is nothing to evaluate: the held-out set is empty.")

    matrix, scored = _confusion(predictions, examples)
    metrics = _class_metrics(matrix)
    supported = [entry for entry in metrics if entry.support > 0]
    errors, out_of_order = _event_errors(predictions, examples)

    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    macro = float(np.mean([entry.f1 for entry in supported])) if supported else 0.0

    players = {example.label.player_id for example in examples}
    provenance = (
        LabelProvenance.HUMAN
        if all(example.label.provenance is LabelProvenance.HUMAN for example in examples)
        else LabelProvenance.SYNTHETIC
    )
    floor = noise_floor_ms(examples)

    warnings: list[str] = []
    if out_of_order:
        warnings.append(
            f"{out_of_order} of {len(examples)} clips produced events out of order -- a "
            "top after impact, or similar. Nothing here repairs that; the rule-based "
            "detector cannot produce it at all, which is a real difference between them."
        )
    untracked = sum(len(example.features) for example in examples) - scored
    if untracked:
        warnings.append(
            f"{untracked} of {untracked + scored} samples were not scored because the "
            "subject was not tracked in them. Both detectors are scored on the same ones."
        )
    empty = [entry.frame_class.value for entry in metrics if entry.support == 0]
    if empty:
        warnings.append(
            f"No held-out samples of: {', '.join(empty)}. Those classes are excluded from "
            "the macro F1 rather than scored zero."
        )

    refusal = _claim_refusal(provenance, players, examples, errors, floor)
    return EvaluationReport(
        detector=detector,
        provenance=provenance,
        clips=len(examples),
        players=len(players),
        samples=scored,
        frame_accuracy=accuracy,
        macro_f1=macro,
        classes=metrics,
        confusion=[[int(value) for value in row] for row in matrix],
        events=errors,
        noise_floor_ms=floor,
        claims_permitted=refusal is None,
        claim_refusal=refusal,
        label_digest=label_digest,
        feature_digest=FEATURE_SPEC.digest(),
        warnings=warnings,
    )


def _claim_refusal(
    provenance: LabelProvenance,
    players: set[str],
    examples: Sequence[Example],
    errors: Sequence[EventError],
    floor: float,
) -> str | None:
    """The first reason this score may not be published, or None.

    First rather than all: the reasons are ordered from the most disqualifying
    down, and a caller who fixes the top one gets the next. Listing them all
    would invite fixing the easy ones.
    """
    if provenance is LabelProvenance.SYNTHETIC:
        return (
            "The held-out clips are synthetic. Their events are inputs to the generator "
            "that drew the motion, so a score against them measures whether the machinery "
            "is wired correctly, not whether a detector works on a golf swing."
        )
    if len(players) < 2:
        return (
            f"The held-out set contains {len(players)} player(s). A number measured on one "
            "golfer describes that golfer."
        )
    if len(players) < MIN_PLAYERS_FOR_CLAIM:
        return (
            f"{len(players)} held-out players, below the {MIN_PLAYERS_FOR_CLAIM} this "
            "project treats as the minimum before a rate is quoted."
        )
    if len(examples) < MIN_TEST_CLIPS_FOR_CLAIM:
        return (
            f"{len(examples)} held-out clips, below the {MIN_TEST_CLIPS_FOR_CLAIM} a rate "
            "could be quoted from."
        )
    inside = [
        entry
        for entry in errors
        if entry.mae_ms is not None and entry.scored > 0 and entry.mae_ms < floor
    ]
    if inside:
        names = ", ".join(entry.event.value for entry in inside)
        return (
            f"The error on {names} is below the {floor:.1f} ms this set can resolve -- the "
            "labels' own bracket plus the resampling floor. The detector cannot be shown "
            "to be more accurate than the numbers it is being scored against; a better "
            "figure here would need better labels, not a better model."
        )
    return None
