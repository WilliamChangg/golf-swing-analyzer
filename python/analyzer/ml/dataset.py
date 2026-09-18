"""Turning labels and footage into arrays, with everything that made them recorded.

A dataset here is three things kept together: the features, the per-sample
targets, and a `DatasetSummary` saying which labels and which feature spec
produced them. They are kept together because the failure this layer exists to
prevent is a model trained on one and evaluated against another -- the same
arrays, the same shapes, no error anywhere, and a number that is about nothing.

## Targets are per frame, not per clip

The model predicts what each sample *is* -- address, backswing, downswing,
follow-through, or none of those -- and the four events come out as the
boundaries between runs of those classes. The alternative, a head per event
regressing a frame number, was not chosen: it cannot say anything about a clip
with no swing in it, and it has to be told how many swings a clip contains before
it can answer.

## A clip that cannot be featurised is dropped loudly

`DatasetSummary.dropped` names every labelled clip that did not make it and why.
A labelled clip is somebody's work; losing one to a silent exception means the
set that gets trained on is smaller than the set that was labelled, and nothing
downstream can tell.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.labels import ClipLabel, LabelProvenance, LabelSet
from analyzer.contracts.ml import (
    DATASET_SCHEMA_VERSION,
    FRAME_CLASSES,
    DatasetSummary,
    FeatureSpec,
    FrameClass,
)
from analyzer.contracts.phases import SwingEvent, SwingPhases
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.ml.features import FEATURE_SPEC, ClipFeatures, FeatureError, clip_features
from analyzer.phases import SignalError, detect_phases

# A callable that produces the filtered pose trajectories for one labelled clip.
# A Protocol-shaped seam, for the same reason `BallDetector` is one: the tests
# hand it synthetic sequences, the CLI hands it an extraction off the cache, and
# neither path is a special case of the other.
PoseProvider = Callable[[ClipLabel], FilteredSequence]

CLASS_INDEX: dict[FrameClass, int] = {value: index for index, value in enumerate(FRAME_CLASSES)}


class DatasetError(RuntimeError):
    """A dataset cannot be built from what was supplied."""


def filter_config_for(fps: float, base: FilterConfig | None = None) -> FilterConfig:
    """The filter settings a clip at this frame rate can actually support.

    The default 0.10 s window holds three samples at 30 fps and the degree-4 fit
    needs five, so every landmark of an ordinary phone clip comes back empty and
    the clip is dropped for having no torso. That is Phase 3's seam behaving
    exactly as documented -- it refuses rather than emitting a fit it cannot
    support -- and a dataset builder that met it would simply lose every 30 fps
    clip in the set, which is most of the footage anyone has.

    So the window is widened to the narrowest one that holds the samples the fit
    needs. This is not free and the cost is in the direction that matters:
    `SmoothingConfig` records that a wider window flattens the velocity peak, by
    13% at 0.30 s. A 30 fps clip therefore reaches the model with a slightly
    flatter speed channel than a 120 fps clip of the same swing.

    That confound is reported rather than removed. Removing it would mean
    widening *every* clip's window to match the worst, which throws away the
    resolution the better footage was captured for -- and the whole argument of
    Phases 10 and 11 is that the better footage is where the answers are.
    """
    resolved = base or FilterConfig()
    smoothing = resolved.smoothing
    needed = smoothing.min_observations or (smoothing.polyorder + 1)
    if fps <= 0.0 or smoothing.window_s * fps >= needed:
        return resolved

    # A hair over, so floating point cannot leave the window one sample short of
    # the count it was widened to reach.
    widened = (needed + 0.01) / fps
    return resolved.model_copy(
        update={"smoothing": smoothing.model_copy(update={"window_s": widened})}
    )


@dataclass(frozen=True)
class Example:
    """One clip, ready to train on or be scored."""

    label: ClipLabel
    features: ClipFeatures
    targets: NDArray[np.int64]
    """Class index per grid sample, in FRAME_CLASSES order."""
    rule: SwingPhases | None = None
    """What the rule-based detector made of the same filtered sequence.

    Computed here, at the same time and from the same trajectories the features
    came from, rather than in the comparison that uses it. That is what makes
    Phase 12.6 a comparison at all: two detectors reading one signal, rather than
    two numbers produced by two pipelines that differ in ways nobody enumerated.
    It costs milliseconds -- `scripts/benchmark_filter.py` measured the whole
    filtering layer at less than that -- so it is computed for every clip whether
    or not a comparison is ever run.
    """

    def __len__(self) -> int:
        return int(self.targets.size)

    @property
    def group_key(self) -> str:
        return self.label.group_key


@dataclass(frozen=True)
class Dataset:
    """Every example, and the record of what produced them."""

    spec: FeatureSpec
    examples: tuple[Example, ...]
    summary: DatasetSummary

    def __len__(self) -> int:
        return len(self.examples)

    def subset(self, keys: set[str]) -> tuple[Example, ...]:
        """The examples whose clip keys are in `keys`, in the dataset's own order."""
        return tuple(item for item in self.examples if item.label.clip_key in keys)


def frame_classes(label: ClipLabel, source_frames: NDArray[np.int64]) -> NDArray[np.int64]:
    """The class of each grid sample, from the frames a labeller marked.

    The tiling matches `phases/detect.py` exactly -- address from frame zero to
    the takeaway, the follow-through ending *on* the finish, everything after it
    outside any phase -- so a learned detector and the rule-based one are being
    asked the same question. If those two tilings ever drift apart, every
    comparison in Phase 12 silently becomes a comparison of two conventions.
    """
    classes = np.full(source_frames.size, CLASS_INDEX[FrameClass.NONE], dtype=np.int64)
    if not label.is_swing:
        return classes

    marks = {entry.event: entry.frame_index for entry in label.events}
    takeaway = marks[SwingEvent.TAKEAWAY]
    top = marks[SwingEvent.TOP]
    impact = marks[SwingEvent.IMPACT]
    finish = marks[SwingEvent.FINISH]

    classes[source_frames < takeaway] = CLASS_INDEX[FrameClass.ADDRESS]
    classes[(source_frames >= takeaway) & (source_frames < top)] = CLASS_INDEX[FrameClass.BACKSWING]
    classes[(source_frames >= top) & (source_frames < impact)] = CLASS_INDEX[FrameClass.DOWNSWING]
    classes[(source_frames >= impact) & (source_frames <= finish)] = CLASS_INDEX[
        FrameClass.FOLLOW_THROUGH
    ]
    return classes


def _median_uncertainty_ms(labels: list[ClipLabel]) -> float | None:
    """Median bracket over every marked event, in real milliseconds.

    Over events rather than over clips: a clip with four confident marks and one
    with four uncertain ones should not weigh the same, and what an evaluation
    needs is the typical precision of a *mark*.
    """
    brackets = [
        1000.0 * seconds
        for label in labels
        for event in (entry.event for entry in label.events)
        if (seconds := label.uncertainty_s(event)) is not None
    ]
    if not brackets:
        return None
    return float(np.median(brackets))


def build_dataset(
    labels: LabelSet,
    provider: PoseProvider,
    spec: FeatureSpec = FEATURE_SPEC,
) -> Dataset:
    """Featurise every labelled clip, and record what the result is made of.

    Raises only when there is nothing left: a set that produced no examples at
    all is not a small dataset, it is a failure, and returning an empty one would
    let a training run report having converged on nothing.
    """
    examples: list[Example] = []
    dropped: list[str] = []

    for label in sorted(labels.clips, key=lambda clip: clip.clip_key):
        try:
            filtered = provider(label)
        except (FeatureError, DatasetError) as exc:
            dropped.append(f"{label.clip_key}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - a provider reaches arbitrary I/O
            # Deliberately broad: a provider decodes video, reads caches and runs
            # a model, and any of those failing is one clip's problem rather than
            # the run's. The reason is kept verbatim in `dropped`, so nothing is
            # swallowed -- it is recorded and reported.
            dropped.append(f"{label.clip_key}: {type(exc).__name__}: {exc}")
            continue

        try:
            features = clip_features(filtered, spec)
        except FeatureError as exc:
            dropped.append(f"{label.clip_key}: {exc}")
            continue

        if label.is_swing and len(label.events) != 4:
            dropped.append(
                f"{label.clip_key}: labelled as a swing with {len(label.events)} of 4 events."
            )
            continue

        try:
            rule = detect_phases(filtered)
        except SignalError:
            # The rule detector refusing a clip is a legitimate answer and is
            # recorded as one. `None` here means it could not read a signal at
            # all, which the comparison scores as having detected nothing.
            rule = None

        examples.append(
            Example(
                label=label,
                features=features,
                targets=frame_classes(label, features.source_frames),
                rule=rule,
            )
        )

    if not examples:
        raise DatasetError(
            "No labelled clip produced features. "
            + (
                "; ".join(dropped)
                if dropped
                else "The label set is empty -- run `analyzer label` on a clip first."
            )
        )

    kept = [item.label for item in examples]
    counts = dict.fromkeys(FRAME_CLASSES, 0)
    for item in examples:
        for index, value in enumerate(FRAME_CLASSES):
            counts[value] += int(np.count_nonzero(item.targets == index))
    total = sum(counts.values())

    warnings: list[str] = []
    provenance = (
        LabelProvenance.HUMAN
        if kept and all(label.provenance is LabelProvenance.HUMAN for label in kept)
        else LabelProvenance.SYNTHETIC
    )
    if provenance is LabelProvenance.SYNTHETIC:
        warnings.append(
            "This dataset contains synthetic labels, so nothing measured on it is a "
            "statement about real swings. Every report derived from it is marked."
        )
    players = {label.player_id for label in kept}
    if len(players) < 2:
        warnings.append(
            f"Every clip comes from {len(players)} player(s). A set with one player "
            "cannot be split into training and held-out golfers, so no number measured "
            "on it describes anything but that person."
        )
    if not any(not label.is_swing for label in kept):
        warnings.append(
            "The set contains no negative clips. A detector that answers every clip "
            "with a swing cannot be caught by a set in which every clip has one."
        )
    poor = [item for item in examples if item.features.valid_fraction < 0.5]
    if poor:
        warnings.append(
            f"{len(poor)} clip(s) are tracked on under half their samples; the model is "
            "fed a validity channel rather than a guess for the rest."
        )
    windows = sorted({round(item.features.window_s, 4) for item in examples})
    if len(windows) > 1:
        warnings.append(
            "Clips were smoothed with different windows ("
            + ", ".join(f"{value * 1000:.0f} ms" for value in windows)
            + ") because their frame rates support different fits. The grid equalises "
            "the sample rate and not the smoothing, so the speed channel is slightly "
            "flatter on the clips that needed the wider window."
        )

    summary = DatasetSummary(
        schema_version=DATASET_SCHEMA_VERSION,
        provenance=provenance,
        feature_version=spec.version,
        feature_digest=spec.digest(),
        sample_rate_hz=spec.sample_rate_hz,
        label_digest=labels.digest(),
        clips=len(examples),
        swings=sum(1 for label in kept if label.is_swing),
        non_swings=sum(1 for label in kept if not label.is_swing),
        players=len(players),
        sessions=len({label.session_key for label in kept}),
        samples=total,
        class_samples=counts,
        class_fraction={
            value: (count / total if total else 0.0) for value, count in counts.items()
        },
        median_label_uncertainty_ms=_median_uncertainty_ms(kept),
        resample_floor_ms=1000.0 * spec.resample_floor_s,
        dropped=dropped,
        warnings=warnings,
    )
    return Dataset(spec=spec, examples=tuple(examples), summary=summary)
