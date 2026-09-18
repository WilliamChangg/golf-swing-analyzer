"""Labels and footage to arrays, and what the summary has to admit.

The tests that matter here are the ones about the tiling and about what happens
to a clip that cannot be featurised. A tiling that drifts from Phase 4's makes
every comparison in Phase 12 a comparison of two conventions; a clip silently
dropped makes the set smaller than the work that went into it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.labels import ClipLabel, LabelProvenance, LabelSet
from analyzer.contracts.ml import FrameClass
from analyzer.contracts.phases import SwingEvent
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.ml.dataset import (
    CLASS_INDEX,
    DatasetError,
    build_dataset,
    filter_config_for,
    frame_classes,
)
from analyzer.phases import detect_phases
from tests import synthetic
from tests.synthetic_labels import make_corpus


def small_corpus(**kwargs: object):  # type: ignore[no-untyped-def]
    return make_corpus(players=3, sessions=1, swings=2, seed=3, **kwargs)  # type: ignore[arg-type]


def test_the_tiling_matches_the_rule_detectors_exactly() -> None:
    """Both detectors must be answering one question, not two conventions.

    Phase 4 tiles address from frame zero to the takeaway, and ends the
    follow-through *on* the finish. A label tiled any other way would score the
    rule-based detector against a definition it does not use.
    """
    sequence = synthetic.swing_sequence(fps=120.0)
    filtered = filter_sequence(sequence, space=LandmarkSpace.FRAME_WIDTHS)
    phases = detect_phases(filtered)
    assert phases.detected

    marks = {entry.event: entry.frame_index for entry in phases.events}
    label = ClipLabel(
        clip_path="/x.mov",
        player_id="p",
        session_id="s",
        swing_id="w",
        provenance=LabelProvenance.SYNTHETIC,
        labeller="t",
        labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
        is_swing=True,
        events=[
            {"event": event, "frame_index": frame, "uncertainty_frames": 0}  # type: ignore[list-item]
            for event, frame in marks.items()
        ],
        frames=len(sequence.frames),
        fps=120.0,
    )

    frames = np.arange(len(sequence.frames), dtype=np.int64)
    from_label = frame_classes(label, frames)
    for index, frame in enumerate(frames):
        phase = phases.phase_at(int(frame))
        expected = FrameClass.NONE if phase is None else FrameClass(phase.value)
        assert from_label[index] == CLASS_INDEX[expected], f"frame {frame}"


def test_a_clip_with_no_swing_is_every_sample_none() -> None:
    corpus = small_corpus(non_swing_every=1)
    built = build_dataset(corpus.labels, corpus.provider())
    for example in built.examples:
        assert not example.label.is_swing
        assert np.all(example.targets == CLASS_INDEX[FrameClass.NONE])


def test_every_class_is_present_in_a_mixed_corpus() -> None:
    corpus = small_corpus()
    built = build_dataset(corpus.labels, corpus.provider())
    assert all(count > 0 for count in built.summary.class_samples.values())
    assert built.summary.imbalance_ratio > 1.0


def test_the_rule_detectors_verdict_is_attached_to_every_example() -> None:
    """Computed here, from the same trajectories, so 12.6 compares one signal."""
    corpus = small_corpus()
    built = build_dataset(corpus.labels, corpus.provider())
    assert any(example.rule is not None and example.rule.detected for example in built.examples)


def test_a_clip_that_cannot_be_featurised_is_named_rather_than_lost() -> None:
    corpus = small_corpus()

    def provider(label: ClipLabel):  # type: ignore[no-untyped-def]
        if label.swing_id == "w1":
            raise RuntimeError("the pose file is missing")
        return filter_sequence(
            corpus.sequences[label.clip_key],
            filter_config_for(label.fps),
            space=LandmarkSpace.FRAME_WIDTHS,
        )

    built = build_dataset(corpus.labels, provider)
    assert built.summary.dropped
    assert all("the pose file is missing" in entry for entry in built.summary.dropped)
    assert built.summary.clips == len(corpus.labels.clips) - len(built.summary.dropped)


def test_a_set_that_produces_nothing_is_an_error_rather_than_an_empty_dataset() -> None:
    corpus = small_corpus()

    def provider(_label: ClipLabel):  # type: ignore[no-untyped-def]
        raise RuntimeError("nothing here")

    with pytest.raises(DatasetError, match="No labelled clip produced features"):
        build_dataset(corpus.labels, provider)


def test_an_empty_label_set_says_what_to_run() -> None:
    with pytest.raises(DatasetError, match="analyzer label"):
        build_dataset(LabelSet(clips=[]), lambda label: None)  # type: ignore[arg-type,return-value]


def test_the_summary_records_the_labels_and_the_features_it_was_built_from() -> None:
    corpus = small_corpus()
    built = build_dataset(corpus.labels, corpus.provider())
    assert built.summary.label_digest == corpus.labels.digest()
    assert built.summary.feature_digest == built.spec.digest()
    assert built.summary.provenance is LabelProvenance.SYNTHETIC


def test_a_synthetic_set_warns_that_nothing_measured_on_it_is_about_golf() -> None:
    corpus = small_corpus()
    built = build_dataset(corpus.labels, corpus.provider())
    assert any("synthetic" in note for note in built.summary.warnings)


def test_a_single_player_set_warns_that_it_cannot_be_split() -> None:
    corpus = make_corpus(players=1, sessions=2, swings=2, seed=0)
    built = build_dataset(corpus.labels, corpus.provider())
    assert any("cannot be split" in note for note in built.summary.warnings)


def test_a_set_with_no_negatives_says_so() -> None:
    corpus = make_corpus(players=2, sessions=1, swings=2, seed=0, non_swing_every=0)
    built = build_dataset(corpus.labels, corpus.provider())
    assert any("no negative clips" in note for note in built.summary.warnings)


def test_mixed_frame_rates_report_the_different_smoothing_windows() -> None:
    """The grid equalises the sample rate; it cannot equalise the filter."""
    corpus = make_corpus(players=6, sessions=3, swings=1, seed=1)
    built = build_dataset(corpus.labels, corpus.provider())
    assert any("different windows" in note for note in built.summary.warnings)


def test_a_thirty_fps_clip_is_widened_rather_than_dropped() -> None:
    """The default window holds three samples at 30 fps and the fit needs five."""
    default = FilterConfig()
    widened = filter_config_for(30.0)
    assert widened.smoothing.window_s > default.smoothing.window_s
    assert widened.smoothing.window_s * 30.0 >= default.smoothing.polyorder + 1

    assert filter_config_for(240.0).smoothing.window_s == default.smoothing.window_s


def test_a_swing_labelled_with_missing_events_is_dropped_with_a_reason() -> None:
    corpus = small_corpus()
    clips = list(corpus.labels.clips)
    broken = clips[0].model_copy(
        update={"events": [entry for entry in clips[0].events if entry.event is not SwingEvent.TOP]}
    )
    corpus_labels = LabelSet(clips=[broken, *clips[1:]])

    built = build_dataset(corpus_labels, corpus.provider())
    assert any("of 4 events" in entry for entry in built.summary.dropped)
