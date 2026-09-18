"""Scoring, the gates on what a score may claim, and the comparison.

The most important tests in this file are the ones where the detector is
*perfect* and the report still refuses to publish the number. That is the shape
of the whole phase: the machinery works, and the data does not entitle anyone to
a figure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analyzer.contracts.labels import LabelProvenance
from analyzer.contracts.ml import FRAME_CLASSES, FrameClass, TCNConfig
from analyzer.contracts.phases import SwingEvent
from analyzer.ml.compare import compare_detectors
from analyzer.ml.dataset import CLASS_INDEX, build_dataset
from analyzer.ml.evaluate import (
    evaluate,
    events_from_classes,
    noise_floor_ms,
    predict_classes,
    rule_classes,
)
from analyzer.ml.splits import split_dataset
from analyzer.ml.train import train
from tests.synthetic_labels import make_corpus

FAST = TCNConfig(hidden_channels=8, layers=3, epochs=2, dropout=0.0)


@pytest.fixture(scope="module")
def built():  # type: ignore[no-untyped-def]
    corpus = make_corpus(players=4, sessions=1, swings=2, seed=13)
    return build_dataset(corpus.labels, corpus.provider())


@pytest.fixture(scope="module")
def swing_example(built):  # type: ignore[no-untyped-def]
    return next(example for example in built.examples if example.label.is_swing)


def test_a_perfect_detector_scores_zero_error(built) -> None:  # type: ignore[no-untyped-def]
    """Predicting the targets themselves is the upper bound the metric must show."""
    report = evaluate("oracle", [example.targets for example in built.examples], built.examples)
    assert report.frame_accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    for error in report.events:
        if error.scored:
            assert error.mae_frames == pytest.approx(0.0, abs=1.0)


def test_a_perfect_detector_still_may_not_publish_a_number(built) -> None:  # type: ignore[no-untyped-def]
    """The honesty gate is on the data, not on the score."""
    report = evaluate("oracle", [example.targets for example in built.examples], built.examples)
    assert report.provenance is LabelProvenance.SYNTHETIC
    assert not report.claims_permitted
    assert "synthetic" in (report.claim_refusal or "")


def test_the_refusal_names_the_most_disqualifying_reason_first(built) -> None:  # type: ignore[no-untyped-def]
    """A caller who fixes the top one gets the next, rather than a list to shop from."""
    pretend_human = [
        replace(
            example,
            label=example.label.model_copy(update={"provenance": LabelProvenance.HUMAN}),
        )
        for example in built.examples
    ]
    report = evaluate("oracle", [example.targets for example in pretend_human], pretend_human)
    assert not report.claims_permitted
    assert "held-out players" in (report.claim_refusal or "")


def test_the_noise_floor_is_the_labels_plus_the_grid(built) -> None:  # type: ignore[no-untyped-def]
    floor = noise_floor_ms(built.examples)
    assert floor > 1000 * built.spec.resample_floor_s
    report = evaluate("oracle", [example.targets for example in built.examples], built.examples)
    assert report.noise_floor_ms == pytest.approx(floor)


def test_events_come_back_as_frames_of_the_source_clip(swing_example) -> None:  # type: ignore[no-untyped-def]
    found = events_from_classes(swing_example.targets, swing_example)
    assert set(found) == set(SwingEvent)
    for event, frame in found.items():
        truth = swing_example.label.event(event)
        assert truth is not None
        assert abs(frame - truth.frame_index) <= 2


def test_a_single_flickering_sample_does_not_move_an_event(swing_example) -> None:  # type: ignore[no-untyped-def]
    """The longest run, not the first: per-sample classifiers flicker."""
    clean = events_from_classes(swing_example.targets, swing_example)
    noisy = swing_example.targets.copy()
    noisy[3] = CLASS_INDEX[FrameClass.BACKSWING]
    noisy[5] = CLASS_INDEX[FrameClass.DOWNSWING]

    assert events_from_classes(noisy, swing_example) == clean


def test_an_absent_class_produces_no_event_rather_than_a_guess(swing_example) -> None:  # type: ignore[no-untyped-def]
    """Inventing a top at the midpoint turns a visible failure into a quiet one."""
    without_downswing = swing_example.targets.copy()
    without_downswing[without_downswing == CLASS_INDEX[FrameClass.DOWNSWING]] = CLASS_INDEX[
        FrameClass.BACKSWING
    ]
    found = events_from_classes(without_downswing, swing_example)
    assert SwingEvent.TOP not in found


def test_a_detector_that_finds_nothing_is_counted_as_missing_rather_than_wrong(built) -> None:  # type: ignore[no-untyped-def]
    silent = [
        np.full_like(example.targets, CLASS_INDEX[FrameClass.NONE]) for example in built.examples
    ]
    report = evaluate("silent", silent, built.examples)

    takeaway = report.event_error(SwingEvent.TAKEAWAY)
    assert takeaway is not None
    assert takeaway.scored == 0
    assert takeaway.missed == sum(1 for example in built.examples if example.label.is_swing)
    assert takeaway.mae_ms is None


def test_events_reported_out_of_order_are_warned_about(swing_example, built) -> None:  # type: ignore[no-untyped-def]
    """The rule detector cannot produce this at all, which is a real difference."""
    scrambled = swing_example.targets.copy()
    downswing = scrambled == CLASS_INDEX[FrameClass.DOWNSWING]
    backswing = scrambled == CLASS_INDEX[FrameClass.BACKSWING]
    scrambled[downswing] = CLASS_INDEX[FrameClass.BACKSWING]
    scrambled[backswing] = CLASS_INDEX[FrameClass.DOWNSWING]

    report = evaluate("scrambled", [scrambled], [swing_example])
    assert any("out of order" in note for note in report.warnings)


def test_the_confusion_matrix_is_true_rows_by_predicted_columns(swing_example) -> None:  # type: ignore[no-untyped-def]
    everything_address = np.full_like(swing_example.targets, CLASS_INDEX[FrameClass.ADDRESS])
    report = evaluate("always address", [everything_address], [swing_example])

    matrix = np.array(report.confusion)
    assert matrix.shape == (len(FRAME_CLASSES), len(FRAME_CLASSES))
    predicted_columns = matrix.sum(axis=0)
    assert predicted_columns[CLASS_INDEX[FrameClass.ADDRESS]] == matrix.sum()


def test_per_frame_accuracy_flatters_a_detector_that_macro_f1_does_not(swing_example) -> None:  # type: ignore[no-untyped-def]
    """The reason macro F1 is the headline number."""
    everything_address = np.full_like(swing_example.targets, CLASS_INDEX[FrameClass.ADDRESS])
    report = evaluate("always address", [everything_address], [swing_example])
    assert report.macro_f1 < report.frame_accuracy


def test_the_rule_detector_is_read_through_the_same_grid(swing_example) -> None:  # type: ignore[no-untyped-def]
    classes = rule_classes(swing_example)
    assert classes.shape == swing_example.targets.shape
    assert set(np.unique(classes)) - {CLASS_INDEX[FrameClass.NONE]}


def test_a_clip_the_rules_refused_is_every_sample_none(swing_example) -> None:  # type: ignore[no-untyped-def]
    """A refusal is an answer -- "no swing here" -- not a penalty."""
    refused = replace(swing_example, rule=None)
    assert np.all(rule_classes(refused) == CLASS_INDEX[FrameClass.NONE])


def test_both_detectors_are_scored_on_exactly_the_same_clips(built) -> None:  # type: ignore[no-untyped-def]
    split = split_dataset(built, seed=0)
    model, _ = train(split.train, split.val, config=FAST, seed=0)
    comparison = compare_detectors(model, split.test, model_id="m")

    assert comparison.rule.clips == comparison.model.clips == len(split.test)
    assert comparison.rule.samples == comparison.model.samples
    assert comparison.rule.noise_floor_ms == comparison.model.noise_floor_ms


def test_a_difference_inside_the_noise_floor_is_not_a_win(built) -> None:  # type: ignore[no-untyped-def]
    """A narrow win reported as a win is how a project fools itself."""
    split = split_dataset(built, seed=0)
    model, _ = train(split.train, split.val, config=FAST, seed=0)
    comparison = compare_detectors(model, split.test, model_id="m")

    for row in comparison.rows:
        if row.difference_ms is not None and abs(row.difference_ms) < row.noise_floor_ms:
            assert row.verdict == "indistinguishable"
    assert not comparison.claims_permitted
    assert "may be quoted" in comparison.verdict


def test_a_comparison_needs_a_held_out_set(built) -> None:  # type: ignore[no-untyped-def]
    split = split_dataset(built, seed=0)
    model, _ = train(split.train, split.val, config=FAST, seed=0)
    with pytest.raises(ValueError, match="empty"):
        compare_detectors(model, (), model_id="m")


def test_predictions_are_one_class_per_grid_sample(swing_example, built) -> None:  # type: ignore[no-untyped-def]
    split = split_dataset(built, seed=0)
    model, _ = train(split.train, split.val, config=FAST, seed=0)
    predicted = predict_classes(model, swing_example)

    assert predicted.shape == swing_example.targets.shape
    assert set(np.unique(predicted)) <= set(range(len(FRAME_CLASSES)))
