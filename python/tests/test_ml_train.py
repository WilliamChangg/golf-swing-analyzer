"""The network and the training loop.

Fast on purpose: a few epochs over a handful of short clips. Nothing here is
trying to establish that the model is good -- that is what a held-out set is for,
and no honest one exists. These test that a run is repeatable, that the loss is
not quietly optimising the wrong thing, and that a checkpoint means what it says.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from analyzer.contracts.ml import FRAME_CLASSES, FrameClass, TCNConfig
from analyzer.ml.dataset import CLASS_INDEX, build_dataset
from analyzer.ml.splits import split_dataset
from analyzer.ml.tcn import SwingTCN
from analyzer.ml.train import TrainingError, class_weights, train
from tests.synthetic_labels import make_corpus

FAST = TCNConfig(hidden_channels=8, layers=3, epochs=2, dropout=0.0)


@pytest.fixture(scope="module")
def split():  # type: ignore[no-untyped-def]
    corpus = make_corpus(players=4, sessions=1, swings=2, seed=11)
    return split_dataset(build_dataset(corpus.labels, corpus.provider()), seed=0)


def test_the_network_preserves_length_so_a_sample_means_its_own_instant() -> None:
    """An unnoticed offset reads downstream as a consistent timing bias."""
    model = SwingTCN(10, FAST)
    out = model(torch.zeros(1, 10, 97))
    assert out.shape == (1, len(FRAME_CLASSES), 97)


def test_a_causal_stack_also_preserves_length() -> None:
    model = SwingTCN(10, FAST.model_copy(update={"causal": True}))
    assert model(torch.zeros(1, 10, 55)).shape[2] == 55


def test_an_even_kernel_is_refused_on_a_non_causal_stack() -> None:
    with pytest.raises(ValueError, match="odd kernel"):
        SwingTCN(10, FAST.model_copy(update={"kernel_size": 4}))


def test_the_receptive_field_is_reported_in_seconds() -> None:
    """Samples are meaningless without the rate; a backswing is about 0.8 s."""
    config = TCNConfig()
    assert config.receptive_field == 127
    assert SwingTCN(10, config).receptive_field == 127
    # Six layers at 60 Hz is 2.1 s, which contains a swing; five would be 1.05 s,
    # which contains a backswing and not the swing around it.
    assert config.receptive_field / 60.0 == pytest.approx(2.117, abs=0.01)


def test_class_weights_are_inverse_frequency_and_zero_for_an_absent_class(split) -> None:  # type: ignore[no-untyped-def]
    """An unweighted loss learns to call every frame address, which scores well."""
    weights = class_weights(split.train).numpy()
    assert weights.shape == (len(FRAME_CLASSES),)
    assert np.all(weights >= 0)

    counts = np.zeros(len(FRAME_CLASSES))
    for example in split.train:
        for index in range(len(FRAME_CLASSES)):
            counts[index] += int(
                np.count_nonzero((example.targets == index) & example.features.valid)
            )
    present = counts > 0
    # Rarer classes carry more weight; absent ones carry none.
    order = np.argsort(counts[present])
    assert np.all(np.diff(weights[present][order]) <= 1e-9)
    assert np.all(weights[~present] == 0.0)


def test_a_run_with_the_same_seed_produces_the_same_weights(split) -> None:  # type: ignore[no-untyped-def]
    first, report = train(split.train, split.val, config=FAST, seed=3)
    second, _ = train(split.train, split.val, config=FAST, seed=3)

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    assert report.seed == 3


def test_a_different_seed_produces_different_weights(split) -> None:  # type: ignore[no-untyped-def]
    first, _ = train(split.train, split.val, config=FAST, seed=3)
    other, _ = train(split.train, split.val, config=FAST, seed=4)
    assert not all(
        torch.equal(value, other.state_dict()[key]) for key, value in first.state_dict().items()
    )


def test_the_report_records_what_it_would_take_to_repeat_the_run(split) -> None:  # type: ignore[no-untyped-def]
    _, report = train(split.train, split.val, config=FAST, seed=0, label_digest="abc")
    assert report.torch_version == torch.__version__
    assert report.device == "cpu"
    assert report.parameters > 0
    assert report.label_digest == "abc"
    assert report.feature_digest
    assert len(report.epochs) == FAST.epochs
    assert report.train_clips == len(split.train)
    assert report.receptive_field_s == pytest.approx(report.receptive_field_samples / 60.0)


def test_training_keeps_the_best_epoch_rather_than_the_last(split) -> None:  # type: ignore[no-untyped-def]
    _, report = train(split.train, split.val, config=FAST.model_copy(update={"epochs": 4}), seed=0)
    losses = [record.val_loss for record in report.epochs]
    assert report.best_val_loss == pytest.approx(
        min(value for value in losses if value is not None)
    )
    assert report.best_epoch is not None


def test_training_without_validation_clips_says_what_that_cost(split) -> None:  # type: ignore[no-untyped-def]
    _, report = train(split.train, (), config=FAST, seed=0)
    assert any("No validation clips" in note for note in report.warnings)
    assert report.best_val_loss is None


def test_an_empty_training_split_is_an_error(split) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TrainingError, match="nothing to train on"):
        train((), split.val, config=FAST, seed=0)


def test_samples_where_the_subject_was_not_tracked_are_not_trained_on(split) -> None:  # type: ignore[no-untyped-def]
    """Otherwise the model learns that a pattern of zeros means a class."""
    blinded = [
        replace(
            example,
            features=replace(example.features, valid=np.zeros_like(example.features.valid)),
        )
        for example in split.train
    ]

    with pytest.raises(TrainingError, match="a single sample in which the subject was tracked"):
        train(blinded, (), config=FAST, seed=0)


def test_a_short_clip_against_a_long_receptive_field_is_reported(split) -> None:  # type: ignore[no-untyped-def]
    """Those outputs see padding, and a poor score should be readable as that."""
    _, report = train(split.train, split.val, config=FAST.model_copy(update={"layers": 9}), seed=0)
    assert any("receptive field" in note for note in report.warnings)


def test_a_checkpoint_carries_what_is_needed_to_rebuild_the_model(split, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "nested" / "model.pt"
    model, _ = train(split.train, split.val, config=FAST, seed=0, checkpoint=target)

    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["in_channels"] == 10
    assert payload["config"]["layers"] == FAST.layers
    assert payload["feature_digest"]

    rebuilt = SwingTCN(payload["in_channels"], TCNConfig.model_validate(payload["config"]))
    rebuilt.load_state_dict(payload["state_dict"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key])


def test_the_model_can_learn_the_classes_it_was_shown(split) -> None:  # type: ignore[no-untyped-def]
    """Not an accuracy claim. Training that never fits anything is a broken loop."""
    model, report = train(
        split.train, split.val, config=FAST.model_copy(update={"epochs": 25}), seed=0
    )
    assert report.epochs[-1].train_loss < report.epochs[0].train_loss

    example = split.train[0]
    values = np.ascontiguousarray(example.features.values.T)
    with torch.no_grad():
        predicted = model(torch.from_numpy(values).unsqueeze(0)).argmax(dim=1)[0].numpy()
    assert set(np.unique(predicted)) - {CLASS_INDEX[FrameClass.NONE]}


def test_forced_cpu_is_used_by_training_and_evaluation(split, monkeypatch) -> None:
    from analyzer.ml.evaluate import predict_classes

    monkeypatch.setenv("GSA_FORCE_CPU", "true")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model, report = train(split.train, split.val, config=FAST, device="cuda")
    assert report.device == "cpu"
    assert any("CPU forced" in note for note in report.warnings)
    prediction = predict_classes(model, split.test[0], device=report.device)
    assert len(prediction) == len(split.test[0].features)
