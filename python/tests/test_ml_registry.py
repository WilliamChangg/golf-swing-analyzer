"""The model registry, whose whole purpose is one refusal.

A checkpoint will happily accept features it was never trained on and produce
confident nonsense. These test that it cannot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from analyzer.contracts.ml import MODEL_CARD_SCHEMA_VERSION, ModelCard, TCNConfig
from analyzer.ml.dataset import build_dataset
from analyzer.ml.features import FEATURE_SPEC
from analyzer.ml.registry import (
    CARD_NAME,
    WEIGHTS_NAME,
    RegistryError,
    list_models,
    load_model,
    model_id,
    save_model,
)
from analyzer.ml.splits import split_dataset
from analyzer.ml.train import train
from tests.synthetic_labels import make_corpus

FAST = TCNConfig(hidden_channels=8, layers=3, epochs=1, dropout=0.0)


@pytest.fixture(scope="module")
def trained():  # type: ignore[no-untyped-def]
    corpus = make_corpus(players=4, sessions=1, swings=1, seed=17)
    built = build_dataset(corpus.labels, corpus.provider())
    split = split_dataset(built, seed=0)
    model, report = train(split.train, split.val, config=FAST, seed=0)
    card = ModelCard(
        model_id="20260917T000000Z-abcdef123456",
        created_at=datetime(2026, 9, 17, tzinfo=UTC).isoformat(),
        architecture="dilated TCN, per-sample classification",
        config=FAST,
        feature_spec=built.spec,
        feature_digest=built.summary.feature_digest,
        label_digest=built.summary.label_digest,
        dataset=built.summary,
        split=split.report,
        training=report,
    )
    return model, card


def test_the_id_is_stable_for_one_recipe_and_changes_with_any_part_of_it() -> None:
    base = {"label_digest": "L", "feature_digest": "F", "seed": 0, "config": FAST}
    first = model_id(**base)  # type: ignore[arg-type]
    assert first.split("-")[1] == model_id(**base).split("-")[1]  # type: ignore[arg-type]
    assert model_id(**{**base, "seed": 1}).split("-")[1] != first.split("-")[1]  # type: ignore[arg-type]
    assert model_id(**{**base, "label_digest": "M"}).split("-")[1] != first.split("-")[1]  # type: ignore[arg-type]


def test_saving_writes_the_weights_and_the_card(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model, card = trained
    directory = save_model(model, card, tmp_path)
    assert (directory / WEIGHTS_NAME).exists()
    assert (directory / CARD_NAME).exists()


def test_a_saved_model_loads_back_with_the_same_weights(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model, card = trained
    save_model(model, card, tmp_path)
    loaded, loaded_card = load_model(card.model_id, tmp_path)

    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    assert loaded_card.label_digest == card.label_digest
    assert loaded_card.split.leakage is not None
    assert loaded_card.split.leakage.passed


def test_a_model_built_for_different_features_is_refused(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The reason the registry exists: nothing about running it would look wrong."""
    model, card = trained
    stale = card.model_copy(update={"feature_digest": "0" * 64})
    save_model(model, stale, tmp_path)

    with pytest.raises(RegistryError, match="channels that no longer exist"):
        load_model(stale.model_id, tmp_path)


def test_an_unknown_model_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="analyzer train"):
        load_model("nothing", tmp_path)


def test_a_card_from_a_future_schema_is_refused(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model, card = trained
    directory = save_model(model, card, tmp_path)
    path = directory / CARD_NAME
    raw = json.loads(path.read_text())
    raw["schema_version"] = MODEL_CARD_SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw))

    with pytest.raises(RegistryError, match="schema version"):
        load_model(card.model_id, tmp_path)


def test_an_interrupted_save_is_skipped_rather_than_listed(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Card last, so a directory without one is visibly incomplete."""
    model, card = trained
    directory = save_model(model, card, tmp_path)
    (directory / CARD_NAME).unlink()
    assert list_models(tmp_path) == []


def test_the_registry_lists_newest_first(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model, card = trained
    save_model(model, card.model_copy(update={"model_id": "20260101T000000Z-aaa"}), tmp_path)
    save_model(model, card.model_copy(update={"model_id": "20260601T000000Z-bbb"}), tmp_path)

    assert [entry.model_id for entry in list_models(tmp_path)] == [
        "20260601T000000Z-bbb",
        "20260101T000000Z-aaa",
    ]


def test_an_absent_registry_lists_nothing_rather_than_failing(tmp_path: Path) -> None:
    assert list_models(tmp_path / "nothing") == []


def test_the_card_records_the_feature_spec_itself_not_only_its_digest(trained) -> None:  # type: ignore[no-untyped-def]
    """A digest says two things differ; the spec says how."""
    _, card = trained
    assert card.feature_spec.names == FEATURE_SPEC.names
    assert card.feature_spec.digest() == card.feature_digest
