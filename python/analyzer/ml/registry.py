"""Where trained models live, and what has to travel with them.

A checkpoint on its own is a file of numbers that will happily accept features it
was never trained on and produce confident predictions from them. Nothing about
that failure is visible: the shapes match, the softmax sums to one, and the
output is wrong in a way that looks like a model having a bad day.

So a model is stored as two files that are written and read together -- the
weights, and a `ModelCard` naming the feature spec, the labels, the split, the
training run and whatever it scored. `load_model` compares the card's feature
digest against what this build computes and refuses on a mismatch. That refusal
is the entire reason the registry exists; everything else here is filing.

Ids are `<timestamp>-<digest>`, where the digest covers the labels, the features,
the seed and the configuration. Two runs over the same data with the same settings
produce the same digest and different timestamps, so the directory sorts by date
and a rerun is visibly a rerun rather than a new experiment.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import ValidationError

from analyzer.contracts.ml import MODEL_CARD_SCHEMA_VERSION, ModelCard, TCNConfig
from analyzer.ml.features import FEATURE_SPEC
from analyzer.ml.tcn import SwingTCN
from analyzer.paths import ml_registry_dir

CARD_NAME = "card.json"
WEIGHTS_NAME = "model.pt"


class RegistryError(RuntimeError):
    """A model could not be stored or loaded, and using it anyway would be worse."""


def registry_root(root: Path | None = None) -> Path:
    return root if root is not None else ml_registry_dir() / "models"


def model_id(*, label_digest: str, feature_digest: str, seed: int, config: TCNConfig) -> str:
    """A stable id for one training recipe, prefixed with when it ran."""
    payload = json.dumps(
        {
            "labels": label_digest,
            "features": feature_digest,
            "seed": seed,
            "config": config.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{digest}"


def save_model(model: SwingTCN, card: ModelCard, root: Path | None = None) -> Path:
    """Write the weights and the card together, card last.

    Card last so an interrupted save leaves a directory without one, and
    `list_models` skips it. The opposite order leaves a card describing weights
    that are not there, which reads as a model until something tries to load it.
    """
    directory = registry_root(root) / card.model_id
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": card.config.model_dump(mode="json"),
            "in_channels": model.stem.in_channels,
            "feature_digest": card.feature_digest,
        },
        directory / WEIGHTS_NAME,
    )
    (directory / CARD_NAME).write_text(
        json.dumps(card.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def read_card(path: Path) -> ModelCard:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{path} could not be read: {exc}") from exc
    if raw.get("schema_version") != MODEL_CARD_SCHEMA_VERSION:
        raise RegistryError(
            f"{path.parent.name} has a model card of schema version "
            f"{raw.get('schema_version')!r}; this build reads {MODEL_CARD_SCHEMA_VERSION}."
        )
    try:
        return ModelCard.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(f"{path.parent.name} has an unreadable model card: {exc}") from exc


def list_models(root: Path | None = None) -> list[ModelCard]:
    """Every complete model in the registry, newest first."""
    directory = registry_root(root)
    if not directory.exists():
        return []
    cards: list[ModelCard] = []
    for entry in sorted(directory.iterdir(), reverse=True):
        card = entry / CARD_NAME
        if entry.is_dir() and card.exists() and (entry / WEIGHTS_NAME).exists():
            cards.append(read_card(card))
    return cards


def load_model(identifier: str, root: Path | None = None) -> tuple[SwingTCN, ModelCard]:
    """Load a stored model, refusing one built for different features.

    The refusal is the point of this function. A feature spec that has changed --
    a channel added, the sample rate moved, the normalisation reconsidered -- makes
    every weight in the checkpoint describe an input that no longer exists, and
    nothing about running it anyway would look wrong.
    """
    directory = registry_root(root) / identifier
    if not directory.exists():
        raise RegistryError(
            f"No model '{identifier}' in {registry_root(root)}. "
            "Run `analyzer train` first, or `analyzer models` to see what is there."
        )

    card = read_card(directory / CARD_NAME)
    current = FEATURE_SPEC.digest()
    if card.feature_digest != current:
        raise RegistryError(
            f"Model '{identifier}' was trained on feature spec {card.feature_digest[:12]} "
            f"and this build computes {current[:12]}. Its weights describe channels that "
            "no longer exist, so it is refused rather than run: retrain it, or check out "
            "the revision that produced it."
        )

    payload = torch.load(directory / WEIGHTS_NAME, map_location="cpu", weights_only=True)
    model = SwingTCN(int(payload["in_channels"]), TCNConfig.model_validate(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, card
