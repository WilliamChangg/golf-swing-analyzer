"""Training a swing detector, seeded and checkpointed.

Three decisions here are worth more than the loop itself.

## The loss is weighted, because per-frame accuracy is a trap

A golf clip is mostly address and mostly follow-through. The downswing is around
a tenth of a second out of several, so a model that predicts "address" for every
sample scores over 0.5 on per-frame accuracy without having learned anything, and
gradient descent finds that solution first. The class weights are inverse
frequency, computed from the training split only -- computing them over the whole
dataset would carry a fact about the held-out clips into the training run, which
is a small leak of exactly the kind this phase exists to take seriously.

## Samples where the subject was not tracked are not trained on

A grid sample whose landmarks were missing is fed to the model as zeros with the
validity channel off. Asking the model to also predict the right class there
would teach it that a particular pattern of zeros means whatever those clips
happened to be labelled, and that association survives a held-out split because
it is a property of the feature format rather than of a player. The loss is
masked; the model still answers everywhere at inference, and the evaluation
reports what fraction of samples it was scored on.

## Batches are one clip

Clips differ in length, and padding them into a batch makes every normalisation
statistic and every loss average depend on how much padding a clip received. The
corpus this trains on is tens of clips; the whole run is seconds. When that stops
being true the fix is bucketing by length, not padding.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analyzer.contracts.ml import (
    FRAME_CLASSES,
    EpochRecord,
    TCNConfig,
    TrainingReport,
)
from analyzer.ml.dataset import Example
from analyzer.ml.features import FEATURE_SPEC
from analyzer.ml.tcn import SwingTCN

# The validity channel's index in the feature array. Read from the spec rather
# than written down, so a channel added before it cannot silently move it.
_VALID = FEATURE_SPEC.names.index("valid")


class TrainingError(RuntimeError):
    """Training cannot run on what was supplied."""


def seed_everything(seed: int) -> bool:
    """Seed every generator this run touches, and report whether that is enough.

    Returns whether torch could be put into a deterministic mode. It is a return
    value rather than an assumption because the answer depends on the build and
    the device: some kernels have no deterministic implementation, and a run that
    claimed reproducibility it did not have would make a rerun's disagreement
    look like a bug in the data.
    """
    random.seed(seed)
    # The legacy global seed, not a Generator: it is the one a third-party library
    # reaching for `np.random` will actually use, and pinning it is the point.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError):
        return False
    return True


def _tensors(
    example: Example, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One clip as (features, targets, mask), shaped for the network."""
    values = torch.from_numpy(np.ascontiguousarray(example.features.values.T))
    features = values.unsqueeze(0).to(device=device, dtype=torch.float32)
    targets = torch.from_numpy(example.targets).unsqueeze(0).to(device=device)
    mask = torch.from_numpy(example.features.valid).unsqueeze(0).to(device=device)
    return features, targets, mask


def class_weights(examples: Sequence[Example]) -> torch.Tensor:
    """Inverse-frequency weights over the classes actually present.

    A class absent from the training split gets weight zero rather than infinity,
    which is what an inverse frequency would be. Zero is the honest weight: the
    model cannot be trained to predict something it was never shown, and a large
    finite weight would just add noise to the classes it *was* shown.
    """
    counts = np.zeros(len(FRAME_CLASSES), dtype=np.float64)
    for example in examples:
        valid = example.features.valid
        for index in range(len(FRAME_CLASSES)):
            counts[index] += int(np.count_nonzero((example.targets == index) & valid))

    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights.astype(np.float32))


def _masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor | None:
    """Mean loss over the samples the subject was tracked in, or None if there are none."""
    if not bool(mask.any()):
        return None
    per_sample = criterion(logits, targets)
    selected = per_sample[mask]
    if selected.numel() == 0:
        return None
    return selected.mean()


def _macro_f1(confusion: np.ndarray) -> float:
    """Unweighted mean F1 over the classes with any support.

    Classes with no support are excluded rather than scored zero: a set with no
    negative clips in it has no NONE samples, and averaging a zero for a class
    nobody was asked about would report a worse model rather than a smaller test.
    """
    scores: list[float] = []
    for index in range(confusion.shape[0]):
        support = confusion[index].sum()
        if support == 0:
            continue
        true_positive = confusion[index, index]
        predicted = confusion[:, index].sum()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores)) if scores else 0.0


def _validate(
    model: SwingTCN,
    examples: Sequence[Example],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float | None, float | None]:
    """Mean validation loss and macro F1, or (None, None) with nothing to score."""
    if not examples:
        return None, None
    model.eval()
    losses: list[float] = []
    confusion = np.zeros((len(FRAME_CLASSES), len(FRAME_CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for example in examples:
            features, targets, mask = _tensors(example, device)
            logits = model(features)
            loss = _masked_loss(logits, targets, mask, criterion)
            if loss is not None:
                losses.append(float(loss.item()))
            predicted = logits.argmax(dim=1)[mask].cpu().numpy()
            truth = targets[mask].cpu().numpy()
            for true_index, predicted_index in zip(truth, predicted, strict=True):
                confusion[true_index, predicted_index] += 1
    if not losses:
        return None, None
    return float(np.mean(losses)), _macro_f1(confusion)


def train(
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    *,
    config: TCNConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Path | None = None,
    label_digest: str = "",
) -> tuple[SwingTCN, TrainingReport]:
    """Fit a model, keeping the epoch that scored best on the validation split.

    The returned model carries the best epoch's weights, not the last epoch's.
    Returning the final state is the more common mistake and the quieter one: it
    reports a validation number from one set of weights and then evaluates a
    different set.

    With no validation clips the run keeps the final epoch and says so in a
    warning, because "best" has no meaning without something to measure it on.
    """
    if not train_examples:
        raise TrainingError("There is nothing to train on: the training split is empty.")

    resolved = config or TCNConfig()
    deterministic = seed_everything(seed)
    from analyzer.environment.hardware import select_torch_device

    selected_device, device_note = select_torch_device(device)
    torch_device = torch.device(selected_device)

    channels = train_examples[0].features.values.shape[1]
    model = SwingTCN(channels, resolved).to(torch_device)
    weights = class_weights(train_examples).to(torch_device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )

    warnings: list[str] = []
    if device_note:
        warnings.append(device_note)
    if not deterministic:
        warnings.append(
            "torch could not be put into a deterministic mode on this build, so a "
            "rerun with the same seed may differ slightly. The seed is still recorded."
        )
    if not val_examples:
        warnings.append(
            "No validation clips: the final epoch is kept rather than the best one, "
            "and nothing here stopped the run from overfitting."
        )
    spans = [len(example) for example in train_examples]
    if resolved.receptive_field > min(spans):
        warnings.append(
            f"The receptive field is {resolved.receptive_field} samples and the shortest "
            f"training clip is {min(spans)}. Those outputs see padding rather than swing."
        )

    order = list(range(len(train_examples)))
    generator = random.Random(seed)  # noqa: S311 - shuffles the clip order, nothing more
    records: list[EpochRecord] = []
    best_loss: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    since_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, resolved.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        generator.shuffle(order)
        losses: list[float] = []
        for index in order:
            features, targets, mask = _tensors(train_examples[index], torch_device)
            optimiser.zero_grad(set_to_none=True)
            loss = _masked_loss(model(features), targets, mask, criterion)
            if loss is None:
                # Every sample of this clip is untracked; there is nothing to
                # learn from it and a zero-sample mean would be a NaN gradient.
                continue
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))

        if not losses:
            raise TrainingError(
                "No training clip has a single sample in which the subject was tracked. "
                "The landmarks, not the model, are what needs attention."
            )

        val_loss, val_f1 = _validate(model, val_examples, criterion, torch_device)
        records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                val_loss=val_loss,
                val_macro_f1=val_f1,
                seconds=time.perf_counter() - epoch_started,
            )
        )

        reference = val_loss if val_loss is not None else float(np.mean(losses))
        if best_loss is None or reference < best_loss - 1e-6:
            best_loss, best_epoch = reference, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if val_examples and since_improvement >= resolved.patience:
                warnings.append(
                    f"Stopped at epoch {epoch}: {resolved.patience} epochs without "
                    "validation improvement."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": resolved.model_dump(mode="json"),
                "feature_digest": FEATURE_SPEC.digest(),
                "in_channels": channels,
                "seed": seed,
            },
            checkpoint,
        )

    report = TrainingReport(
        seed=seed,
        device=str(torch_device),
        torch_version=torch.__version__,
        deterministic=deterministic,
        parameters=model.parameter_count(),
        receptive_field_samples=model.receptive_field,
        receptive_field_s=model.receptive_field / FEATURE_SPEC.sample_rate_hz,
        epochs=records,
        best_epoch=best_epoch,
        best_val_loss=best_loss if val_examples else None,
        elapsed_s=time.perf_counter() - started,
        feature_digest=FEATURE_SPEC.digest(),
        label_digest=label_digest,
        train_clips=len(train_examples),
        val_clips=len(val_examples),
        class_weights=[float(value) for value in weights.cpu().numpy()],
        warnings=warnings,
    )
    return model, report
