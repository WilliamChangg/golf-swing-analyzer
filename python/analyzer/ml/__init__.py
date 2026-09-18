"""The learned swing detector, and the apparatus that decides whether to believe it.

Phase 12. Phase 4 already finds the four events with rules that can be read, and
finds them well enough that nothing in this package is needed to make the product
work. What this package adds is the ability to say **how well** — which needs a
labelled set, a split that holds whole golfers out, and a comparison run on the
same clips with the same metric.

The order the modules were built in is the order they appear here, and it is the
order that matters:

    labels      what a human judgement about a frame is, and what it must record
    labeltool   the tool that produces one, built before anything consumed one
    features    what the model is fed, versioned by the content of its definition
    dataset     labels and footage to arrays, with everything that made them kept
    splits      whole players held out, or a refusal explaining why not
    tcn         a small dilated stack, sized against the data that exists
    train       seeded, checkpointed, weighted against a class balance that lies
    evaluate    per-event error, and the gates that stop a number being quoted
    compare     the model against the rules, on one set, with one verdict word
    registry    weights and the card that says what they mean, stored together

Nothing here is reachable over the IPC boundary and nothing is exported to
TypeScript. Training is a developer operation on a corpus that does not ship, and
`scripts/gen_types.py` exports only what a UI draws.
"""

from analyzer.ml.dataset import Dataset, DatasetError, Example, build_dataset
from analyzer.ml.evaluate import evaluate, events_from_classes, predict_classes, rule_classes
from analyzer.ml.features import FEATURE_SPEC, ClipFeatures, FeatureError, clip_features
from analyzer.ml.labels import LabelStoreError, load_label_set, write_label
from analyzer.ml.splits import Split, SplitRefused, plan_split, split_dataset
from analyzer.ml.tcn import SwingTCN
from analyzer.ml.train import TrainingError, train

__all__ = [
    "FEATURE_SPEC",
    "ClipFeatures",
    "Dataset",
    "DatasetError",
    "Example",
    "FeatureError",
    "LabelStoreError",
    "Split",
    "SplitRefused",
    "SwingTCN",
    "TrainingError",
    "build_dataset",
    "clip_features",
    "evaluate",
    "events_from_classes",
    "load_label_set",
    "plan_split",
    "predict_classes",
    "rule_classes",
    "split_dataset",
    "train",
    "write_label",
]
