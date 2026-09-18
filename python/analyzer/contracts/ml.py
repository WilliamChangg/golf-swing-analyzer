"""Typed contracts for the learned swing detector.

Phase 4 finds the four events with rules that can be read and argued with. This
layer trains a model to do the same job, for one reason: nobody can say how good
the rules are without something to compare them against on data neither of them
has seen. The contracts here exist mostly to stop that comparison being made
dishonestly, and the three models that do most of that work are `FeatureSpec`,
`SplitReport` and `EvaluationReport`.

**`FeatureSpec` is versioned by its own content.** A model is a function of the
numbers it was fed, and those numbers are a function of a feature definition
that will change. Storing a version integer alone leaves it possible to change
what a feature means without changing the integer, so the spec hashes itself:
channel names, sample rate, normalisation policy and window. A checkpoint records
that digest and refuses to run against features that do not match it.

**`SplitReport` is a measurement of the split, not a description of it.** The
overlap counts are recomputed from the assignment that was actually produced
rather than asserted by the code that produced it, so a future change to the
splitter cannot make the leak check pass by construction.

**`EvaluationReport` may refuse to permit a claim.** `claims_permitted` is false
whenever the set it scored was synthetic, too small, or drawn from too few
players -- the project's standing rule that no labelled dataset means no accuracy
claim, expressed where it cannot be forgotten. It is also false when the model's
error is smaller than the labels' own uncertainty, which is the failure that
looks like success: a model cannot be shown to be more accurate than the numbers
it is being scored against.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.labels import LabelProvenance
from analyzer.contracts.phases import SwingEvent

# Bump when the meaning of a stored artifact changes.
FEATURE_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
MODEL_CARD_SCHEMA_VERSION = 1

# The smallest set that may support an accuracy claim, in distinct players. Not
# a statistical result: it is the smallest number that leaves a held-out set with
# more than one player in it after train and validation have taken theirs, which
# is the weakest form of "the number is not about one person". Real work needs
# far more, and `EvaluationReport` says so in its own warnings.
MIN_PLAYERS_FOR_CLAIM = 8
MIN_TEST_CLIPS_FOR_CLAIM = 20


class FrameClass(StrEnum):
    """What one frame is, as a classifier sees it.

    Five, not four. `SwingPhase` has four values and tiles only the part of a
    clip that is a swing: address runs from frame zero and the follow-through
    ends at the finish, so every frame after the finish -- and every frame of a
    clip with no swing in it -- belongs to no phase at all. `SwingPhases.phase_at`
    returns None for those, and a classifier needs somewhere to put them.

    Collapsing NONE into ADDRESS was the alternative and it is wrong in a way that
    matters: a player walking away after the finish would be trained as a player
    standing over the ball, and a clip of somebody warming up would be trained as
    a swing that never starts.
    """

    NONE = "none"
    ADDRESS = "address"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    FOLLOW_THROUGH = "follow_through"


# Fixed order. The index of a class in this tuple is its integer label in every
# array, every checkpoint and every confusion matrix, so reordering it silently
# invalidates every model ever trained. It is a constant for that reason.
FRAME_CLASSES: tuple[FrameClass, ...] = (
    FrameClass.NONE,
    FrameClass.ADDRESS,
    FrameClass.BACKSWING,
    FrameClass.DOWNSWING,
    FrameClass.FOLLOW_THROUGH,
)


class FeatureSpec(BaseModel, frozen=True):
    """What the model is fed, precisely enough to hash.

    Every channel is normalised by the subject's own torso length rather than by
    anything measured over the whole clip. That is not a style choice. A feature
    divided by the clip's peak hand speed carries a fact about the end of the
    clip into every frame of the beginning, which is fine for an offline
    classifier and fatal the moment anyone runs it while recording -- and worse,
    it is a difference that never shows up in an offline score. Torso length is a
    property of the body, so a channel built on it means the same thing at frame
    zero as at the finish.
    """

    version: int = FEATURE_SCHEMA_VERSION
    names: tuple[str, ...] = Field(description="Channel names, in array column order.")
    sample_rate_hz: float = Field(
        gt=0.0,
        description=(
            "The common grid every clip is resampled onto, in **real** seconds. A "
            "dilation stack spans a fixed number of samples, so without this a "
            "receptive field would mean 2.1 s on one clip and 0.26 s on another."
        ),
    )
    normalisation: str = Field(
        description="How the channels are made comparable between subjects, in words."
    )
    interpolation: str = Field(description="How a clip's own timestamps are mapped onto the grid.")

    @property
    def channels(self) -> int:
        return len(self.names)

    @property
    def resample_floor_s(self) -> float:
        """The best an event located on this grid can be placed, in real seconds.

        Half a sample: a prediction on the grid is mapped back to the nearest
        frame of the clip, and nothing about that mapping can be more precise
        than the spacing it was made on. Reported on every evaluation so no
        reader credits a model with a resolution the resampling already spent.
        """
        return 0.5 / self.sample_rate_hz

    def digest(self) -> str:
        """Content hash of the whole spec. The thing a checkpoint stores."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class DatasetSummary(BaseModel):
    """What a built dataset is made of, and what it is not big enough for."""

    schema_version: int = DATASET_SCHEMA_VERSION
    provenance: LabelProvenance
    feature_version: int
    feature_digest: str
    sample_rate_hz: float
    label_digest: str = Field(description="Digest of the label set this was built from.")

    clips: int
    swings: int
    non_swings: int
    players: int
    sessions: int
    samples: int = Field(description="Total frames on the canonical grid, across all clips.")

    class_samples: dict[FrameClass, int]
    class_fraction: dict[FrameClass, float]

    median_label_uncertainty_ms: float | None = Field(
        default=None,
        description=(
            "Median over every marked event of the labeller's own bracket, in real "
            "milliseconds. None when nothing carries one. This is the floor any "
            "score against these labels is measured against."
        ),
    )
    resample_floor_ms: float = Field(
        description="Half a sample of the canonical grid, in milliseconds. See `FeatureSpec`."
    )
    dropped: list[str] = Field(
        default_factory=list,
        description="Clips that were labelled but could not be turned into features, with why.",
    )
    warnings: list[str] = Field(default_factory=list)

    @property
    def imbalance_ratio(self) -> float:
        """Largest class over smallest non-empty class.

        Per-frame accuracy on a golf clip is close to meaningless, and this is
        the number that says by how much: address and the follow-through occupy
        most of every clip while the downswing is a tenth of a second.
        """
        counts = [count for count in self.class_samples.values() if count > 0]
        if not counts:
            return float("nan")
        return max(counts) / min(counts)


class TCNConfig(BaseModel, extra="forbid"):
    """The shape of the network, and the knobs a training run may turn.

    Small on purpose. Sixteen hidden channels over six dilated layers is a few
    tens of thousands of parameters, which is chosen against the size of the
    labelled set that exists rather than against the difficulty of the task. A
    network with the capacity to memorise forty clips will memorise forty clips,
    and on a set this size that is indistinguishable from learning until it meets
    a player it has not seen.
    """

    hidden_channels: int = Field(default=32, gt=0)
    layers: int = Field(
        default=6,
        gt=0,
        description=(
            "Dilations double per layer, so the receptive field is roughly 2^layers "
            "samples. Six at 60 Hz is 2.1 s, which covers a swing; five would be "
            "1.05 s, which covers a backswing and not the swing containing it."
        ),
    )
    kernel_size: int = Field(default=3, gt=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    causal: bool = Field(
        default=False,
        description=(
            "False: each output sample sees equally far forward and back. This is "
            "offline analysis of a recorded clip, where the frames after an instant "
            "are as available as the frames before it -- and the top of a backswing "
            "is defined by what happens next. A live detector would need this true, "
            "and would be a different model with a different score."
        ),
    )
    learning_rate: float = Field(default=3e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    epochs: int = Field(default=60, gt=0)
    patience: int = Field(
        default=15,
        gt=0,
        description="Epochs without validation improvement before training stops.",
    )

    @property
    def receptive_field(self) -> int:
        """Samples one output can see, counting both directions."""
        span = sum((self.kernel_size - 1) * (2**layer) for layer in range(self.layers))
        return 1 + span


class SplitRole(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SplitGroup(BaseModel):
    """One group's assignment. The group is a player; nothing smaller is split."""

    role: SplitRole
    player_id: str
    sessions: int
    clips: int
    swings: int


class LeakageCheck(BaseModel):
    """Measured overlap between the three roles, after assignment.

    Computed from the produced assignment rather than from the splitter's
    intentions, which is the only version of this check worth having: an
    assertion inside the splitter tests that the splitter agrees with itself.

    `clip_overlap` looks redundant beside the other two and is not. It is the one
    that catches the same clip having been labelled twice under two player ids --
    a copy of a file in two directories, which is how duplicate footage actually
    arrives.
    """

    player_overlap: int = Field(ge=0, description="Players appearing in more than one role.")
    session_overlap: int = Field(ge=0, description="Sessions appearing in more than one role.")
    clip_overlap: int = Field(
        ge=0, description="Clip content keys appearing in more than one role."
    )

    @property
    def passed(self) -> bool:
        return self.player_overlap == 0 and self.session_overlap == 0 and self.clip_overlap == 0


class SplitReport(BaseModel):
    """How a labelled set was divided, or why it could not be.

    `refused` is the common outcome on a real project at this stage and is a
    result rather than an error. A set of four clips of one golfer cannot be
    divided into training and held-out players, and a splitter that returned
    something anyway would be inventing the only thing that makes a held-out
    number mean anything.
    """

    seed: int
    grouped_by: str = Field(
        default="player",
        description=(
            "What was held together. Player rather than session: a session belongs "
            "to one player, so grouping by player holds sessions together too, "
            "while the reverse leaves the same golfer on both sides of the split."
        ),
    )
    refused: bool = False
    refusal: str | None = Field(
        default=None, description="Why no split was produced. None when one was."
    )
    groups: list[SplitGroup] = Field(default_factory=list)
    assignment: dict[str, SplitRole] = Field(
        default_factory=dict,
        description=(
            "Clip key to role, which is the split itself rather than a summary of "
            "it. Stored so a model card records exactly which clips were held out, "
            "and so a rerun can be checked against the run it claims to reproduce "
            "instead of against a set of counts that several assignments share."
        ),
    )
    clips: dict[SplitRole, int] = Field(default_factory=dict)
    players: dict[SplitRole, int] = Field(default_factory=dict)
    leakage: LeakageCheck | None = None
    label_digest: str = ""
    warnings: list[str] = Field(default_factory=list)


class EpochRecord(BaseModel):
    """One pass over the training set."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_macro_f1: float | None = None
    seconds: float


class TrainingReport(BaseModel):
    """What a training run did, in enough detail to repeat it.

    `deterministic` is reported rather than promised. The seeding here fixes
    Python, NumPy and torch, and torch's own determinism depends on the kernels a
    given build and device select; on a backend that cannot be made deterministic
    this says so instead of claiming a reproducibility the run does not have.
    """

    seed: int
    device: str
    torch_version: str
    deterministic: bool
    parameters: int = Field(description="Trainable parameters.")
    receptive_field_samples: int
    receptive_field_s: float = Field(
        description=(
            "How much of a clip the model can see at once, in real seconds. A "
            "model whose receptive field is shorter than a backswing cannot use "
            "the structure the rules key on, however well it scores per frame."
        )
    )
    epochs: list[EpochRecord] = Field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    elapsed_s: float = 0.0
    feature_digest: str = ""
    label_digest: str = ""
    train_clips: int = 0
    val_clips: int = 0
    class_weights: list[float] = Field(
        default_factory=list,
        description=(
            "Per-class loss weights actually used. An unweighted loss on this "
            "problem learns to call every frame address, which scores well."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


class ClassMetrics(BaseModel):
    """Per-frame scores for one class. Never reported without support."""

    frame_class: FrameClass
    support: int = Field(description="Frames of this class in the scored set.")
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class EventError(BaseModel):
    """How far a detector put one event from where the label put it.

    Frames and milliseconds are both reported because neither alone is
    comparable: frames are what a labeller marked and what a reader can check by
    stepping the clip, and milliseconds are the only unit in which a 30 fps clip
    and a 240 fps clip can be averaged together.

    `within_label_uncertainty` is the honest version of an accuracy figure for
    this task. An error of one frame against a label the labeller marked "give or
    take three" is not an error at all, and a mean absolute error reports it as
    one.
    """

    event: SwingEvent
    scored: int = Field(description="Clips where both the label and the detector had this event.")
    missed: int = Field(description="Clips where the label had it and the detector did not.")
    spurious: int = Field(description="Clips where the detector had it and the label did not.")
    median_frames: float | None = None
    mae_frames: float | None = None
    mae_ms: float | None = None
    p90_ms: float | None = None
    bias_ms: float | None = Field(
        default=None,
        description="Signed mean. Positive means the detector is late. A bias is correctable; scatter is not.",
    )
    within_label_uncertainty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of scored clips landing inside the labeller's own bracket.",
    )


class EvaluationReport(BaseModel):
    """What a detector scored on a held-out set, and whether that may be quoted.

    The per-frame numbers and the per-event numbers answer different questions
    and the second is the one that matters. A model can score 0.97 per frame by
    getting address and the follow-through right -- they are most of every clip --
    while placing impact ten frames out.
    """

    detector: str = Field(description="What was scored: a model id, or the rule-based detector.")
    provenance: LabelProvenance
    clips: int
    players: int
    samples: int

    frame_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(
        ge=0.0,
        le=1.0,
        description="Unweighted mean of per-class F1, so the downswing counts as much as address.",
    )
    classes: list[ClassMetrics] = Field(default_factory=list)
    confusion: list[list[int]] = Field(
        default_factory=list,
        description="Rows are true classes, columns predicted, both in FRAME_CLASSES order.",
    )
    events: list[EventError] = Field(default_factory=list)

    noise_floor_ms: float = Field(
        description=(
            "The smallest error that could be distinguished here: the labels' own "
            "median uncertainty and the resampling floor added together."
        )
    )
    claims_permitted: bool = Field(
        description="Whether any accuracy number here may be published as a result."
    )
    claim_refusal: str | None = Field(
        default=None, description="Why not. None when a claim is permitted."
    )
    label_digest: str = ""
    feature_digest: str = ""
    warnings: list[str] = Field(default_factory=list)

    def event_error(self, which: SwingEvent) -> EventError | None:
        return next((entry for entry in self.events if entry.event is which), None)


class ComparisonRow(BaseModel):
    """One event, scored two ways on the same clips.

    `verdict` is a word rather than a winner because the difference is usually
    smaller than what the labels can resolve. "indistinguishable" is the correct
    result of a fair comparison on a small set, and reporting it as a narrow win
    for whichever side came out ahead is how a project talks itself into
    believing a model helped.
    """

    event: SwingEvent
    scored: int
    rule_mae_ms: float | None = None
    model_mae_ms: float | None = None
    difference_ms: float | None = Field(
        default=None, description="rule - model. Positive means the model is closer."
    )
    noise_floor_ms: float = 0.0
    verdict: str = Field(description="One of: model, rule, indistinguishable, not scored.")


class DetectorComparison(BaseModel):
    """The rule-based detector and a trained model on one held-out set.

    Both are scored on the same clips with the same metric and the same labels.
    That is the whole point of the object: two numbers produced by two different
    evaluations on two different subsets are not a comparison, however carefully
    each was computed.
    """

    provenance: LabelProvenance
    clips: int
    players: int
    model_id: str
    rule: EvaluationReport
    model: EvaluationReport
    rows: list[ComparisonRow] = Field(default_factory=list)
    verdict: str = Field(description="The overall reading, in one sentence.")
    claims_permitted: bool
    warnings: list[str] = Field(default_factory=list)


class ModelCard(BaseModel):
    """The registry entry for one trained model.

    A checkpoint on its own is a file of numbers that will happily produce
    predictions from features it was never trained on. Everything that decides
    whether those predictions mean anything -- which features, which labels,
    which split, what it scored, whether that score may be quoted -- lives here,
    and `analyzer.ml.registry` refuses to load a checkpoint whose features do not
    match the digest recorded in its card.
    """

    schema_version: int = MODEL_CARD_SCHEMA_VERSION
    model_id: str = Field(description="Stable id: the timestamp and a digest of what made it.")
    created_at: str
    architecture: str
    config: TCNConfig
    feature_spec: FeatureSpec
    feature_digest: str
    label_digest: str
    dataset: DatasetSummary
    split: SplitReport
    training: TrainingReport
    evaluation: EvaluationReport | None = Field(
        default=None,
        description="None until the model has been scored on the held-out set.",
    )
    notes: str = ""
