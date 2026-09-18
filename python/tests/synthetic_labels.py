"""A corpus of labelled swings whose events are known by construction.

## What this is for, and what it is not for

Phase 12's machinery -- features, splits, training, evaluation, comparison -- has
to be exercised end to end before anyone can trust a number that comes out of it,
and no labelled footage exists for this project. So the corpus is generated: a
number of players, each with sessions, each with swings, each drawn by
`tests/synthetic.py` from event times that are **inputs** rather than readings.

Every report derived from it carries `LabelProvenance.SYNTHETIC`, and
`analyzer/ml/evaluate.py` refuses to permit an accuracy claim from one. That
refusal is not caution. The events here are the parameters of the curve the hands
follow, so a detector scored against them is being asked whether it can recover
the inputs to a generator it was effectively told about. Real footage supplies
none of that: motion blur through the downswing, a wrist behind a torso, a
labeller who can see the top to within three frames and says so.

## What it *can* measure honestly

Two things, both of which are properties of the machinery rather than of golf.

**That leakage is worth something.** Swings inside one session here are near
duplicates of each other -- the same player's shape, jittered -- exactly as they
are in real footage, because that is the one property of a real session this can
reproduce faithfully. Training under a player-grouped split and under a
clip-random one gives the difference between what a leaky evaluation reports and
what the model can do, and neither number needs to be a statement about golf for
the *difference* to be one about method.

**That the plumbing is right.** A feature spec that resamples correctly, a split
that holds groups apart, a training loop that converges, an evaluation that maps
grid samples back to frames. These either work or they do not, and a fixture can
tell.

## How the players differ

Tempo, arc size, the angle at the top, body proportions and where the camera was
put. Within a session the swings differ only by jitter in the landmarks and a
small tempo wobble, which is what makes them near-duplicates. Frame rates differ
between sessions, because a set that is all 60 fps cannot catch a feature layer
that quietly assumes one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.labels import ClipLabel, EventLabel, LabelProvenance, LabelSet
from analyzer.contracts.phases import SwingEvent
from analyzer.contracts.pose import Landmark, PoseSequence
from tests.synthetic import GROUND_Y, SwingShape, hand_path, pose_sequence

# Frame rates a session might have been shot at. Deliberately a mix: 30 is a
# phone, 60 a better phone, 120 the slow-motion mode people actually use.
SESSION_RATES = (30.0, 60.0, 120.0)

# How long after the finish the clip keeps rolling. Frames beyond the finish
# belong to no phase, so without them the NONE class has no positive examples in
# any swing clip and the classifier never learns that a clip can end.
TAIL_S = 0.5


@dataclass(frozen=True)
class Player:
    """One synthetic golfer: a shape, a body, and a name."""

    player_id: str
    shape: SwingShape
    torso_length: float
    centre_x: float
    shoulder_y: float

    def body(self) -> dict[int, tuple[float, float]]:
        """The static landmarks for this player, in image coordinates.

        Proportions vary between players and the ground stays where it is, which
        is what a camera pointed at a driving range sees: different people, same
        turf. Every feature this project computes is divided by torso length, so
        two players of different sizes are the cheapest check that the division
        is doing something.
        """
        hip_y = self.shoulder_y + self.torso_length
        return {
            int(Landmark.LEFT_SHOULDER): (self.centre_x - 0.05, self.shoulder_y),
            int(Landmark.RIGHT_SHOULDER): (self.centre_x + 0.05, self.shoulder_y),
            int(Landmark.LEFT_HIP): (self.centre_x - 0.04, hip_y),
            int(Landmark.RIGHT_HIP): (self.centre_x + 0.04, hip_y),
            int(Landmark.LEFT_ANKLE): (self.centre_x - 0.04, GROUND_Y),
            int(Landmark.RIGHT_ANKLE): (self.centre_x + 0.04, GROUND_Y),
        }


def make_players(count: int, *, seed: int = 0) -> list[Player]:
    """`count` golfers who differ in tempo, arc and build."""
    generator = random.Random(seed)  # noqa: S311 - a fixture, not a secret
    players: list[Player] = []
    for index in range(count):
        takeaway = generator.uniform(0.35, 0.70)
        backswing = generator.uniform(0.60, 1.00)
        downswing = generator.uniform(0.22, 0.38)
        players.append(
            Player(
                player_id=f"p{index:02d}",
                shape=SwingShape(
                    takeaway_s=takeaway,
                    top_s=takeaway + backswing,
                    impact_s=takeaway + backswing + downswing,
                    arc_radius=generator.uniform(0.17, 0.26),
                    arc_low_height=generator.uniform(0.28, 0.38),
                    arc_top_angle=generator.uniform(1.9, 2.5),
                ),
                torso_length=generator.uniform(0.20, 0.30),
                centre_x=generator.uniform(0.44, 0.56),
                shoulder_y=generator.uniform(0.20, 0.30),
            )
        )
    return players


def _clip(
    player: Player,
    *,
    fps: float,
    jitter: float,
    tempo: float,
    generator: random.Random,
) -> tuple[PoseSequence, SwingShape]:
    """One recording of one player's swing, and the shape that produced it.

    `tempo` stretches the swing in time, which is how two swings from one session
    differ by more than noise without becoming a different golfer. The returned
    shape is the stretched one, so the event times are the times in *this* clip.
    """
    shape = SwingShape(
        takeaway_s=player.shape.takeaway_s * tempo,
        top_s=player.shape.top_s * tempo,
        impact_s=player.shape.impact_s * tempo,
        arc_radius=player.shape.arc_radius,
        arc_low_height=player.shape.arc_low_height,
        arc_top_angle=player.shape.arc_top_angle,
    )
    duration = shape.finish_s + TAIL_S
    clock = np.arange(0.0, duration, 1.0 / fps)
    x, y = hand_path(clock, shape)

    # Jitter is added to the wrist path rather than to every landmark, because
    # it stands in for landmark noise on the fastest-moving point -- which is
    # where a real estimator is least certain and where the events are.
    noise = np.array([generator.gauss(0.0, jitter) for _ in range(2 * clock.size)])
    x = x + noise[: clock.size]
    y = y + noise[clock.size :]

    sequence = pose_sequence(x, y, clock, body=player.body(), path=f"/synthetic/{player.player_id}")
    return sequence, shape


def _label_for(
    player: Player,
    session: str,
    swing: str,
    shape: SwingShape,
    *,
    fps: float,
    frames: int,
    uncertainty: int,
) -> ClipLabel:
    """The ground-truth label: event times converted to frames of this clip."""

    def frame_of(seconds: float) -> int:
        return int(min(frames - 1, max(0, round(seconds * fps))))

    marks = {
        SwingEvent.TAKEAWAY: shape.takeaway_s,
        SwingEvent.TOP: shape.top_s,
        SwingEvent.IMPACT: shape.impact_s,
        SwingEvent.FINISH: shape.finish_s,
    }
    return ClipLabel(
        clip_path=f"/synthetic/{player.player_id}/{session}/{swing}.mov",
        content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED,
            digest=f"{abs(hash((player.player_id, session, swing))):064x}"[:64],
            size_bytes=1,
        ),
        player_id=player.player_id,
        session_id=session,
        swing_id=swing,
        provenance=LabelProvenance.SYNTHETIC,
        labeller="generator",
        labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
        is_swing=True,
        events=[
            EventLabel(
                event=event,
                frame_index=frame_of(seconds),
                # A bracket even though the generator knows the answer exactly.
                # Zero would make the noise floor the resampling alone, and every
                # evaluation run against this corpus would then be calibrated
                # against a precision no human labeller can supply -- which is
                # the opposite of what the fixture is for.
                uncertainty_frames=uncertainty,
            )
            for event, seconds in marks.items()
        ],
        frames=frames,
        fps=fps,
    )


@dataclass(frozen=True)
class Corpus:
    """Labelled clips and the pose sequences behind them, kept together."""

    labels: LabelSet
    sequences: dict[str, PoseSequence]
    """Keyed by `ClipLabel.clip_key`, which is how a provider finds one."""

    def provider(self) -> object:
        """A `PoseProvider` that filters this corpus' sequences on demand.

        Uses `filter_config_for`, exactly as the CLI's provider does, so the 30 fps
        sessions here meet the same widened smoothing window that a 30 fps phone
        clip does. A fixture that quietly filtered everything at 120 fps settings
        would hide the one confound this corpus can genuinely reproduce.
        """
        from analyzer.contracts.pose import LandmarkSpace
        from analyzer.filtering.landmarks import filter_sequence
        from analyzer.ml.dataset import filter_config_for

        def provide(label: ClipLabel) -> object:
            return filter_sequence(
                self.sequences[label.clip_key],
                filter_config_for(label.fps),
                space=LandmarkSpace.FRAME_WIDTHS,
            )

        return provide


def make_corpus(
    *,
    players: int = 9,
    sessions: int = 2,
    swings: int = 3,
    seed: int = 0,
    jitter: float = 0.002,
    uncertainty_frames: int = 1,
    non_swing_every: int = 6,
    rates: tuple[float, ...] = SESSION_RATES,
) -> Corpus:
    """Generate a labelled corpus of near-duplicate swings grouped by player.

    `rates` is the pool of frame rates sessions are shot at. Pass a single rate
    to hold it fixed, which is how a sweep isolates what the capture rate costs
    from everything else that differs between sessions.

    `non_swing_every` sprinkles clips containing no swing -- a subject standing
    over the ball, moving a little and never taking the club back. Without them
    the set cannot measure the behaviour that most distinguishes the rule-based
    detector: refusing to answer.
    """
    generator = random.Random(seed)  # noqa: S311 - a fixture, not a secret
    roster = make_players(players, seed=seed)
    labels: list[ClipLabel] = []
    sequences: dict[str, PoseSequence] = {}
    made = 0

    for player in roster:
        for session_index in range(sessions):
            session = f"s{session_index}"
            fps = rates[(seed + session_index + roster.index(player)) % len(rates)]
            for swing_index in range(swings):
                swing = f"w{swing_index}"
                made += 1
                if non_swing_every and made % non_swing_every == 0:
                    label, sequence = _still_clip(
                        player, session, swing, fps=fps, generator=generator
                    )
                else:
                    # Tempo wobble within a session is small: two swings minutes
                    # apart are the same swing, which is exactly why they leak.
                    sequence, shape = _clip(
                        player,
                        fps=fps,
                        jitter=jitter,
                        tempo=generator.uniform(0.97, 1.03),
                        generator=generator,
                    )
                    label = _label_for(
                        player,
                        session,
                        swing,
                        shape,
                        fps=fps,
                        frames=len(sequence.frames),
                        uncertainty=uncertainty_frames,
                    )
                labels.append(label)
                sequences[label.clip_key] = sequence

    return Corpus(labels=LabelSet(clips=labels), sequences=sequences)


def _still_clip(
    player: Player,
    session: str,
    swing: str,
    *,
    fps: float,
    generator: random.Random,
) -> tuple[ClipLabel, PoseSequence]:
    """A negative example: somebody at address who never swings.

    Not a frozen picture. A still subject still moves -- a waggle, a look at the
    target, the estimator's own noise -- and a negative clip of perfectly
    stationary landmarks would be a much easier question than a real one. Phase 4
    refuses this clip on hand travel; the point of including it is to find out
    whether a learned detector does.
    """
    duration = 2.0
    clock = np.arange(0.0, duration, 1.0 / fps)
    base_x, base_y = hand_path(np.zeros_like(clock), player.shape)
    wobble = np.array([generator.gauss(0.0, 0.004) for _ in range(2 * clock.size)])
    sequence = pose_sequence(
        base_x + wobble[: clock.size],
        base_y + wobble[clock.size :],
        clock,
        body=player.body(),
        path=f"/synthetic/{player.player_id}/still",
    )
    label = ClipLabel(
        clip_path=f"/synthetic/{player.player_id}/{session}/{swing}.mov",
        content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED,
            digest=f"{abs(hash((player.player_id, session, swing, 'still'))):064x}"[:64],
            size_bytes=1,
        ),
        player_id=player.player_id,
        session_id=session,
        swing_id=swing,
        provenance=LabelProvenance.SYNTHETIC,
        labeller="generator",
        labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
        is_swing=False,
        events=[],
        frames=len(sequence.frames),
        fps=fps,
    )
    return label, sequence
