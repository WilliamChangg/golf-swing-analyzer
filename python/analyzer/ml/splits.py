"""Dividing a labelled set, and refusing to when it cannot be divided honestly.

The single most consequential file in Phase 12, and the shortest way to say why
is that every published golf-swing accuracy figure is either produced by
something like this or is wrong.

## What leaks

Swings from one session are near-duplicates. Same player, same club, same
camera, same light, same ball position, minutes apart, and a tour professional's
swing is repeatable to a degree that makes two of them differ by less than the
noise in the landmarks. Put one in training and its neighbour in test and the
model is being asked a question it has already been given the answer to. The
number that comes out is not a small overstatement -- `scripts/benchmark_ml.py`
measures it on the synthetic corpus, where the answer is knowable.

## What is held together

The whole player. Grouping by session is the tempting half-measure: it keeps
near-duplicates together and still lets one golfer appear on both sides of the
split, which measures how well the model recognises *that golfer* on a different
day. Grouping by player holds sessions together as a consequence -- a session
belongs to exactly one player -- so the stronger grouping is also the simpler
one.

## What refusal means

A set of four clips of one golfer has no held-out player, so it produces a
`SplitReport` with `refused` set and a reason, not a split. That is the state
this project is in as Phase 12 is written, and the refusal is the phase's
result rather than an obstacle to it. A splitter that returned something anyway
would be manufacturing the only thing that makes a held-out number mean
anything.

## The leak check measures the output

`LeakageCheck` is computed from the assignment that came out, not asserted by the
code that made it. An assertion inside a splitter checks that the splitter agrees
with itself; this checks the thing that will actually be trained on. It is what
catches `random_clip_split` below -- a deliberately leaky split kept here as a
measuring instrument, which fails the check by construction and can therefore
never be mistaken for the real one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from analyzer.contracts.labels import ClipLabel
from analyzer.contracts.ml import (
    MIN_PLAYERS_FOR_CLAIM,
    MIN_TEST_CLIPS_FOR_CLAIM,
    LeakageCheck,
    SplitGroup,
    SplitReport,
    SplitRole,
)
from analyzer.ml.dataset import Dataset, Example

ROLES: tuple[SplitRole, ...] = (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)

# Fractions of clips, not of players: players hold wildly different numbers of
# clips, and a split that balanced players would leave a test set of one golfer
# who happened to be filmed once.
DEFAULT_RATIOS: dict[SplitRole, float] = {
    SplitRole.TRAIN: 0.6,
    SplitRole.VAL: 0.2,
    SplitRole.TEST: 0.2,
}


class SplitRefused(RuntimeError):  # noqa: N818 - see below
    """No honest split exists for this set. Carries the report that says why.

    Named for the outcome rather than with an `Error` suffix, deliberately. A
    refusal here is a correct answer about the data -- the state this project is
    in -- and `except SplitRefused` reads as the case it is, where
    `SplitRefusedError` would read as a fault to be fixed.
    """

    def __init__(self, report: SplitReport) -> None:
        super().__init__(report.refusal or "This set cannot be split.")
        self.report = report


@dataclass(frozen=True)
class Split:
    """One assignment of examples to the three roles."""

    train: tuple[Example, ...]
    val: tuple[Example, ...]
    test: tuple[Example, ...]
    report: SplitReport

    def role(self, which: SplitRole) -> tuple[Example, ...]:
        return {
            SplitRole.TRAIN: self.train,
            SplitRole.VAL: self.val,
            SplitRole.TEST: self.test,
        }[which]


def measure_leakage(assignment: dict[str, SplitRole], labels: list[ClipLabel]) -> LeakageCheck:
    """Count what appears in more than one role, reading the assignment itself.

    Takes clip keys mapped to roles and the labels behind them, so it can be
    called on any assignment from any source -- including one produced by a
    splitter that is not this module's.
    """
    players: dict[str, set[SplitRole]] = {}
    sessions: dict[str, set[SplitRole]] = {}
    contents: dict[str, set[SplitRole]] = {}

    for label in labels:
        role = assignment.get(label.clip_key)
        if role is None:
            continue
        players.setdefault(label.player_id, set()).add(role)
        sessions.setdefault(label.session_key, set()).add(role)
        if label.content_key is not None:
            contents.setdefault(label.content_key.digest, set()).add(role)

    return LeakageCheck(
        player_overlap=sum(1 for roles in players.values() if len(roles) > 1),
        session_overlap=sum(1 for roles in sessions.values() if len(roles) > 1),
        clip_overlap=sum(1 for roles in contents.values() if len(roles) > 1),
    )


def _report(
    assignment: dict[str, SplitRole],
    labels: list[ClipLabel],
    *,
    seed: int,
    grouped_by: str,
    label_digest: str,
) -> SplitReport:
    """Describe an assignment: its groups, its counts, and its measured overlap."""
    by_player: dict[str, list[ClipLabel]] = {}
    for label in labels:
        if label.clip_key in assignment:
            by_player.setdefault(label.player_id, []).append(label)

    # One row per player and role. A player appearing in two rows is a leak, and
    # is listed that way rather than being folded into whichever role holds most
    # of their clips -- the table is the place it is most visible.
    groups: list[SplitGroup] = []
    for player, clips in sorted(by_player.items()):
        for role in ROLES:
            held = [clip for clip in clips if assignment[clip.clip_key] is role]
            if not held:
                continue
            groups.append(
                SplitGroup(
                    role=role,
                    player_id=player,
                    sessions=len({clip.session_key for clip in held}),
                    clips=len(held),
                    swings=sum(1 for clip in held if clip.is_swing),
                )
            )

    clips_per_role = {
        role: sum(1 for value in assignment.values() if value is role) for role in ROLES
    }
    players_per_role = {
        role: len({label.player_id for label in labels if assignment.get(label.clip_key) is role})
        for role in ROLES
    }

    warnings: list[str] = []
    if players_per_role[SplitRole.TEST] < 2:
        warnings.append(
            f"The held-out set contains {players_per_role[SplitRole.TEST]} player(s). "
            "Any number measured on it is a statement about that person."
        )
    if len(by_player) < MIN_PLAYERS_FOR_CLAIM:
        warnings.append(
            f"{len(by_player)} players in the whole set, below the {MIN_PLAYERS_FOR_CLAIM} "
            "this project treats as the minimum for quoting a number at all."
        )
    if clips_per_role[SplitRole.TEST] < MIN_TEST_CLIPS_FOR_CLAIM:
        warnings.append(
            f"{clips_per_role[SplitRole.TEST]} clips held out, below the "
            f"{MIN_TEST_CLIPS_FOR_CLAIM} a rate could be quoted from."
        )
    if not any(
        not label.is_swing and assignment.get(label.clip_key) is SplitRole.TEST for label in labels
    ):
        warnings.append(
            "The held-out set has no clip without a swing in it, so it cannot measure "
            "whether a detector knows when to answer nothing."
        )

    return SplitReport(
        seed=seed,
        grouped_by=grouped_by,
        refused=False,
        groups=groups,
        assignment=dict(assignment),
        clips=clips_per_role,
        players=players_per_role,
        leakage=measure_leakage(assignment, labels),
        label_digest=label_digest,
        warnings=warnings,
    )


def _refusal(reason: str, *, seed: int, label_digest: str) -> SplitReport:
    return SplitReport(seed=seed, refused=True, refusal=reason, label_digest=label_digest)


def plan_split(
    labels: list[ClipLabel],
    *,
    seed: int = 0,
    ratios: dict[SplitRole, float] | None = None,
    label_digest: str = "",
) -> SplitReport:
    """Assign whole players to the three roles, or explain why that is impossible.

    Deterministic in `seed`: the players are sorted before they are shuffled, so
    the assignment does not depend on the order a directory happened to be walked
    in. Two runs of this project on two machines produce the same split or the
    split is not a property of the data.
    """
    shares = ratios or DEFAULT_RATIOS
    players = sorted({label.player_id for label in labels})

    if not labels:
        return _refusal(
            "There are no labels to split. Phase 12's machinery is built and has "
            "nothing to run on until a labelled set exists.",
            seed=seed,
            label_digest=label_digest,
        )
    if len(players) < len(ROLES):
        return _refusal(
            f"A train/validation/test split needs at least {len(ROLES)} players and this "
            f"set has {len(players)}: {', '.join(players)}. Splitting by clip instead "
            "would put the same golfer on both sides, and every number measured after "
            "that would be about how well the model recognises a person it has already "
            "been trained on. More clips of the same player do not help; more players do.",
            seed=seed,
            label_digest=label_digest,
        )

    counts = {player: sum(1 for label in labels if label.player_id == player) for player in players}
    total = sum(counts.values())
    targets = {role: shares.get(role, 0.0) * total for role in ROLES}

    order = list(players)
    # Reproducibility, not secrecy: the seed is recorded on the report so the
    # same data and the same seed give the same split on another machine.
    random.Random(seed).shuffle(order)  # noqa: S311

    # Largest players first, so the biggest indivisible lump is placed while every
    # role still has room. Placing them in arrival order routinely gives a test
    # set of one prolific player and a training set of everyone else.
    order.sort(key=lambda player: counts[player], reverse=True)

    held: dict[SplitRole, list[str]] = {role: [] for role in ROLES}
    filled = dict.fromkeys(ROLES, 0.0)

    # Seed every role with one player before filling any of them. Greedy
    # shortfall alone does not do this: with three equal players and a 60/20/20
    # target, the first two both go to training -- its shortfall is still the
    # largest after the first -- and the held-out set ends up empty. Seeding in
    # order of target size gives the largest player to the largest role, which is
    # also what the greedy pass would have done for that one.
    seeding = sorted(ROLES, key=lambda role: targets[role], reverse=True)
    for role, player in zip(seeding, order, strict=False):
        held[role].append(player)
        filled[role] += counts[player]

    for player in order[len(ROLES) :]:
        # To whichever role is furthest below its target, measured as a shortfall
        # in clips rather than as a ratio: a ratio sends every early player to
        # whichever role is emptiest regardless of how many clips that needs.
        role = max(ROLES, key=lambda role: (targets[role] - filled[role], -ROLES.index(role)))
        held[role].append(player)
        filled[role] += counts[player]

    assignment = {
        label.clip_key: role
        for role in ROLES
        for player in held[role]
        for label in labels
        if label.player_id == player
    }
    return _report(assignment, labels, seed=seed, grouped_by="player", label_digest=label_digest)


def random_clip_split(
    labels: list[ClipLabel],
    *,
    seed: int = 0,
    ratios: dict[SplitRole, float] | None = None,
    label_digest: str = "",
) -> SplitReport:
    """The wrong split: clips assigned individually, ignoring who swung them.

    Kept, and kept here beside the right one, because the size of the mistake is
    a measurement this project wanted and could not otherwise make. Training
    under this split and under `plan_split` on the same corpus, with the same
    seed and the same model, gives the difference between what a leaky evaluation
    reports and what the model can actually do. `scripts/benchmark_ml.py` runs
    exactly that.

    It cannot be mistaken for the real thing: the report it produces fails its own
    leak check, because the check reads the assignment rather than the intent, and
    every consumer in this package treats a failed check as disqualifying.
    """
    shares = ratios or DEFAULT_RATIOS
    keys = sorted(label.clip_key for label in labels)
    if not keys:
        return _refusal("There are no labels to split.", seed=seed, label_digest=label_digest)

    random.Random(seed).shuffle(keys)  # noqa: S311 - reproducibility, not secrecy
    cuts = {
        SplitRole.TRAIN: round(shares[SplitRole.TRAIN] * len(keys)),
        SplitRole.VAL: round(shares[SplitRole.VAL] * len(keys)),
    }
    assignment: dict[str, SplitRole] = {}
    for index, key in enumerate(keys):
        if index < cuts[SplitRole.TRAIN]:
            assignment[key] = SplitRole.TRAIN
        elif index < cuts[SplitRole.TRAIN] + cuts[SplitRole.VAL]:
            assignment[key] = SplitRole.VAL
        else:
            assignment[key] = SplitRole.TEST

    report = _report(
        assignment, labels, seed=seed, grouped_by="clip (LEAKY)", label_digest=label_digest
    )
    report.warnings.insert(
        0,
        "This split ignores who is swinging. It exists to measure what that costs and "
        "must never produce a published number.",
    )
    return report


def split_dataset(
    dataset: Dataset,
    *,
    seed: int = 0,
    ratios: dict[SplitRole, float] | None = None,
    planner: object = None,
) -> Split:
    """Apply a split to a built dataset, raising `SplitRefused` when none exists.

    `planner` takes `plan_split` or `random_clip_split`; the default is the
    honest one, and the leaky one has to be passed in by name at the call site so
    that it appears in the code of anything that uses it.
    """
    plan = planner if planner is not None else plan_split
    labels = [item.label for item in dataset.examples]
    report = plan(  # type: ignore[operator]
        labels, seed=seed, ratios=ratios, label_digest=dataset.summary.label_digest
    )
    if report.refused:
        raise SplitRefused(report)

    roles: dict[SplitRole, list[Example]] = {role: [] for role in ROLES}
    for item in dataset.examples:
        role = report.assignment.get(item.label.clip_key)
        if role is not None:
            roles[role].append(item)

    return Split(
        train=tuple(roles[SplitRole.TRAIN]),
        val=tuple(roles[SplitRole.VAL]),
        test=tuple(roles[SplitRole.TEST]),
        report=report,
    )
