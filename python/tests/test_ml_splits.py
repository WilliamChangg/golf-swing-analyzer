"""Group-aware splitting, and the refusal that is Phase 12's actual result.

The leak check is tested against a splitter built to fail it. A check that has
only ever been run on output it was designed alongside has not been tested at
all.
"""

from __future__ import annotations

import pytest

from analyzer.contracts.ml import SplitRole
from analyzer.ml.dataset import build_dataset
from analyzer.ml.splits import (
    SplitRefused,
    measure_leakage,
    plan_split,
    random_clip_split,
    split_dataset,
)
from tests.synthetic_labels import make_corpus


def labels(players: int = 6, sessions: int = 2, swings: int = 2, seed: int = 0):  # type: ignore[no-untyped-def]
    return list(
        make_corpus(players=players, sessions=sessions, swings=swings, seed=seed).labels.clips
    )


def test_a_single_player_set_is_refused_with_the_reason() -> None:
    """The state this project is in, and the phase's real finding."""
    report = plan_split(labels(players=1))
    assert report.refused
    assert "at least 3 players" in (report.refusal or "")
    assert "More clips of the same player do not help" in (report.refusal or "")


def test_two_players_are_still_refused() -> None:
    report = plan_split(labels(players=2))
    assert report.refused


def test_an_empty_set_is_refused_with_a_different_reason() -> None:
    report = plan_split([])
    assert report.refused
    assert "no labels" in (report.refusal or "")


def test_three_players_fill_all_three_roles() -> None:
    """Greedy shortfall alone sends the first two to training and empties test."""
    report = plan_split(labels(players=3))
    assert not report.refused
    assert all(report.players[role] >= 1 for role in SplitRole)


def test_no_player_session_or_clip_crosses_the_split() -> None:
    report = plan_split(labels(players=8))
    assert report.leakage is not None
    assert report.leakage.passed
    assert report.leakage.player_overlap == 0
    assert report.leakage.session_overlap == 0
    assert report.leakage.clip_overlap == 0


def test_every_clip_is_assigned_exactly_once() -> None:
    clips = labels(players=8)
    report = plan_split(clips)
    assert len(report.assignment) == len(clips)
    assert set(report.assignment) == {clip.clip_key for clip in clips}


def test_the_split_is_a_property_of_the_data_and_the_seed() -> None:
    """Two machines walking a directory differently must get the same split."""
    clips = labels(players=8)
    first = plan_split(clips, seed=7)
    second = plan_split(list(reversed(clips)), seed=7)
    assert first.assignment == second.assignment

    other_seed = plan_split(clips, seed=8)
    assert other_seed.assignment != first.assignment


def test_the_leak_check_catches_a_split_built_to_fail_it() -> None:
    """`random_clip_split` is a measuring instrument and must never pass for real."""
    clips = labels(players=6)
    report = random_clip_split(clips)
    assert not report.refused
    assert report.leakage is not None
    assert not report.leakage.passed
    assert report.leakage.player_overlap > 0
    assert "must never produce a published number" in report.warnings[0]


def test_the_leak_check_reads_an_assignment_it_did_not_produce() -> None:
    """It is a measurement of output, so it works on any splitter's output."""
    clips = labels(players=4)
    everything_in_train = dict.fromkeys((clip.clip_key for clip in clips), SplitRole.TRAIN)
    assert measure_leakage(everything_in_train, clips).passed

    leaked = dict(everything_in_train)
    leaked[clips[0].clip_key] = SplitRole.TEST
    check = measure_leakage(leaked, clips)
    assert check.player_overlap == 1
    assert not check.passed


def test_a_small_set_warns_even_when_it_can_be_split() -> None:
    report = plan_split(labels(players=3))
    joined = " ".join(report.warnings)
    assert "minimum for quoting a number" in joined
    assert "clips held out" in joined


def test_splitting_a_dataset_carries_the_examples_into_the_roles() -> None:
    corpus = make_corpus(players=4, sessions=1, swings=2, seed=2)
    built = build_dataset(corpus.labels, corpus.provider())
    split = split_dataset(built, seed=0)

    assert len(split.train) + len(split.val) + len(split.test) == len(built.examples)
    train_players = {example.label.player_id for example in split.train}
    test_players = {example.label.player_id for example in split.test}
    assert not (train_players & test_players)


def test_splitting_a_single_player_dataset_raises_with_the_report_attached() -> None:
    corpus = make_corpus(players=1, sessions=2, swings=2, seed=0)
    built = build_dataset(corpus.labels, corpus.provider())

    with pytest.raises(SplitRefused) as raised:
        split_dataset(built, seed=0)
    assert raised.value.report.refused
    assert raised.value.report.label_digest == built.summary.label_digest


def test_the_leaky_split_survives_into_the_dataset_split() -> None:
    """Rebuilding the assignment by player would quietly un-leak it."""
    corpus = make_corpus(players=4, sessions=1, swings=3, seed=5)
    built = build_dataset(corpus.labels, corpus.provider())
    split = split_dataset(built, seed=0, planner=random_clip_split)

    train_players = {example.label.player_id for example in split.train}
    test_players = {example.label.player_id for example in split.test}
    assert train_players & test_players
    assert split.report.leakage is not None
    assert not split.report.leakage.passed
