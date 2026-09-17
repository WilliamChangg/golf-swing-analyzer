"""Turning two clips' swing events, or a person's picks, into anchor pairs.

An anchor is a claim that one frame in each clip shows the same instant. Where it
comes from decides what can be said about it, not how it is used: the fit treats
a detected takeaway and a hand-picked impact identically, and `AnchorSource` is
carried so a reader can tell which is which.

## Pairing is by name, and checking is by consistency

Events pair unambiguously, because both clips were asked for the same four
instants and each came back named. The risk is not mispairing; it is that one of
the eight detections is simply wrong -- a top located on the wrong pause, an
impact on a blur -- and a single bad pair drags a two-parameter fit across all
four.

So the pairs are checked against each other before anything is fitted. Each pair
implies an offset of its own; a swing filmed by two cameras makes those offsets
agree to within the frame quantisation plus whatever clock difference exists, and
a pair that disagrees with the median by more than `max_anchor_disagreement_s` is
dropped with a note naming it. The median is used rather than the mean because
the thing being guarded against is exactly the outlier that would move a mean.

**This check cannot run on fewer than three pairs.** With two, "the median" is
their midpoint and each is equidistant from it, so no outlier is ever
identifiable -- which is stated here rather than discovered when a two-event clip
silently keeps a bad anchor.

## Manual anchors are the input, not an estimate of it

A person who has looked at both clips and picked the frames is asserting the
alignment; the engine's job is the arithmetic and the error accounting, not
second-guessing. They therefore score 1.0 on `confidence`, which means "as well
located as this clip's frame rate allows" and not "certainly correct". What
checks a manual anchor is the residual against the other anchors, and with a
single pick there is nothing to check it with -- which is what
`SyncConfidence.agreement` reports as zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyzer.contracts.phases import SwingEvent, SwingPhases
from analyzer.contracts.sync import AnchorSource, SyncAnchor, SyncConfig


@dataclass(frozen=True)
class ManualPick:
    """One instant a person identified in both clips, by frame number."""

    label: str
    reference_frame: int
    target_frame: int


class AnchorError(ValueError):
    """An anchor could not be built from what was supplied."""


def _timestamp(timestamps: np.ndarray, frame: int, *, clip: str) -> float:
    if not 0 <= frame < timestamps.size:
        raise AnchorError(
            f"Frame {frame} is outside the {clip} clip, which has "
            f"{timestamps.size} frames (0-{timestamps.size - 1})."
        )
    return float(timestamps[frame])


def manual_anchors(
    picks: list[ManualPick],
    reference_t: np.ndarray,
    target_t: np.ndarray,
) -> list[SyncAnchor]:
    """Convert frame picks into anchors on each clip's real-seconds clock.

    The conversion has to happen here rather than in the caller because only the
    engine holds the clips' presentation timestamps, and on variable-rate footage
    `frame / fps` is not the time the frame was taken -- which is the whole
    reason Phase 1 reads the packet index.
    """
    anchors: list[SyncAnchor] = []
    for pick in picks:
        anchors.append(
            SyncAnchor(
                label=pick.label,
                event=None,
                source=AnchorSource.MANUAL,
                reference_frame=pick.reference_frame,
                target_frame=pick.target_frame,
                reference_s=_timestamp(reference_t, pick.reference_frame, clip="reference"),
                target_s=_timestamp(target_t, pick.target_frame, clip="target"),
                confidence=1.0,
            )
        )
    return anchors


def event_anchors(reference: SwingPhases, target: SwingPhases) -> list[SyncAnchor]:
    """Pair the swing events both clips located, in swing order.

    An event found in one clip and not the other yields no anchor. That is not a
    failure of either detection: a down-the-line camera loses the hands to motion
    blur through impact far more readily than a face-on one, and an event the
    engine declined to locate is precisely an event it should not be aligning on.
    """
    anchors: list[SyncAnchor] = []
    for event in SwingEvent:
        left, right = reference.event(event), target.event(event)
        if left is None or right is None:
            continue
        anchors.append(
            SyncAnchor(
                label=event.value,
                event=event,
                source=AnchorSource.DETECTED,
                reference_frame=left.frame_index,
                target_frame=right.frame_index,
                reference_s=left.timestamp_s,
                target_s=right.timestamp_s,
                # The weaker of the two: an instant is only as well located as
                # the clip that saw it worst, and a product would punish a pair
                # of good detections for being two rather than one.
                confidence=min(left.confidence.overall, right.confidence.overall),
            )
        )
    return anchors


def reject_outliers(
    anchors: list[SyncAnchor], config: SyncConfig
) -> tuple[list[SyncAnchor], list[str]]:
    """Drop anchor pairs whose implied offset disagrees with the rest.

    Returns the survivors and a note for each rejection, naming the anchor and
    by how much it disagreed, because a silently discarded anchor is a silently
    changed answer.
    """
    if len(anchors) < 3:
        return list(anchors), []

    offsets = np.array([anchor.target_s - anchor.reference_s for anchor in anchors])
    median = float(np.median(offsets))
    deviation = np.abs(offsets - median)
    keep = deviation <= config.max_anchor_disagreement_s

    if bool(np.all(keep)):
        return list(anchors), []

    # Never strip the evidence down below what determines the map. If most of the
    # pairs disagree with each other there is no majority to trust, and throwing
    # away the minority would manufacture agreement out of an arbitrary choice.
    if int(np.count_nonzero(keep)) < 2:
        return list(anchors), [
            "The event pairs disagree with each other by more than "
            f"{config.max_anchor_disagreement_s:g} s and no two of them agree, so none were "
            "dropped: there is no majority here to believe. The residual below is the "
            "measurement of that disagreement."
        ]

    notes = [
        f"The '{anchors[index].label}' pair implies an offset {offsets[index] - median:+.3f} s "
        f"from the median of the others and was dropped before fitting; one of its two "
        "detections is in the wrong place."
        for index in range(len(anchors))
        if not keep[index]
    ]
    return [anchor for anchor, kept in zip(anchors, keep, strict=True) if kept], notes
