"""Gap policy: what to do about the frames where nothing was observed.

Two decisions, and the second one matters more.

The first is how to bridge a short absence. A frame or two lost to a blurred
hand at the bottom of a downswing is a hole the surrounding motion genuinely
determines, and linear interpolation in time across it is defensible: it cannot
overshoot, it introduces no oscillation, and it is exactly recovered by the
local polynomial fit that runs afterwards.

The second is when to refuse. A local polynomial fit does not know it is sitting
in the middle of a hole -- give it observations at both edges of its window and
it will produce a smooth, plausible, entirely fictional value in between, and
that value arrives downstream indistinguishable from a measured one. `max_gap_s`
is the line, and samples past it are marked `blocked`, which every later stage
is required to honour. A blocked sample stays NaN through to the output.

Absences at the very start or end of a clip are always blocked, whatever their
length. Bridging them would mean extrapolating from one side only, and a swing's
extremes are where extrapolation is least defensible: the fastest motion in the
clip is near impact, which is often exactly where tracking is lost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.filtering import GapPolicy, StageReport
from analyzer.filtering.signal import Signal


def missing_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal half-open [start, stop) runs of True in `mask`."""
    if mask.size == 0 or not bool(mask.any()):
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


class GapPolicyStage:
    """Bridge short absences; mark long ones as permanently unknown."""

    def __init__(self, policy: GapPolicy) -> None:
        self._policy = policy

    @property
    def name(self) -> str:
        return "gap_policy"

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        count = len(signal)
        value = signal.value.copy()
        weight = signal.weight.copy()
        filled = signal.filled.copy()
        blocked = signal.blocked.copy()

        runs = missing_runs(np.isnan(signal.value))
        longest_gap_s = 0.0
        bridged = refused = unanchored = 0

        for start, stop in runs:
            has_left, has_right = start > 0, stop < count

            # Elapsed time the absence covers: from the last observation before
            # it to the first after it. Clamped to the clip at the ends, so a
            # leading or trailing run still reports a length even though it can
            # never be bridged.
            left_t = signal.t[start - 1] if has_left else signal.t[start]
            right_t = signal.t[stop] if has_right else signal.t[stop - 1]
            span = float(right_t - left_t)
            longest_gap_s = max(longest_gap_s, span)

            if not (has_left and has_right):
                blocked[start:stop] = True
                unanchored += 1
                continue

            if span > self._policy.max_gap_s:
                blocked[start:stop] = True
                refused += 1
                continue

            anchors_t = signal.t[[start - 1, stop]]
            interior = signal.t[start:stop]
            value[start:stop] = np.interp(interior, anchors_t, signal.value[[start - 1, stop]])

            # The filled sample's confidence is interpolated from its anchors
            # and then scaled by the policy factor -- zero by default, because
            # an interpolated value is a function of the neighbours that will
            # already be in the fitting window, and weighting it would count
            # them twice.
            anchor_weight = np.interp(interior, anchors_t, signal.weight[[start - 1, stop]])
            weight[start:stop] = self._policy.filled_weight * anchor_weight
            filled[start:stop] = True
            bridged += 1

        report = StageReport(
            stage=self.name,
            counts={
                "samples": count,
                "gaps": len(runs),
                "gaps_bridged": bridged,
                "gaps_refused": refused,
                "gaps_unanchored": unanchored,
                "samples_filled": int(np.count_nonzero(filled & ~signal.filled)),
                "samples_blocked": int(np.count_nonzero(blocked & ~signal.blocked)),
            },
            measurements={
                "max_gap_s": self._policy.max_gap_s,
                "longest_gap_s": longest_gap_s,
                "filled_weight": self._policy.filled_weight,
            },
        )
        return signal.with_values(value, weight=weight, filled=filled, blocked=blocked), report
