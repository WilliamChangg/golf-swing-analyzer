"""Confidence gating: deciding which detections count as observations.

A pose estimator reports a landmark for every frame in which it found a body,
including landmarks it could not actually see. An elbow behind the torso still
gets coordinates, and those coordinates are the model's prior rather than a
measurement of anything. Passing them into a smoother launders a guess into a
trajectory, and the trajectory carries no mark saying which parts were guessed.

This stage removes them, and does it by converting the detection into the same
NaN that a frame with no detection at all produces. Downstream there is then one
representation of "not observed", not two, and the gap policy below cannot be
fooled into bridging an absence it could not see.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.filtering import ConfidenceGate, StageReport
from analyzer.filtering.signal import Signal


def below_threshold(signal: Signal, gate: ConfidenceGate) -> NDArray[np.bool_]:
    """Samples whose reported confidence falls short of the policy.

    Written as `~(x >= threshold)` rather than `x < threshold` so that a NaN
    confidence gates the sample out. NaN compares False either way, and the
    difference is whether an unknown confidence is treated as acceptable or as
    unacceptable. Unknown is not evidence of visibility.
    """
    return ~(signal.visibility >= gate.min_visibility) | ~(signal.presence >= gate.min_presence)


def gated_observations(signal: Signal, gate: ConfidenceGate) -> NDArray[np.bool_]:
    """Frames the estimator saw well enough for the gate to keep.

    Exported because callers need this question answered about the *input* to
    filtering, and `Signal.observed` on the pipeline's output cannot answer it:
    by then `value` holds fitted numbers, so the mask reports where the fit
    succeeded rather than where the estimator looked. The two differ at the ends
    of every clip, and conflating them turns a window-width problem into a
    reported camera problem.
    """
    return ~np.isnan(signal.value) & ~below_threshold(signal, gate)


class ConfidenceGateStage:
    """Reject detections whose reported confidence is below the policy."""

    def __init__(self, gate: ConfidenceGate) -> None:
        self._gate = gate

    @property
    def name(self) -> str:
        return "confidence_gate"

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        present = ~np.isnan(signal.value)
        below = below_threshold(signal, self._gate)
        rejected = present & below

        value = np.where(rejected, np.nan, signal.value)
        weight = np.where(rejected, 0.0, signal.weight)

        report = StageReport(
            stage=self.name,
            counts={
                "samples": len(signal),
                "detected": int(np.count_nonzero(present)),
                "gated_out": int(np.count_nonzero(rejected)),
                "never_detected": int(np.count_nonzero(~present)),
                "observed": int(np.count_nonzero(present & ~below)),
                "low_visibility": int(
                    np.count_nonzero(present & ~(signal.visibility >= self._gate.min_visibility))
                ),
                "low_presence": int(
                    np.count_nonzero(present & ~(signal.presence >= self._gate.min_presence))
                ),
            },
            measurements={
                "min_visibility": self._gate.min_visibility,
                "min_presence": self._gate.min_presence,
            },
        )
        return signal.with_values(value, weight=weight), report
