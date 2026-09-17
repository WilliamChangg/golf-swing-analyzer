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

from analyzer.contracts.filtering import ConfidenceGate, StageReport
from analyzer.filtering.signal import Signal


class ConfidenceGateStage:
    """Reject detections whose reported confidence is below the policy."""

    def __init__(self, gate: ConfidenceGate) -> None:
        self._gate = gate

    @property
    def name(self) -> str:
        return "confidence_gate"

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        present = ~np.isnan(signal.value)

        # Written as `~(x >= threshold)` rather than `x < threshold` so that a
        # NaN confidence gates the sample out. NaN compares False either way,
        # and the difference is whether an unknown confidence is treated as
        # acceptable or as unacceptable. Unknown is not evidence of visibility.
        below = ~(signal.visibility >= self._gate.min_visibility) | ~(
            signal.presence >= self._gate.min_presence
        )
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
