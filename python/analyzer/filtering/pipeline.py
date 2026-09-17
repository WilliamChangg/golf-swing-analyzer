"""Composing filter stages, and recording what each one did.

`FilterStage` is a Protocol for the same reason `FrameSource` and `PoseEstimator`
are: the stages here are the obvious baseline, not the last word. A Kalman
smoother, an outlier rejector keyed on bone-length consistency, or a learned
denoiser from Phase 12 can each be dropped into the sequence without any caller
knowing, provided it can say what it did.

Saying what it did is the part that is not optional. Every stage returns a
`StageReport` alongside the signal, and the pipeline keeps them in order, so a
filtered trajectory can always be traced back through the decisions that shaped
it -- how many detections the gate rejected, how many gaps were bridged and how
many refused, how many windows lacked the support to fit. Without that record a
filtered signal is just an array of plausible numbers.

The pipeline enforces one invariant of its own: **a blocked sample is NaN in the
output.** Stages are asked to honour it, and the pipeline checks rather than
trusting, because a stage that quietly filled a refused gap would produce
exactly the kind of fabricated value this layer exists to prevent.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from analyzer.contracts.filtering import FilterConfig, StageReport
from analyzer.filtering.gaps import GapPolicyStage
from analyzer.filtering.gating import ConfidenceGateStage
from analyzer.filtering.signal import Signal
from analyzer.filtering.smoothing import LocalPolynomialStage


class FilterPipelineError(RuntimeError):
    """A stage violated a guarantee the pipeline is responsible for."""


@runtime_checkable
class FilterStage(Protocol):
    """One step of the filtering pipeline.

    A stage takes a signal and returns a new one plus a report of what it
    changed. Signals are frozen, so a stage cannot modify its input in place and
    a caller's copy stays valid -- which is what makes it possible to run the
    same input through two configurations and compare them.
    """

    @property
    def name(self) -> str:
        """Stable identifier, used in reports."""
        ...

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]: ...


class FilterPipeline:
    """An ordered list of stages, applied in sequence."""

    def __init__(self, stages: list[FilterStage]) -> None:
        if not stages:
            raise FilterPipelineError("A pipeline needs at least one stage.")
        self._stages = list(stages)

    @property
    def stages(self) -> tuple[FilterStage, ...]:
        return tuple(self._stages)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    def run(self, signal: Signal) -> tuple[Signal, list[StageReport]]:
        """Apply every stage in order, collecting reports."""
        reports: list[StageReport] = []
        current = signal

        for stage in self._stages:
            current, report = stage.apply(current)
            if len(current) != len(signal):
                raise FilterPipelineError(
                    f"Stage '{stage.name}' returned {len(current)} samples for an input of "
                    f"{len(signal)}. A stage may change values, not the timeline."
                )
            reports.append(report)

        self._assert_blocked_are_empty(current)
        return current, reports

    @staticmethod
    def _assert_blocked_are_empty(signal: Signal) -> None:
        """A gap the policy refused must not have acquired a value on the way out.

        Checked rather than assumed. This is the guarantee the whole layer is
        built to make, and a stage added later that smooths across a blocked
        span would otherwise break it silently and produce fabricated positions
        that look exactly like measured ones.
        """
        if not np.any(signal.blocked):
            return

        channels = {
            "value": signal.value,
            "velocity": signal.velocity,
            "acceleration": signal.acceleration,
        }
        for name, channel in channels.items():
            if channel is None:
                continue
            leaked = int(np.count_nonzero(signal.blocked & ~np.isnan(channel)))
            if leaked:
                raise FilterPipelineError(
                    f"{leaked} sample(s) inside a gap the policy refused to bridge came out "
                    f"with a {name}. Those values are not supported by any observation and "
                    "must stay NaN."
                )


def default_pipeline(config: FilterConfig) -> FilterPipeline:
    """The Phase 3 baseline: gate, then gap policy, then fit.

    The order is forced by what each stage needs. Gating must precede the gap
    policy, or a rejected detection is not yet an absence and the gap it belongs
    to is measured too short. The fit must come last, because it is the only
    stage that reads weights, and both stages before it set them.
    """
    return FilterPipeline(
        [
            ConfidenceGateStage(config.gate),
            GapPolicyStage(config.gaps),
            LocalPolynomialStage(config.smoothing),
        ]
    )
