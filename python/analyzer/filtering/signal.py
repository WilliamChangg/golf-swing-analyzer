"""The value that flows through the filter pipeline.

One scalar channel of one landmark -- the left wrist's x through time, say --
together with everything needed to decide how much of it is real.

A `Signal` carries more than samples because every stage downstream has to make
a decision that depends on provenance. `visibility` and `presence` are what the
model said about each detection, so the gate can apply its two thresholds
without reaching back to the pose sequence. `weight` is how much the fit should
trust each sample. `filled` marks values that were interpolated rather than
observed, and `blocked` marks samples the gap policy has ruled permanently
unknowable -- an instruction to every later stage, not a description of the
current values.

The type is frozen, and the only way to change values is `with_values`, which
**drops any derivatives that were attached.** That is the one invariant worth
enforcing in the type: a velocity computed from values that have since been
replaced is not wrong in any way a reader would notice, which is exactly why it
must be impossible to carry forward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


class SignalError(ValueError):
    """A signal was constructed from arrays that cannot describe a time series."""


@dataclass(frozen=True)
class Signal:
    """One scalar channel through time, with its provenance."""

    t: NDArray[np.float64]
    value: NDArray[np.float64]
    visibility: NDArray[np.float64]
    presence: NDArray[np.float64]
    weight: NDArray[np.float64]
    filled: NDArray[np.bool_]
    blocked: NDArray[np.bool_]
    velocity: NDArray[np.float64] | None = None
    acceleration: NDArray[np.float64] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        count = self.t.size
        named = {
            "value": self.value,
            "visibility": self.visibility,
            "presence": self.presence,
            "weight": self.weight,
            "filled": self.filled,
            "blocked": self.blocked,
        }
        for name, array in named.items():
            if array.size != count:
                raise SignalError(
                    f"{name} has {array.size} samples but t has {count}; every channel of a "
                    "signal must be sampled at the same instants."
                )

        if count > 1 and not np.all(np.diff(self.t) > 0):
            # Not merely tidiness: the windowing below is a binary search over t,
            # and the local fit re-centres on t, so an out-of-order or repeated
            # timestamp silently produces a fit over the wrong neighbourhood.
            raise SignalError(
                "Timestamps must be strictly increasing. Pose sequences come from the "
                "container's sorted packet index, so this usually means the samples were "
                "concatenated or reordered somewhere upstream."
            )

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def observed(self) -> NDArray[np.bool_]:
        """Samples carrying a real, positively-weighted observation.

        Interpolated samples are excluded however they are weighted: they were
        not observed, and that is what this property is asked.
        """
        return (~np.isnan(self.value)) & (~self.filled) & (self.weight > 0.0)

    @property
    def missing(self) -> NDArray[np.bool_]:
        """Samples with no value at all."""
        return np.isnan(self.value)

    def with_values(
        self,
        value: NDArray[np.float64],
        *,
        weight: NDArray[np.float64] | None = None,
        filled: NDArray[np.bool_] | None = None,
        blocked: NDArray[np.bool_] | None = None,
    ) -> Signal:
        """A copy with new values, and **no derivatives**.

        Derivatives are dropped rather than recomputed or carried: they
        described the previous values, and a stale derivative is the kind of
        wrong number that survives review because it looks plausible.
        """
        return replace(
            self,
            value=value,
            weight=self.weight if weight is None else weight,
            filled=self.filled if filled is None else filled,
            blocked=self.blocked if blocked is None else blocked,
            velocity=None,
            acceleration=None,
        )

    def with_derivatives(
        self,
        value: NDArray[np.float64],
        velocity: NDArray[np.float64],
        acceleration: NDArray[np.float64],
    ) -> Signal:
        """A copy carrying a fitted value and the derivatives of that same fit.

        Separate from `with_values` because these three came out of one
        polynomial and are consistent with each other by construction; nothing
        else in the pipeline is allowed to attach derivatives.
        """
        return replace(self, value=value, velocity=velocity, acceleration=acceleration)


def signal_from_arrays(
    t: NDArray[np.float64],
    value: NDArray[np.float64],
    *,
    visibility: NDArray[np.float64] | None = None,
    presence: NDArray[np.float64] | None = None,
    label: str = "",
) -> Signal:
    """Build a signal from samples, defaulting confidence to full where observed.

    The confidence default is deliberate: a caller with no confidence
    information -- a synthetic trajectory in a test, or a source that reports
    none -- is asserting that the samples it does have are observations. A
    default of zero would instead silently gate the entire signal out.
    """
    missing = np.isnan(value)

    def _confidence(supplied: NDArray[np.float64] | None) -> NDArray[np.float64]:
        if supplied is not None:
            return np.asarray(supplied, dtype=np.float64)
        return np.where(missing, np.nan, 1.0)

    vis = _confidence(visibility)
    pres = _confidence(presence)

    # The weaker of the two, not their product: they describe related events
    # (in frame, unoccluded) and multiplying them would assert an independence
    # the model does not claim. A landmark is as trustworthy as its weakest
    # reported property.
    weight = np.fmin(vis, pres)
    weight = np.where(missing | np.isnan(weight), 0.0, weight)

    count = t.size
    return Signal(
        t=np.asarray(t, dtype=np.float64),
        value=np.asarray(value, dtype=np.float64),
        visibility=vis,
        presence=pres,
        weight=weight,
        filled=np.zeros(count, dtype=np.bool_),
        blocked=np.zeros(count, dtype=np.bool_),
        label=label,
    )
