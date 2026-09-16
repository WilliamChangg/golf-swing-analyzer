"""The smoothing and differentiation stage.

Thin by design: all the numerics are in `localpoly`, and this exists to adapt
them to the pipeline and to count what happened. The one decision it makes is
that position, velocity and acceleration are produced together, by one call,
from one polynomial per sample -- so they are mutually consistent and no later
stage has the opportunity to differentiate the smoothed positions and get a
different, worse answer.
"""

from __future__ import annotations

import numpy as np

from analyzer.contracts.filtering import SmoothingConfig, StageReport
from analyzer.filtering.localpoly import local_polynomial_fit
from analyzer.filtering.signal import Signal


class LocalPolynomialStage:
    """Fit a local polynomial at every sample; read position and derivatives off it."""

    def __init__(self, config: SmoothingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "local_polynomial"

    @staticmethod
    def _remedy(signal: Signal, required: int) -> str:
        """Turn "nothing could be fitted" into a number the caller can act on.

        Saying the frame rate is too low is true and useless on its own. The
        widest sensible response is either to widen the window -- computed here
        from the clip's own measured sampling interval -- or to recapture, and
        which of those is right depends on how much motion the wider window
        would then average over. Both are stated so the caller can choose.
        """
        if len(signal) < 2:
            return "The clip is too short to fit anything over."

        interval = float(np.median(np.diff(signal.t)))
        if interval <= 0:
            return "The clip's timestamps do not advance."

        # Half an interval of headroom, so the window's boundary does not land
        # on a sample and decide the count by floating-point luck.
        needed = (required - 0.5) * interval
        return (
            f"Sampling is about {1.0 / interval:.0f} fps, so this order needs a window of at "
            f"least {needed:.3f} s. Either widen it, which averages over more of the swing, "
            "or recapture at a higher frame rate -- the capture protocol asks for 120 fps or "
            "more for exactly this reason."
        )

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        observed = signal.observed
        fit = local_polynomial_fit(
            signal.t,
            signal.value,
            signal.weight,
            window_s=self._config.window_s,
            polyorder=self._config.polyorder,
            min_observations=self._config.required_observations,
            blocked=signal.blocked,
            residual_mask=observed,
        )

        # Separated so the report distinguishes "the policy forbade a value
        # here" from "there was not enough data to fit one". They need different
        # responses: the first is a capture problem, the second may be a window
        # that is too narrow for the clip's frame rate.
        unsupported = ~fit.supported & ~signal.blocked

        notes: list[str] = []
        required = self._config.required_observations
        if fit.window_samples_max < required:
            notes.append(
                f"A {self._config.window_s:g} s window covers at most "
                f"{fit.window_samples_max} samples on this clip, fewer than the {required} a "
                f"degree-{self._config.polyorder} fit needs, so nothing could be fitted "
                f"anywhere. {self._remedy(signal, required)}"
            )

        report = StageReport(
            stage=self.name,
            counts={
                "samples": len(signal),
                "observed": int(np.count_nonzero(observed)),
                "valid": int(np.count_nonzero(fit.supported)),
                "blocked": int(np.count_nonzero(signal.blocked)),
                "unsupported": int(np.count_nonzero(unsupported)),
                "window_samples_max": fit.window_samples_max,
                "window_samples_min": int(fit.observations.min()) if len(signal) else 0,
            },
            measurements={
                "window_s": self._config.window_s,
                "polyorder": float(self._config.polyorder),
                "residual_rms": fit.residual_rms,
            },
            notes=notes,
        )
        return signal.with_derivatives(fit.value, fit.velocity, fit.acceleration), report
