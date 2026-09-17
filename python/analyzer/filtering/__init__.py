"""Temporal filtering: from detections to trajectories with derivatives.

The numerical core of the pipeline. Pose estimation produces a landmark position
per frame, independently, with no notion that the frames form a motion; this
layer turns that into a trajectory that can be differentiated, and decides which
parts of it are supported by enough evidence to use.

Three things here are load-bearing:

**The fit runs on real timestamps.** Savitzky-Golay assumes uniform sampling and
phone footage frequently is not uniform, so the fit is solved locally at each
sample rather than applied as a fixed kernel. Savitzky-Golay is the uniform-grid
special case, and the test suite pins that equivalence against SciPy.

**Derivatives come from the fit.** Velocity and acceleration are coefficients of
the local polynomial, not finite differences of smoothed positions, so they are
consistent with the position and do not carry a second unstated filter's noise
response.

**Absence survives.** A low-confidence detection becomes NaN, a gap longer than
the policy allows stays NaN through every stage, and the pipeline verifies that
on the way out rather than trusting its stages to have honoured it.
"""

from analyzer.filtering.gaps import GapPolicyStage, missing_runs
from analyzer.filtering.gating import ConfidenceGateStage
from analyzer.filtering.landmarks import (
    FilteredLandmark,
    FilteredSequence,
    filter_landmark,
    filter_sequence,
    signals_from_series,
)
from analyzer.filtering.localpoly import (
    LocalPolynomialError,
    LocalPolynomialFit,
    local_polynomial_fit,
)
from analyzer.filtering.pipeline import (
    FilterPipeline,
    FilterPipelineError,
    FilterStage,
    default_pipeline,
)
from analyzer.filtering.signal import Signal, SignalError, signal_from_arrays
from analyzer.filtering.smoothing import LocalPolynomialStage

__all__ = [
    "ConfidenceGateStage",
    "FilterPipeline",
    "FilterPipelineError",
    "FilterStage",
    "FilteredLandmark",
    "FilteredSequence",
    "GapPolicyStage",
    "LocalPolynomialError",
    "LocalPolynomialFit",
    "LocalPolynomialStage",
    "Signal",
    "SignalError",
    "default_pipeline",
    "filter_landmark",
    "filter_sequence",
    "local_polynomial_fit",
    "missing_runs",
    "signal_from_arrays",
    "signals_from_series",
]
