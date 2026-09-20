"""Performance measurement and cache primitives.

This package has no domain logic.  It measures the work the engine already
does, and keys reusable derived results on the two facts that make them valid:
the input bytes and the complete analysis configuration.
"""

from analyzer.performance.cache import AnalysisCache, config_digest
from analyzer.performance.telemetry import PerformanceRecorder, recording, stage

__all__ = [
    "AnalysisCache",
    "PerformanceRecorder",
    "config_digest",
    "recording",
    "stage",
]
