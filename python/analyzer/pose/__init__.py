"""Pose estimation: video frames in, landmarks out.

`PoseEstimator` is the seam every later phase depends on. MediaPipe is one
implementation behind it and its types stop at the adapter, so the model can be
replaced -- by another variant, by something learned in Phase 12, or by a fake
in a test -- without touching any consumer.

The two coordinate spaces are kept apart on purpose. See `contracts/pose.py`:
`HIP_LOCAL` is MediaPipe's "world" output, which is hip-centred and only roughly
metric, and is not calibrated world geometry. Real world coordinates arrive in
Phase 9.
"""

from __future__ import annotations

from analyzer.pose.estimator import (
    PoseEstimationError,
    PoseEstimator,
    resolve_model,
)
from analyzer.pose.extract import (
    extract_and_store,
    extract_poses,
    output_path_for,
)
from analyzer.pose.series import LandmarkSeries, all_series, landmark_series
from analyzer.pose.store import PoseStoreError, read_sequence, write_sequence

__all__ = [
    "LandmarkSeries",
    "PoseEstimationError",
    "PoseEstimator",
    "PoseStoreError",
    "all_series",
    "extract_and_store",
    "extract_poses",
    "landmark_series",
    "output_path_for",
    "read_sequence",
    "resolve_model",
    "write_sequence",
]
