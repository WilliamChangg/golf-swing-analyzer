"""Phase 18: measure inventory verification and forced-CPU inference on a real clip.

Usage: python scripts/benchmark_models.py data/amateur/face-on/PW_face-on.mp4
Runs fresh inference in a temporary directory; never reuses the pose cache.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    os.environ["GSA_FORCE_CPU"] = "1"

    import torch
    from analyzer.contracts.ml import TCNConfig
    from analyzer.environment.hardware import select_torch_device
    from analyzer.environment.model_manager import inventory
    from analyzer.ml.tcn import SwingTCN
    from analyzer.pose.extract import extract_and_store
    from analyzer.pose.mediapipe_estimator import MediaPipePoseEstimator

    started = time.perf_counter()
    models = inventory()
    verification_ms = (time.perf_counter() - started) * 1000
    device, reason = select_torch_device("auto")
    temporal = SwingTCN(10, TCNConfig(hidden_channels=8, layers=2)).to(device).eval()
    with torch.no_grad():
        output = temporal(torch.zeros(1, 10, 32, device=device))
    with (
        tempfile.TemporaryDirectory() as directory,
        MediaPipePoseEstimator() as estimator,
    ):
        poses = extract_and_store(
            args.video, estimator, output=Path(directory) / "poses.parquet"
        )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "inventory_verify_ms": verification_ms,
                "models": models.model_dump(mode="json"),
                "forced_cpu": True,
                "temporal_device": str(output.device),
                "temporal_output_finite": bool(torch.isfinite(output).all()),
                "selection_reason": reason,
                "pose_delegate": poses.model.delegate,
                "pose_stats": poses.stats.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
