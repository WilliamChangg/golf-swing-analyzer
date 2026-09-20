"""Exercise persisted data through public dispatch boundaries, including misses.

Synthetic poses test downstream behavior, not pose-estimator accuracy. Real
committed clips separately test metadata, decoding, and extraction together.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from analyzer import dispatch
from analyzer.contracts.coaching import CoachingReport
from analyzer.contracts.metrics import MetricSet, MetricUnit
from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.pose import LANDMARK_COUNT
from analyzer.ingestion import probe_video
from analyzer.performance.cache import AnalysisCache
from analyzer.pose.extract import extract_and_store
from analyzer.pose.store import read_sequence, write_sequence
from tests.conftest import CFR_30FPS, ROTATED_CCW90, VFR_30_TO_15FPS, FakeEstimator, requires_ffmpeg
from tests.synthetic import hand_path, pose_sequence, swing_sequence
from tests.synthetic_signals import irregular_clock


@pytest.fixture
def swing_file(tmp_path: Path) -> Path:
    path = tmp_path / "swing.parquet"
    write_sequence(swing_sequence(), path, landmark_count=LANDMARK_COUNT)
    return path


@pytest.mark.parametrize("method", ["compute_metrics", "coach_swing"])
@pytest.mark.parametrize("payload", [b"[]", b"null", b"true", b"42", b'"text"', b"{", b"\xff"])
def test_corrupt_report_is_recomputed(swing_file: Path, method: str, payload: bytes) -> None:
    params = {"path": str(swing_file)}
    first = dispatch.call(method, params)
    from analyzer.paths import cache_dir

    cached = list((cache_dir() / "analysis-results").rglob("*.json"))
    assert len(cached) == 1
    cached[0].write_bytes(payload)
    second = dispatch.call(method, params)
    assert second == first
    assert isinstance(json.loads(cached[0].read_text()), dict)


@pytest.mark.parametrize("method", ["compute_metrics", "coach_swing"])
def test_warm_report_reuses_work_but_configuration_changes_recompute(
    swing_file: Path, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = dispatch._metrics_chain

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch, "_metrics_chain", counted)
    params = {"path": str(swing_file)}
    first = dispatch.call(method, params)
    assert dispatch.call(method, params) == first
    assert len(calls) == 1
    dispatch.call(method, {**params, "filter": {"smoothing": {"window_s": 0.15}}})
    assert len(calls) == 2

    sequence = read_sequence(swing_file)
    sequence.extracted_at += timedelta(seconds=1)
    write_sequence(sequence, swing_file, landmark_count=LANDMARK_COUNT)
    dispatch.call(method, params)
    assert len(calls) == 3
    sequence.model.sha256 = "b" * 64
    write_sequence(sequence, swing_file, landmark_count=LANDMARK_COUNT)
    dispatch.call(method, params)
    assert len(calls) == 4


def test_slow_motion_change_recomputes_durations(swing_file: Path) -> None:
    ordinary = dispatch.call("compute_metrics", {"path": str(swing_file)})
    slow = dispatch.call(
        "compute_metrics",
        {
            "path": str(swing_file),
            "slow_motion_factor": 2.0,
            "filter": {"smoothing": {"window_s": 0.05}},
        },
    )
    assert isinstance(ordinary, MetricSet) and isinstance(slow, MetricSet)
    durations = {m.name: m.value for m in ordinary.metrics if m.unit == MetricUnit.SECONDS}
    assert durations
    assert {m.name for m in slow.metrics if m.unit == MetricUnit.SECONDS} == set(durations)
    for metric in slow.metrics:
        if metric.name in durations:
            assert metric.value == pytest.approx(durations[metric.name] / 2)


@pytest.mark.parametrize("condition", ["missing", "low_visibility", "low_presence"])
def test_unusable_landmarks_never_produce_metrics_or_findings(
    tmp_path: Path, condition: str
) -> None:
    clock = irregular_clock()
    x, y = hand_path(clock)
    sequence = pose_sequence(
        x,
        y,
        clock,
        detected=np.zeros(clock.size, dtype=bool) if condition == "missing" else None,
        visibility=np.zeros(clock.size) if condition == "low_visibility" else None,
    )
    if condition == "low_presence":
        for frame in sequence.frames:
            for point in [*frame.image, *frame.hip_local]:
                point.presence = 0.0
    path = tmp_path / "unusable.parquet"
    write_sequence(sequence, path, landmark_count=LANDMARK_COUNT)
    phases = dispatch.call("detect_phases", {"path": str(path)})
    metrics = dispatch.call("compute_metrics", {"path": str(path)})
    coaching = dispatch.call("coach_swing", {"path": str(path)})
    assert isinstance(phases, SwingPhases) and not phases.detected and not phases.events
    assert isinstance(metrics, MetricSet) and not metrics.computed and not metrics.metrics
    assert isinstance(coaching, CoachingReport) and not coaching.findings
    assert phases.warnings and metrics.warnings


@requires_ffmpeg
@pytest.mark.parametrize("clip", [CFR_30FPS, VFR_30_TO_15FPS, ROTATED_CCW90])
def test_metadata_decode_pose_store_and_phases_share_frame_identity(
    tmp_path: Path, clip: Path
) -> None:
    metadata = probe_video(clip)
    index = dispatch.call("seek_index", {"path": str(clip)})
    estimator = FakeEstimator(detect_every=3)
    output = tmp_path / "poses.parquet"
    extract_and_store(clip, estimator, output=output)
    sequence = read_sequence(output)
    assert sequence.video_content_key == metadata.content_key
    assert (sequence.geometry.width, sequence.geometry.height) == (
        metadata.stream.display_width,
        metadata.stream.display_height,
    )
    assert len(sequence.frames) == metadata.timing.frame_count
    assert [f.timestamp_s for f in sequence.frames] == pytest.approx(index.timestamps_s)
    assert [f.frame_index for f in sequence.frames if f.detected] == list(
        range(0, len(sequence.frames), 3)
    )
    phases = dispatch.call("detect_phases", {"path": str(output)})
    assert isinstance(phases, SwingPhases) and not phases.detected


@requires_ffmpeg
def test_renamed_video_reuses_metadata_with_the_current_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original.mp4"
    original.write_bytes(CFR_30FPS.read_bytes())
    first = probe_video(original)
    renamed = original.rename(tmp_path / "renamed.mp4")

    def unexpected_probe(*args, **kwargs):
        pytest.fail("renaming identical bytes should reuse metadata")

    monkeypatch.setattr("analyzer.ingestion.probe", unexpected_probe)
    second = probe_video(renamed)
    assert second.content_key == first.content_key
    assert second.path == str(renamed.resolve())


def test_disabled_cache_neither_reads_nor_writes(
    swing_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = read_sequence(swing_file)
    cache = AnalysisCache(swing_file.parent / "disabled")
    result = dispatch.call("compute_metrics", {"path": str(swing_file)})
    cache.store(sequence.video_content_key, "metrics", {}, result)
    monkeypatch.setenv("GSA_DISABLE_ANALYSIS_CACHE", "1")
    assert cache.load(sequence.video_content_key, "metrics", {}, MetricSet) is None
    cache.store(sequence.video_content_key, "metrics", {"new": True}, result)
    assert len(list((swing_file.parent / "disabled").rglob("*.json"))) == 1


@pytest.mark.parametrize("method", ["compute_metrics", "coach_swing"])
def test_project_analysis_bypasses_report_cache(
    swing_file: Path, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = dispatch.call("create_project", {"name": "Changing calibration"})

    def forbidden_cache(*args, **kwargs):
        pytest.fail("project calibration and sync can change independently of the video")

    monkeypatch.setattr(AnalysisCache, "load", forbidden_cache)
    monkeypatch.setattr(AnalysisCache, "store", forbidden_cache)
    result = dispatch.call(method, {"path": str(swing_file), "project_id": project.id})
    assert isinstance(result, (MetricSet, CoachingReport))


@requires_ffmpeg
@pytest.mark.parametrize("payload", [b"\xff", b"null", b"[]", b"{"])
def test_corrupt_metadata_reprobes_without_changing_frame_identity(payload: bytes) -> None:
    from analyzer.paths import cache_dir

    first = probe_video(VFR_30_TO_15FPS)
    entry = next((cache_dir() / "video-metadata").glob("*.json"))
    entry.write_bytes(payload)
    second = probe_video(VFR_30_TO_15FPS)
    assert second.content_key == first.content_key
    assert second.stream == first.stream
    assert second.timing == first.timing
