"""Phase 17 performance records and result-cache tests."""

from __future__ import annotations

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.health import ComponentStatus, HealthStatus
from analyzer.performance.cache import AnalysisCache, config_digest
from analyzer.performance.telemetry import PerformanceRecorder, recording, stage


def _key(digest: str = "a" * 64) -> ContentKey:
    return ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest=digest, size_bytes=123)


def test_config_digest_is_independent_of_mapping_order() -> None:
    assert config_digest({"filter": {"window": 0.1}, "model": "full"}) == config_digest(
        {"model": "full", "filter": {"window": 0.1}}
    )


def test_result_cache_requires_matching_content_and_configuration(tmp_path) -> None:
    cache = AnalysisCache(tmp_path)
    result = ComponentStatus(name="ffmpeg", status=HealthStatus.OK, detail="ready")
    config = {"window_s": 0.1}
    cache.store(_key(), "metrics", config, result)

    assert cache.load(_key(), "metrics", config, ComponentStatus) == result
    assert cache.load(_key(), "metrics", {"window_s": 0.2}, ComponentStatus) is None
    assert cache.load(_key("b" * 64), "metrics", config, ComponentStatus) is None


def test_nested_stage_is_collected_only_while_a_recorder_is_active() -> None:
    recorder = PerformanceRecorder("test")
    with recording(recorder), stage("outer"), stage("inner"):
        pass

    assert [entry.name for entry in recorder.stages] == ["inner", "outer"]
    assert all(entry.elapsed_s >= 0 for entry in recorder.stages)
    assert recorder.as_dict()["operation"] == "test"
