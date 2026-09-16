"""Tests for the probe that verifies pose inference actually runs.

This probe exists because Phase 0 shipped a green health check for a pipeline
that could not process a single frame: `import mediapipe` succeeded while the
pose graph aborted the process on first use. So the tests that matter most here
are the ones asserting it reports *failure* correctly -- a probe that can only
say "ok" is the thing being replaced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from analyzer.contracts.health import HealthStatus
from analyzer.environment import pose_smoke
from analyzer.environment.pose_runtime import COMPONENT_NAME, probe_pose_runtime
from tests.conftest import requires_pose_model


@pytest.fixture
def not_a_model(tmp_path: Path) -> Path:
    target = tmp_path / "broken.task"
    target.write_bytes(b"this is not a tflite bundle")
    return target


class TestReportsFailure:
    def test_a_missing_model_is_missing_not_error(self, tmp_path: Path) -> None:
        """Nothing is broken; the claim simply went unverified."""
        status = probe_pose_runtime(tmp_path / "absent.task")
        assert status.status is HealthStatus.MISSING
        assert status.remediation is not None

    def test_a_corrupt_model_blames_the_model_not_the_build(self, not_a_model: Path) -> None:
        """A catchable failure means the interpreter survived, so the build is fine."""
        status = probe_pose_runtime(not_a_model)
        assert status.status is HealthStatus.ERROR
        assert "corrupt" in (status.remediation or "")
        assert "mediapipe==" not in (status.remediation or "")

    def test_a_corrupt_model_reports_what_mediapipe_said(self, not_a_model: Path) -> None:
        status = probe_pose_runtime(not_a_model)
        assert "Error" in status.detail or "error" in status.detail

    def test_a_process_killed_by_a_signal_blames_the_build(
        self, monkeypatch: pytest.MonkeyPatch, not_a_model: Path
    ) -> None:
        """An abort is what a MediaPipe that cannot open its graph looks like.

        Simulated, because the whole point of pinning 1.0.0 is that the real
        abort no longer happens here -- and a guard that only works on a broken
        install is a guard nobody can test.
        """

        def killed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=-6, stdout="", stderr="Check failed: service_"
            )

        monkeypatch.setattr(subprocess, "run", killed)
        status = probe_pose_runtime(not_a_model)

        assert status.status is HealthStatus.ERROR
        assert "signal 6" in status.detail
        assert "mediapipe==1.0.0" in (status.remediation or "")

    def test_a_timeout_is_an_error_not_a_hang(
        self, monkeypatch: pytest.MonkeyPatch, not_a_model: Path
    ) -> None:
        """A health check that hangs is worse than one that reports a failure."""

        def times_out(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="pose_smoke", timeout=60)

        monkeypatch.setattr(subprocess, "run", times_out)
        status = probe_pose_runtime(not_a_model)

        assert status.status is HealthStatus.ERROR
        assert "did not finish" in status.detail

    def test_reports_the_last_line_of_output_not_the_first(
        self, monkeypatch: pytest.MonkeyPatch, not_a_model: Path
    ) -> None:
        """A child that dies noisily prints warnings before what killed it."""

        def noisy(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="RuntimeWarning: ignore me\nthe real cause"
            )

        monkeypatch.setattr(subprocess, "run", noisy)
        assert "the real cause" in probe_pose_runtime(not_a_model).detail

    def test_a_zero_exit_without_the_sentinel_is_not_success(
        self, monkeypatch: pytest.MonkeyPatch, not_a_model: Path
    ) -> None:
        """A process that aborted after doing the work must not read as having done it."""

        def silent(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", silent)
        assert probe_pose_runtime(not_a_model).status is HealthStatus.ERROR


@requires_pose_model
class TestReportsSuccess:
    def test_running_inference_reports_ok(self) -> None:
        status = probe_pose_runtime()
        assert status.name == COMPONENT_NAME
        assert status.status is HealthStatus.OK

    def test_success_states_the_measured_time(self) -> None:
        """The claim is that inference ran, so the evidence is how long it took."""
        assert "ms" in probe_pose_runtime().detail

    def test_the_smoke_test_exits_cleanly_on_a_real_model(self) -> None:
        from analyzer.environment.pose_runtime import _smallest_available_model

        model = _smallest_available_model()
        assert model is not None
        assert pose_smoke.main([str(model)]) == pose_smoke.EXIT_OK


class TestSmokeTestExitCodes:
    def test_wrong_argument_count(self) -> None:
        assert pose_smoke.main([]) == pose_smoke.EXIT_USAGE

    def test_missing_model_file(self, tmp_path: Path) -> None:
        assert pose_smoke.main([str(tmp_path / "absent.task")]) == pose_smoke.EXIT_MODEL_MISSING

    def test_unusable_model_file(self, not_a_model: Path) -> None:
        assert pose_smoke.main([str(not_a_model)]) == pose_smoke.EXIT_INFERENCE_FAILED
