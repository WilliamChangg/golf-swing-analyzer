"""Installation failures must preserve usable weights; CPU claims need execution."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from analyzer.contracts.models import ModelInventory
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.dispatch import call
from analyzer.environment.hardware import compute_info, select_torch_device
from analyzer.environment.model_manager import ModelInstallError, download_artifact
from analyzer.progress import RecordingReporter
from tests.test_models import CONTENT, _entry


class Response(io.BytesIO):
    def geturl(self) -> str:
        return "https://example.invalid/model.task"


@pytest.fixture
def model_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GSA_MODELS_DIR", str(tmp_path))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_pose_model": "test_model",
                "models": [asdict(_entry())],
            }
        )
    )
    return tmp_path


def test_inventory_download_verify_update_through_rpc(model_directory: Path) -> None:
    before = call("list_models")
    assert isinstance(before, ModelInventory)
    assert before.models[0].state == "missing"
    reporter = RecordingReporter()
    with patch("urllib.request.urlopen", return_value=Response(CONTENT)):
        after = call("install_model", {"name": "test_model"}, reporter)
    assert isinstance(after, ModelInventory)
    assert after.models[0].state == "verified"
    assert after.models[0].backend == "mediapipe_tasks"
    assert after.models[0].device == "cpu"
    assert after.models[0].input_requirements
    assert reporter.events[-1].stage == "verify"
    assert reporter.events[-1].current == 1
    (model_directory / "test_model.task").write_bytes(b"old version")
    assert call("list_models").models[0].state == "mismatch"
    with patch("urllib.request.urlopen", return_value=Response(CONTENT)):
        assert call("install_model", {"name": "test_model"}).models[0].state == "verified"


@pytest.mark.parametrize("payload", [b"short", b"x" * len(CONTENT), CONTENT + b"extra"])
def test_failed_verification_preserves_previous_file(tmp_path: Path, payload: bytes) -> None:
    target = tmp_path / "model.task"
    target.write_bytes(b"previous model")
    with (
        patch("urllib.request.urlopen", return_value=Response(payload)),
        pytest.raises(ModelInstallError),
    ):
        download_artifact(_entry(), target)
    assert target.read_bytes() == b"previous model"
    assert list(tmp_path.glob("*.part")) == []


def test_interrupted_download_cleans_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "model.task"
    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("offline")),
        pytest.raises(TimeoutError),
    ):
        download_artifact(_entry(), target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_install_refuses_arbitrary_url_or_unknown_name(model_directory: Path) -> None:
    with pytest.raises(EngineError) as error:
        call("install_model", {"name": "test_model", "url": "https://example.com"})
    assert error.value.code == ErrorCode.INVALID_PARAMS
    with pytest.raises(EngineError, match="Unknown model"):
        call("install_model", {"name": "../../elsewhere"})


@pytest.mark.parametrize("requested", ["auto", "cuda", "mps", "cpu"])
def test_force_cpu_overrides_available_accelerators(monkeypatch, requested: str) -> None:
    monkeypatch.setenv("GSA_FORCE_CPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert compute_info().selected_device == "cpu"
    selected, reason = select_torch_device(requested)
    assert selected == "cpu"
    assert reason and "forced" in reason
    # Run the actual temporal network on the resolved device, not just a string assertion.
    from analyzer.contracts.ml import TCNConfig
    from analyzer.ml.tcn import SwingTCN

    model = SwingTCN(10, TCNConfig(hidden_channels=8, layers=2)).to(selected)
    result = model(torch.zeros(1, 10, 32, device=selected))
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()


def test_unavailable_accelerator_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("GSA_FORCE_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    selected, reason = select_torch_device("cuda")
    assert selected == "cpu"
    assert reason and "unavailable" in reason


def test_accelerator_runtime_failure_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("GSA_FORCE_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with patch("torch.ones", side_effect=RuntimeError("driver unavailable")):
        selected, reason = select_torch_device("cuda")
    assert selected == "cpu"
    assert reason and "driver unavailable" in reason
