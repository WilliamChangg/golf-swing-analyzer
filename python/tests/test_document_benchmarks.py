"""Documentation must preserve measurements instead of trusting copied summaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "document_benchmarks", ROOT / "scripts/document_benchmarks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table_recomputes_medians_from_samples(generator: ModuleType) -> None:
    record = json.loads(generator.EVIDENCE.read_text())
    workflow = record["runs"]["workflow"]["output"]
    for mode, samples in (("baseline", (0.001, 0.009, 9.0)), ("warm", (0.001, 0.003, 1.0))):
        workflow[mode]["runs"] = [
            [{"operation": "extract_poses", "elapsed_s": value}] for value in samples
        ]
        workflow[mode]["median"] = [{"operation": "extract_poses", "median_elapsed_s": 999}]
    assert "| `extract_poses` | 9.00 | 3.00 | 3.00\N{MULTIPLICATION SIGN} |" in generator.render(
        record
    )


def test_check_rejects_stale_appendix_without_rewriting(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    appendix = tmp_path / "appendix.md"
    appendix.write_text("stale figures\n")
    monkeypatch.setattr(generator, "APPENDIX", appendix)
    monkeypatch.setattr(sys, "argv", ["document_benchmarks.py", "--check"])
    assert generator.main() == 1
    assert appendix.read_text() == "stale figures\n"
    appendix.write_text(generator.render(json.loads(generator.EVIDENCE.read_text())))
    assert generator.main() == 0


def test_failed_capture_preserves_previous_evidence(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("previous verified evidence\n")
    appendix = tmp_path / "appendix.md"
    appendix.write_text("previous appendix\n")
    video = tmp_path / "clip.mp4"
    video.touch()
    monkeypatch.setattr(generator, "EVIDENCE", evidence)
    monkeypatch.setattr(generator, "APPENDIX", appendix)
    monkeypatch.setattr(sys, "argv", ["document_benchmarks.py", "--capture", "--video", str(video)])
    original = generator.output

    def fail_benchmark(argv: list[str]) -> str:
        if "scripts/benchmark.py" in argv:
            raise subprocess.CalledProcessError(1, argv)
        return original(argv)

    monkeypatch.setattr(generator, "output", fail_benchmark)
    with pytest.raises(subprocess.CalledProcessError):
        generator.main()
    assert evidence.read_text() == "previous verified evidence\n"
    assert appendix.read_text() == "previous appendix\n"
