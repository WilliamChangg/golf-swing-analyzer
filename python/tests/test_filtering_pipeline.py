"""Pipeline composition, and the guarantee it refuses to take on trust.

The pipeline's own contribution is small -- run stages in order, collect
reports -- with one exception that carries the whole layer: a gap the policy
refused must be empty on the way out. That is checked rather than assumed,
because a stage added later that smooths across a blocked span would otherwise
break it silently and produce fabricated positions indistinguishable from
measured ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.contracts.filtering import FilterConfig, GapPolicy, SmoothingConfig, StageReport
from analyzer.filtering.pipeline import (
    FilterPipeline,
    FilterPipelineError,
    FilterStage,
    default_pipeline,
)
from analyzer.filtering.signal import Signal, signal_from_arrays

FPS = 120.0


def _signal(count: int = 60) -> Signal:
    t = np.arange(count, dtype=np.float64) / FPS
    return signal_from_arrays(t, np.sin(2 * np.pi * 2.0 * t))


class _Recorder:
    """A stage that changes nothing and records that it ran."""

    def __init__(self, name: str, log: list[str]) -> None:
        self._name = name
        self._log = log

    @property
    def name(self) -> str:
        return self._name

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        self._log.append(self._name)
        return signal, StageReport(stage=self._name)


class _Truncating:
    """A stage that returns the wrong number of samples."""

    @property
    def name(self) -> str:
        return "truncating"

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        half = len(signal) // 2
        return signal_from_arrays(signal.t[:half], signal.value[:half]), StageReport(
            stage=self.name
        )


class _FillsBlockedGaps:
    """A stage that produces a value where the policy refused one."""

    @property
    def name(self) -> str:
        return "overreaching"

    def apply(self, signal: Signal) -> tuple[Signal, StageReport]:
        blocked = np.zeros(len(signal), dtype=np.bool_)
        blocked[10:20] = True
        filled = np.nan_to_num(signal.value, nan=0.5)
        return (
            Signal(
                t=signal.t,
                value=filled,
                visibility=signal.visibility,
                presence=signal.presence,
                weight=signal.weight,
                filled=signal.filled,
                blocked=blocked,
            ),
            StageReport(stage=self.name),
        )


class TestComposition:
    def test_runs_stages_in_order(self) -> None:
        log: list[str] = []
        pipeline = FilterPipeline(
            [_Recorder("first", log), _Recorder("second", log), _Recorder("third", log)]
        )
        _, reports = pipeline.run(_signal())

        assert log == ["first", "second", "third"]
        assert [report.stage for report in reports] == ["first", "second", "third"]

    def test_exposes_its_stage_names(self) -> None:
        assert default_pipeline(FilterConfig()).names == (
            "confidence_gate",
            "gap_policy",
            "local_polynomial",
        )

    def test_rejects_an_empty_pipeline(self) -> None:
        with pytest.raises(FilterPipelineError, match="at least one stage"):
            FilterPipeline([])

    def test_rejects_a_stage_that_changes_the_timeline(self) -> None:
        """A stage may change values; the frames are not its to redefine.

        Silently shortening the signal would break the correspondence between
        sample index and video frame, which everything downstream relies on.
        """
        pipeline = FilterPipeline([_Truncating()])
        with pytest.raises(FilterPipelineError, match="A stage may change values"):
            pipeline.run(_signal())

    def test_the_default_stages_satisfy_the_protocol(self) -> None:
        for stage in default_pipeline(FilterConfig()).stages:
            assert isinstance(stage, FilterStage)


class TestBlockedGuarantee:
    def test_raises_when_a_stage_fills_a_refused_gap(self) -> None:
        pipeline = FilterPipeline([_FillsBlockedGaps()])
        with pytest.raises(FilterPipelineError, match="must stay NaN"):
            pipeline.run(_signal())

    def test_holds_for_the_real_pipeline_on_a_long_absence(self) -> None:
        count = 240
        t = np.arange(count, dtype=np.float64) / FPS
        values = np.sin(2 * np.pi * 2.0 * t)
        values[80:140] = np.nan  # half a second, far past the policy

        signal = signal_from_arrays(t, values)
        output, reports = default_pipeline(FilterConfig()).run(signal)

        assert np.all(np.isnan(output.value[80:140]))
        assert output.velocity is not None
        assert np.all(np.isnan(output.velocity[80:140]))
        assert reports[1].counts["gaps_refused"] == 1

    def test_a_short_absence_is_bridged_end_to_end(self) -> None:
        count = 240
        t = np.arange(count, dtype=np.float64) / FPS
        values = np.sin(2 * np.pi * 2.0 * t)
        values[100:103] = np.nan  # 4 intervals at 120 fps = 33 ms

        signal = signal_from_arrays(t, values)
        output, reports = default_pipeline(FilterConfig()).run(signal)

        assert reports[1].counts["gaps_bridged"] == 1
        assert np.all(np.isfinite(output.value[100:103]))
        # And the fitted values are close to the truth they interpolated.
        assert np.allclose(output.value[100:103], values[100:103], atol=1e-3, equal_nan=True) or (
            np.max(np.abs(output.value[100:103] - np.sin(2 * np.pi * 2.0 * t[100:103]))) < 5e-3
        )


class TestOrdering:
    def test_gating_runs_before_the_gap_policy(self) -> None:
        """Otherwise a rejected detection is not yet an absence, and the gap it
        belongs to is measured shorter than it is."""
        count = 120
        t = np.arange(count, dtype=np.float64) / FPS
        values = np.sin(2 * np.pi * t)
        visibility = np.full(count, 0.9)
        presence = np.ones(count)
        # A run of low-confidence detections long enough to exceed the policy.
        visibility[40:60] = 0.1

        signal = signal_from_arrays(t, values, visibility=visibility, presence=presence)
        output, reports = default_pipeline(FilterConfig()).run(signal)

        assert reports[0].counts["gated_out"] == 20
        assert reports[1].counts["gaps_refused"] == 1
        assert np.all(np.isnan(output.value[40:60]))

    def test_a_wider_gap_policy_changes_what_the_fit_may_produce(self) -> None:
        count = 120
        t = np.arange(count, dtype=np.float64) / FPS
        values = np.sin(2 * np.pi * t)
        values[50:56] = np.nan  # 58 ms, past the 50 ms default

        strict = default_pipeline(FilterConfig())
        lenient = default_pipeline(
            FilterConfig(gaps=GapPolicy(max_gap_s=0.20), smoothing=SmoothingConfig())
        )

        signal = signal_from_arrays(t, values)
        assert np.all(np.isnan(strict.run(signal)[0].value[50:56]))
        assert np.all(np.isfinite(lenient.run(signal)[0].value[50:56]))
