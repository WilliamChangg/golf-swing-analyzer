"""Confidence gating and gap policy.

Both stages exist to turn a guess into an absence, so every test here is asking
the same question from a different direction: does something the model made up
survive into the output, and is the system able to say how much of the result
rests on nothing?
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.contracts.filtering import ConfidenceGate, GapPolicy
from analyzer.filtering.gaps import GapPolicyStage, missing_runs
from analyzer.filtering.gating import ConfidenceGateStage
from analyzer.filtering.signal import signal_from_arrays

FPS = 30.0


def _signal(
    values: list[float],
    *,
    visibility: list[float] | None = None,
    presence: list[float] | None = None,
):
    t = np.arange(len(values), dtype=np.float64) / FPS
    return signal_from_arrays(
        t,
        np.asarray(values, dtype=np.float64),
        visibility=np.asarray(visibility, dtype=np.float64) if visibility else None,
        presence=np.asarray(presence, dtype=np.float64) if presence else None,
    )


class TestConfidenceGate:
    def test_rejects_detections_below_the_visibility_threshold(self) -> None:
        signal = _signal([1.0, 2.0, 3.0], visibility=[0.9, 0.2, 0.9], presence=[1.0, 1.0, 1.0])
        out, report = ConfidenceGateStage(ConfidenceGate(min_visibility=0.5)).apply(signal)

        assert np.isnan(out.value[1])
        assert out.weight[1] == 0.0
        assert report.counts["gated_out"] == 1
        assert report.counts["low_visibility"] == 1

    def test_rejects_detections_below_the_presence_threshold(self) -> None:
        signal = _signal([1.0, 2.0, 3.0], visibility=[0.9, 0.9, 0.9], presence=[1.0, 0.1, 1.0])
        out, report = ConfidenceGateStage(ConfidenceGate(min_presence=0.5)).apply(signal)

        assert np.isnan(out.value[1])
        assert report.counts["low_presence"] == 1

    def test_the_two_thresholds_are_independent(self) -> None:
        """Kept separate because presence and visibility mean different things:
        in frame is not the same question as unoccluded."""
        signal = _signal([1.0, 2.0], visibility=[0.9, 0.9], presence=[0.6, 0.6])
        gate = ConfidenceGate(min_visibility=0.95, min_presence=0.5)
        _, report = ConfidenceGateStage(gate).apply(signal)
        assert report.counts["low_visibility"] == 2
        assert report.counts["low_presence"] == 0

    def test_an_unknown_confidence_is_not_treated_as_acceptable(self) -> None:
        """NaN confidence on a present value gates out rather than through.

        The natural spelling, `x < threshold`, is False for NaN and would let it
        pass. Unknown is not evidence of visibility.
        """
        t = np.arange(2, dtype=np.float64)
        signal = signal_from_arrays(
            t,
            np.array([1.0, 2.0]),
            visibility=np.array([np.nan, 0.9]),
            presence=np.array([1.0, 1.0]),
        )
        out, report = ConfidenceGateStage(ConfidenceGate()).apply(signal)
        assert np.isnan(out.value[0])
        assert report.counts["gated_out"] == 1

    def test_counts_frames_that_never_had_a_detection_separately(self) -> None:
        """A rejected detection and no detection need different responses."""
        signal = _signal(
            [1.0, float("nan"), 3.0], visibility=[0.1, 0.0, 0.9], presence=[1.0, 0.0, 1.0]
        )
        _, report = ConfidenceGateStage(ConfidenceGate()).apply(signal)
        assert report.counts["never_detected"] == 1
        assert report.counts["gated_out"] == 1
        assert report.counts["observed"] == 1

    def test_passes_everything_through_when_nothing_is_below_threshold(self) -> None:
        signal = _signal([1.0, 2.0, 3.0])
        out, report = ConfidenceGateStage(ConfidenceGate()).apply(signal)
        assert np.array_equal(out.value, signal.value)
        assert report.counts["gated_out"] == 0

    def test_drops_derivatives_because_the_values_changed(self) -> None:
        signal = _signal([1.0, 2.0, 3.0], visibility=[0.9, 0.1, 0.9], presence=[1.0, 1.0, 1.0])
        with_derivatives = signal.with_derivatives(signal.value, np.ones(3), np.ones(3))
        out, _ = ConfidenceGateStage(ConfidenceGate()).apply(with_derivatives)
        assert out.velocity is None


class TestMissingRuns:
    def test_finds_maximal_runs(self) -> None:
        mask = np.array([False, True, True, False, True, False])
        assert missing_runs(mask) == [(1, 3), (4, 5)]

    def test_handles_runs_at_both_ends(self) -> None:
        mask = np.array([True, False, True])
        assert missing_runs(mask) == [(0, 1), (2, 3)]

    def test_no_runs_in_a_complete_signal(self) -> None:
        assert missing_runs(np.zeros(5, dtype=bool)) == []

    def test_an_empty_mask_has_no_runs(self) -> None:
        assert missing_runs(np.zeros(0, dtype=bool)) == []


class TestGapPolicy:
    def test_bridges_a_gap_shorter_than_the_limit(self) -> None:
        signal = _signal([0.0, 1.0, float("nan"), 3.0, 4.0])
        stage = GapPolicyStage(GapPolicy(max_gap_s=0.10))
        out, report = stage.apply(signal)

        # Anchors at 1.0 and 3.0, so the midpoint interpolates to 2.0.
        assert out.value[2] == pytest.approx(2.0)
        assert out.filled[2]
        assert not out.blocked[2]
        assert report.counts["gaps_bridged"] == 1

    def test_interpolation_is_linear_in_time_not_in_index(self) -> None:
        """On a variable-rate clip the two differ, and index would be wrong."""
        t = np.array([0.0, 0.1, 0.4, 0.5])
        signal = signal_from_arrays(t, np.array([0.0, 10.0, np.nan, 40.0]))
        out, _ = GapPolicyStage(GapPolicy(max_gap_s=1.0)).apply(signal)
        # t=0.4 is 3/4 of the way from 0.1 to 0.5, so 10 + 0.75 * 30 = 32.5.
        assert out.value[2] == pytest.approx(32.5)

    def test_refuses_a_gap_longer_than_the_limit(self) -> None:
        values = [0.0, 1.0, *([float("nan")] * 6), 8.0]
        signal = _signal(values)
        out, report = GapPolicyStage(GapPolicy(max_gap_s=0.10)).apply(signal)

        assert np.all(np.isnan(out.value[2:8]))
        assert np.all(out.blocked[2:8])
        assert not np.any(out.filled)
        assert report.counts["gaps_refused"] == 1
        assert report.measurements["longest_gap_s"] == pytest.approx(7 / FPS)

    def test_blocks_a_leading_absence_whatever_its_length(self) -> None:
        """Nothing on the early side constrains it, so bridging would extrapolate."""
        signal = _signal([float("nan"), 1.0, 2.0, 3.0])
        out, report = GapPolicyStage(GapPolicy(max_gap_s=10.0)).apply(signal)
        assert out.blocked[0]
        assert np.isnan(out.value[0])
        assert report.counts["gaps_unanchored"] == 1

    def test_blocks_a_trailing_absence_whatever_its_length(self) -> None:
        signal = _signal([1.0, 2.0, 3.0, float("nan")])
        out, report = GapPolicyStage(GapPolicy(max_gap_s=10.0)).apply(signal)
        assert out.blocked[3]
        assert report.counts["gaps_unanchored"] == 1

    def test_measures_the_gap_between_bounding_observations(self) -> None:
        """One missing frame is a two-interval absence, not a one-interval one.

        The quantity that matters is how long the subject went unobserved, which
        is the elapsed time from the last real sample to the next one.
        """
        signal = _signal([0.0, 1.0, float("nan"), 3.0])
        _, report = GapPolicyStage(GapPolicy(max_gap_s=1.0)).apply(signal)
        assert report.measurements["longest_gap_s"] == pytest.approx(2 / FPS)

    def test_filled_samples_carry_no_weight_by_default(self) -> None:
        """An interpolated value is a function of neighbours already in the
        fitting window; weighting it would count them twice."""
        signal = _signal([0.0, 1.0, float("nan"), 3.0])
        out, _ = GapPolicyStage(GapPolicy(max_gap_s=1.0)).apply(signal)
        assert out.weight[2] == 0.0
        assert out.filled[2]

    def test_filled_weight_can_be_raised_deliberately(self) -> None:
        signal = _signal([0.0, 1.0, float("nan"), 3.0])
        out, _ = GapPolicyStage(GapPolicy(max_gap_s=1.0, filled_weight=0.5)).apply(signal)
        assert out.weight[2] == pytest.approx(0.5)

    def test_the_limit_separates_bridged_from_refused(self) -> None:
        """Probed either side of the threshold rather than exactly on it.

        The gap here spans two frame intervals, and `2 / FPS` computed two
        different ways differs in the last bit -- so an assertion sitting exactly
        on the boundary would be testing floating-point arithmetic rather than
        the policy. No epsilon is applied in the stage either: a gap within a
        nanosecond of the limit is physically a coin flip, and inventing a
        tolerance would imply a precision the timestamps do not carry.
        """
        signal = _signal([0.0, 1.0, float("nan"), 3.0])
        span = 2 / FPS

        inside, _ = GapPolicyStage(GapPolicy(max_gap_s=span * 1.01)).apply(signal)
        outside, _ = GapPolicyStage(GapPolicy(max_gap_s=span * 0.99)).apply(signal)

        assert inside.filled[2]
        assert outside.blocked[2]

    def test_reports_nothing_to_do_on_a_complete_signal(self) -> None:
        signal = _signal([0.0, 1.0, 2.0])
        out, report = GapPolicyStage(GapPolicy()).apply(signal)
        assert report.counts["gaps"] == 0
        assert report.measurements["longest_gap_s"] == 0.0
        assert np.array_equal(out.value, signal.value)
