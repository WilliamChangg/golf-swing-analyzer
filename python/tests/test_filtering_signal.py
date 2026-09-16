"""Signal construction and its one enforced invariant.

Most of what `Signal` does is hold arrays. The part worth testing is what it
refuses: a timeline that is not a timeline, and derivatives that have outlived
the values they described.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.filtering.signal import Signal, SignalError, signal_from_arrays


def _signal(values: list[float], fps: float = 30.0) -> Signal:
    t = np.arange(len(values), dtype=np.float64) / fps
    return signal_from_arrays(t, np.asarray(values, dtype=np.float64))


class TestConstruction:
    def test_confidence_defaults_to_full_where_a_value_exists(self) -> None:
        """A caller with no confidence data is asserting its samples are observations.

        Defaulting to zero instead would silently gate away every synthetic
        signal, and the failure would look like a filter bug rather than a
        missing default.
        """
        signal = _signal([1.0, 2.0, 3.0])
        assert signal.visibility.tolist() == [1.0, 1.0, 1.0]
        assert signal.weight.tolist() == [1.0, 1.0, 1.0]

    def test_confidence_is_nan_where_the_value_is_missing(self) -> None:
        signal = _signal([1.0, float("nan"), 3.0])
        assert np.isnan(signal.visibility[1])
        assert signal.weight[1] == 0.0

    def test_weight_is_the_weaker_of_visibility_and_presence(self) -> None:
        """Not the product: the two describe related events, and multiplying
        them would assert an independence the model never claimed."""
        t = np.arange(3, dtype=np.float64)
        signal = signal_from_arrays(
            t,
            np.array([1.0, 2.0, 3.0]),
            visibility=np.array([0.9, 0.2, 0.5]),
            presence=np.array([0.4, 0.8, 0.5]),
        )
        assert signal.weight.tolist() == pytest.approx([0.4, 0.2, 0.5])

    def test_missing_samples_get_zero_weight_whatever_confidence_says(self) -> None:
        t = np.arange(2, dtype=np.float64)
        signal = signal_from_arrays(
            t,
            np.array([np.nan, 1.0]),
            visibility=np.array([0.9, 0.9]),
            presence=np.array([0.9, 0.9]),
        )
        assert signal.weight[0] == 0.0

    def test_rejects_mismatched_channel_lengths(self) -> None:
        with pytest.raises(SignalError, match="same instants"):
            Signal(
                t=np.arange(3, dtype=np.float64),
                value=np.zeros(2),
                visibility=np.ones(3),
                presence=np.ones(3),
                weight=np.ones(3),
                filled=np.zeros(3, dtype=np.bool_),
                blocked=np.zeros(3, dtype=np.bool_),
            )

    @pytest.mark.parametrize("times", [[0.0, 0.1, 0.05], [0.0, 0.1, 0.1]])
    def test_rejects_timestamps_that_do_not_strictly_increase(self, times: list[float]) -> None:
        """Repeated or reordered times silently fit the wrong neighbourhood.

        The window is a binary search over t, so a duplicate timestamp does not
        error -- it produces a fit centred somewhere other than where the caller
        believes, which is the kind of wrongness that never surfaces.
        """
        with pytest.raises(SignalError, match="strictly increasing"):
            signal_from_arrays(np.asarray(times), np.zeros(len(times)))

    def test_a_single_sample_is_a_valid_timeline(self) -> None:
        signal = signal_from_arrays(np.array([0.0]), np.array([1.0]))
        assert len(signal) == 1


class TestProvenance:
    def test_observed_excludes_interpolated_samples(self) -> None:
        signal = _signal([1.0, 2.0, 3.0])
        filled = signal.filled.copy()
        filled[1] = True
        updated = signal.with_values(signal.value, filled=filled)
        assert updated.observed.tolist() == [True, False, True]

    def test_observed_excludes_zero_weighted_samples(self) -> None:
        signal = _signal([1.0, 2.0, 3.0])
        weight = signal.weight.copy()
        weight[2] = 0.0
        assert signal.with_values(signal.value, weight=weight).observed.tolist() == [
            True,
            True,
            False,
        ]

    def test_with_values_drops_stale_derivatives(self) -> None:
        """The one invariant the type enforces.

        A velocity computed from values that were since replaced is not wrong in
        any way a reader would notice, which is exactly why carrying it forward
        must be impossible rather than merely discouraged.
        """
        signal = _signal([1.0, 2.0, 3.0])
        with_derivatives = signal.with_derivatives(signal.value, np.ones(3), np.zeros(3))
        assert with_derivatives.velocity is not None

        replaced = with_derivatives.with_values(np.array([9.0, 9.0, 9.0]))
        assert replaced.velocity is None
        assert replaced.acceleration is None

    def test_with_values_leaves_the_original_untouched(self) -> None:
        """Frozen for a reason: two configurations must be comparable on one input."""
        signal = _signal([1.0, 2.0, 3.0])
        signal.with_values(np.array([7.0, 7.0, 7.0]))
        assert signal.value.tolist() == [1.0, 2.0, 3.0]

    def test_missing_marks_exactly_the_nan_samples(self) -> None:
        signal = _signal([1.0, float("nan"), 3.0])
        assert signal.missing.tolist() == [False, True, False]
