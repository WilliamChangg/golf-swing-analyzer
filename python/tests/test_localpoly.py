"""Tests for the local polynomial fit.

The numerical core, so the tests are correspondingly demanding. Three kinds
appear here and they check different things:

*Equivalence* -- on a uniform grid this must reproduce Savitzky-Golay exactly.
That is the strongest available check, because SciPy's implementation is
independent and the agreement has to hold to floating-point precision rather
than to a tolerance someone chose.

*Exactness* -- a polynomial of degree at most `polyorder` must be recovered
perfectly, derivatives included, on any sampling whatsoever. This is a property
of least squares rather than a numerical accident, so any failure means the
design matrix or the derivative scaling is wrong.

*Error bounds* -- against trajectories with closed-form derivatives, with
asserted RMS limits. These are the tests that would catch a filter that is
merely plausible, and the bounds come from `scripts/benchmark_filter.py` rather
than from whatever the code currently happens to produce.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.signal import savgol_filter

from analyzer.filtering.localpoly import (
    MAX_WINDOW_SAMPLES,
    LocalPolynomialError,
    local_polynomial_fit,
)

# A uniform grid, and a window half-width placed at a non-integer multiple of
# the sample interval so that exactly `width` samples fall inside it whichever
# way the floating-point comparison lands.
FPS = 120.0
DT = 1.0 / FPS


def _uniform(count: int) -> np.ndarray:
    return np.arange(count) / FPS


def _window_for(samples: int) -> float:
    return 2.0 * (samples // 2 + 0.25) * DT


def _interior(count: int, width: int) -> slice:
    return slice(width // 2, count - width // 2)


class TestSavitzkyGolayEquivalence:
    """On a uniform grid this is Savitzky-Golay, and must agree with SciPy's."""

    @pytest.mark.parametrize("width", [7, 11, 15])
    @pytest.mark.parametrize("polyorder", [2, 3, 4])
    def test_matches_scipy_for_value_and_derivatives(self, width: int, polyorder: int) -> None:
        rng = np.random.default_rng(0)
        count = 200
        t = _uniform(count)
        y = np.sin(2 * np.pi * 3 * t) + 0.01 * rng.standard_normal(count)

        fit = local_polynomial_fit(
            t,
            y,
            np.ones(count),
            window_s=_window_for(width),
            polyorder=polyorder,
            min_observations=polyorder + 1,
        )
        assert fit.window_samples_max == width

        inner = _interior(count, width)
        for order, ours in ((0, fit.value), (1, fit.velocity), (2, fit.acceleration)):
            theirs = savgol_filter(y, width, polyorder, deriv=order, delta=DT, mode="interp")
            scale = max(float(np.max(np.abs(theirs))), 1.0)
            assert np.allclose(ours[inner], theirs[inner], rtol=0, atol=1e-11 * scale), (
                f"derivative {order} disagrees with scipy"
            )

    def test_edges_are_not_compared_because_the_policy_differs(self) -> None:
        """SciPy shifts its window at the edges; this truncates it.

        Documented rather than reconciled. Shifting keeps the sample count
        constant but stops the window being centred on the evaluation point,
        which on non-uniform samples means fitting a neighbourhood the point is
        not in. Truncating keeps the window honest and accepts that there is
        less data near the ends.
        """
        count, width = 60, 11
        t = _uniform(count)
        y = t**2
        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=_window_for(width), polyorder=2, min_observations=3
        )
        # Still exact at the edges -- a parabola is in the model space -- but
        # supported by fewer samples than the interior.
        assert fit.observations[0] == width // 2 + 1
        assert fit.observations[count // 2] == width
        assert fit.value[0] == pytest.approx(0.0, abs=1e-12)


class TestExactness:
    """A polynomial inside the model space must come back exactly."""

    @pytest.mark.parametrize("polyorder", [2, 3, 4])
    def test_recovers_a_polynomial_and_its_derivatives(self, polyorder: int) -> None:
        rng = np.random.default_rng(3)
        count = 120
        # Deliberately non-uniform: exactness is a property of the fit, not of
        # the grid, and a uniform grid would hide a timestamp bug.
        t = np.sort(rng.uniform(0.0, 2.0, count))
        coefficients = rng.normal(size=polyorder + 1)

        value = np.polyval(coefficients, t)
        first = np.polyval(np.polyder(coefficients, 1), t)
        second = np.polyval(np.polyder(coefficients, 2), t)

        fit = local_polynomial_fit(
            t,
            value,
            np.ones(count),
            window_s=0.4,
            polyorder=polyorder,
            min_observations=polyorder + 1,
        )
        supported = fit.supported
        assert np.count_nonzero(supported) > count // 2

        assert np.allclose(fit.value[supported], value[supported], atol=1e-9)
        assert np.allclose(fit.velocity[supported], first[supported], atol=1e-7)
        assert np.allclose(fit.acceleration[supported], second[supported], atol=1e-5)

    def test_residual_of_an_exact_fit_is_zero(self) -> None:
        count = 100
        t = _uniform(count)
        y = 3.0 - 2.0 * t + 0.5 * t**2
        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=_window_for(9), polyorder=2, min_observations=3
        )
        assert fit.residual_rms == pytest.approx(0.0, abs=1e-12)


class TestErrorBounds:
    """Asserted limits against trajectories whose derivatives are known exactly.

    Bounds are set from `scripts/benchmark_filter.py` with headroom, not from
    the current output. A change that degrades accuracy by a factor of two
    should fail here even though the filter still "works".
    """

    @staticmethod
    def _sine(count: int, frequency: float = 2.0, amplitude: float = 0.2):
        t = _uniform(count)
        omega = 2 * np.pi * frequency
        return (
            t,
            amplitude * np.sin(omega * t),
            amplitude * omega * np.cos(omega * t),
            -amplitude * omega**2 * np.sin(omega * t),
        )

    def test_recovers_velocity_and_acceleration_of_a_sinusoid(self) -> None:
        count = 400
        t, y, velocity, acceleration = self._sine(count)
        rng = np.random.default_rng(1)
        noisy = y + 0.0014 * rng.standard_normal(count)

        fit = local_polynomial_fit(
            t, noisy, np.ones(count), window_s=0.10, polyorder=4, min_observations=5
        )
        inner = slice(20, count - 20)

        def rms(error: np.ndarray) -> float:
            return float(np.sqrt(np.mean(error**2)))

        assert rms(fit.value[inner] - y[inner]) < 0.0015
        assert rms(fit.velocity[inner] - velocity[inner]) < 0.10
        assert rms(fit.acceleration[inner] - acceleration[inner]) < 12.0

    def test_beats_finite_differences_on_noisy_data(self) -> None:
        """The reason this layer exists.

        Differencing noisy samples amplifies the noise by 1/dt, which at 120 fps
        is a factor of 120. The measured improvement at the shipped defaults is
        about 3.2x for both velocity and acceleration -- less than the noise
        amplification alone would suggest, because a degree-4 fit deliberately
        keeps enough freedom to follow real motion and therefore keeps some
        noise with it. The bound is set below the measured figure, not at it.
        """
        count = 400
        t, y, velocity, acceleration = self._sine(count)
        rng = np.random.default_rng(2)
        noisy = y + 0.0014 * rng.standard_normal(count)

        fit = local_polynomial_fit(
            t, noisy, np.ones(count), window_s=0.10, polyorder=4, min_observations=5
        )
        first = np.gradient(noisy, t)
        second = np.gradient(first, t)

        inner = slice(20, count - 20)

        def rms(error: np.ndarray) -> float:
            return float(np.sqrt(np.mean(error**2)))

        assert (
            rms(fit.velocity[inner] - velocity[inner]) < rms(first[inner] - velocity[inner]) / 2.5
        )
        assert (
            rms(fit.acceleration[inner] - acceleration[inner])
            < rms(second[inner] - acceleration[inner]) / 2.5
        )

    def test_smooths_position_rather_than_passing_noise_through(self) -> None:
        count = 400
        t, y, _, _ = self._sine(count)
        rng = np.random.default_rng(4)
        noisy = y + 0.0014 * rng.standard_normal(count)

        fit = local_polynomial_fit(
            t, noisy, np.ones(count), window_s=0.10, polyorder=4, min_observations=5
        )
        inner = slice(20, count - 20)
        residual = float(np.sqrt(np.mean((fit.value[inner] - y[inner]) ** 2)))
        raw = float(np.sqrt(np.mean((noisy[inner] - y[inner]) ** 2)))
        assert residual < raw / 1.5

    def test_does_not_need_a_uniform_grid_to_do_it(self) -> None:
        """The whole design in one assertion.

        Same trajectory, sampled with a rate change part-way through -- which is
        what phone slow-motion produces. Fitting on the real timestamps must beat
        applying a fixed kernel and calling the spacing uniform.
        """
        fast = np.arange(0.0, 0.8, DT)
        slow = np.arange(0.8, 1.6, 4 * DT)
        t = np.concatenate([fast, slow])
        omega = 2 * np.pi * 2.0
        y = 0.2 * np.sin(omega * t)
        velocity = 0.2 * omega * np.cos(omega * t)

        fit = local_polynomial_fit(
            t, y, np.ones(t.size), window_s=0.10, polyorder=4, min_observations=5
        )
        assumed_uniform = savgol_filter(
            y, 13, 4, deriv=1, delta=float(np.mean(np.diff(t))), mode="interp"
        )

        inner = slice(10, t.size - 10)
        ours = np.sqrt(np.nanmean((fit.velocity[inner] - velocity[inner]) ** 2))
        theirs = np.sqrt(np.mean((assumed_uniform[inner] - velocity[inner]) ** 2))
        assert ours < theirs / 10

    def test_costs_nothing_on_a_uniform_grid(self) -> None:
        """Generality is free where the assumption would have held anyway."""
        count = 300
        t, y, _, _ = self._sine(count)
        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=_window_for(13), polyorder=4, min_observations=5
        )
        assumed_uniform = savgol_filter(y, 13, 4, deriv=1, delta=DT, mode="interp")
        inner = _interior(count, 13)
        assert np.allclose(fit.velocity[inner], assumed_uniform[inner], atol=1e-10)


class TestRefusal:
    """Where the fit declines to produce a number, and why."""

    def test_emits_nothing_where_the_window_is_too_sparse(self) -> None:
        count = 40
        t = _uniform(count)
        y = np.sin(t)
        # A window narrow enough to hold 3 samples cannot determine a cubic.
        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=_window_for(3), polyorder=3, min_observations=4
        )
        assert not np.any(fit.supported)
        assert np.all(np.isnan(fit.value))
        assert fit.unsupported_count == count

    def test_does_not_extrapolate_before_the_first_observation(self) -> None:
        """Bracketing's actual job: no value outside the range of the data.

        A leading absence cannot be bridged, because there is nothing on the
        early side to constrain the polynomial. This is where an unconstrained
        fit diverges fastest, and it is the start of the clip -- exactly where a
        wrong address position would bias every phase measurement after it.
        """
        count = 120
        t = _uniform(count)
        y = np.sin(2 * np.pi * t)
        y[:8] = np.nan
        y[-8:] = np.nan

        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=0.20, polyorder=3, min_observations=4
        )
        assert not np.any(fit.supported[:8]), "fit extrapolated before the first observation"
        assert not np.any(fit.supported[-8:]), "fit extrapolated past the last observation"
        assert fit.supported[8] and fit.supported[-9]

    def test_interpolates_an_interior_gap_because_that_is_the_policy_layer_s_job(self) -> None:
        """An interior absence is bridged here; whether it *should* be is decided elsewhere.

        Recording the division of responsibility, because it is easy to assume
        this function refuses interior gaps and to build on that assumption.
        With observations on both sides the polynomial is constrained and the
        fit is legitimate; how long a gap may be bridged is `GapPolicyStage`'s
        decision, delivered through `blocked`.
        """
        count = 120
        t = _uniform(count)
        y = np.sin(2 * np.pi * t)
        y[55:65] = np.nan

        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=0.30, polyorder=3, min_observations=4
        )
        assert np.all(fit.supported[55:65])

        blocked = np.zeros(count, dtype=np.bool_)
        blocked[55:65] = True
        gated = local_polynomial_fit(
            t, y, np.ones(count), window_s=0.30, polyorder=3, min_observations=4, blocked=blocked
        )
        assert np.all(np.isnan(gated.value[55:65]))

    def test_honours_the_blocked_mask(self) -> None:
        count = 100
        t = _uniform(count)
        y = np.sin(2 * np.pi * t)
        blocked = np.zeros(count, dtype=np.bool_)
        blocked[30:40] = True

        fit = local_polynomial_fit(
            t,
            y,
            np.ones(count),
            window_s=0.10,
            polyorder=3,
            min_observations=4,
            blocked=blocked,
        )
        assert np.all(np.isnan(fit.value[30:40]))
        assert np.all(np.isnan(fit.velocity[30:40]))
        assert np.all(np.isnan(fit.acceleration[30:40]))

    def test_zero_weight_samples_do_not_contribute(self) -> None:
        """A zero-weighted sample must be as absent as a missing one.

        Asserted by equality against the same fit with the sample removed
        entirely, rather than against the true trajectory. Comparing against the
        truth would fold in the fit's own model error and could only ever show
        that the outlier was *mostly* excluded.
        """
        count = 80
        t = _uniform(count)
        y = np.sin(2 * np.pi * t)

        corrupted = y.copy()
        corrupted[40] = 1e6
        weight = np.ones(count)
        weight[40] = 0.0

        removed = y.copy()
        removed[40] = np.nan

        kwargs = {"window_s": _window_for(11), "polyorder": 3, "min_observations": 4}
        weighted = local_polynomial_fit(t, corrupted, weight, **kwargs)  # type: ignore[arg-type]
        absent = local_polynomial_fit(t, removed, np.ones(count), **kwargs)  # type: ignore[arg-type]

        assert np.array_equal(weighted.value, absent.value, equal_nan=True)
        assert np.array_equal(weighted.velocity, absent.velocity, equal_nan=True)

    def test_a_constant_fit_reports_no_velocity_rather_than_zero(self) -> None:
        """Degree 0 carries no slope; zero would be a claim it never made."""
        count = 50
        t = _uniform(count)
        fit = local_polynomial_fit(
            t, np.sin(t), np.ones(count), window_s=_window_for(9), polyorder=0, min_observations=1
        )
        assert np.all(np.isnan(fit.velocity))
        assert np.all(np.isnan(fit.acceleration))
        assert np.any(np.isfinite(fit.value))

    def test_a_linear_fit_reports_no_acceleration(self) -> None:
        count = 50
        t = _uniform(count)
        fit = local_polynomial_fit(
            t, np.sin(t), np.ones(count), window_s=_window_for(9), polyorder=1, min_observations=2
        )
        assert np.any(np.isfinite(fit.velocity))
        assert np.all(np.isnan(fit.acceleration))


class TestValidation:
    def test_rejects_a_negative_polyorder(self) -> None:
        with pytest.raises(LocalPolynomialError, match="non-negative"):
            local_polynomial_fit(
                _uniform(10),
                np.zeros(10),
                np.ones(10),
                window_s=0.1,
                polyorder=-1,
                min_observations=1,
            )

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(LocalPolynomialError, match="window_s must be positive"):
            local_polynomial_fit(
                _uniform(10),
                np.zeros(10),
                np.ones(10),
                window_s=0.0,
                polyorder=2,
                min_observations=3,
            )

    def test_rejects_too_few_required_observations(self) -> None:
        with pytest.raises(LocalPolynomialError, match="at least 4 observations"):
            local_polynomial_fit(
                _uniform(10),
                np.zeros(10),
                np.ones(10),
                window_s=0.1,
                polyorder=3,
                min_observations=3,
            )

    def test_rejects_a_window_covering_absurdly_many_samples(self) -> None:
        count = MAX_WINDOW_SAMPLES + 50
        t = np.arange(count) * 1e-6
        with pytest.raises(LocalPolynomialError, match="past the"):
            local_polynomial_fit(
                t, np.zeros(count), np.ones(count), window_s=1.0, polyorder=2, min_observations=3
            )

    def test_an_empty_signal_produces_empty_output(self) -> None:
        fit = local_polynomial_fit(
            np.array([]), np.array([]), np.array([]), window_s=0.1, polyorder=2, min_observations=3
        )
        assert fit.value.size == 0
        assert np.isnan(fit.residual_rms)


class TestResidual:
    def test_residual_estimates_the_noise_that_was_removed(self) -> None:
        """Within a factor of two of the noise actually injected."""
        count = 600
        t = _uniform(count)
        sigma = 0.002
        rng = np.random.default_rng(5)
        y = 0.2 * np.sin(2 * np.pi * 1.5 * t) + sigma * rng.standard_normal(count)

        fit = local_polynomial_fit(
            t, y, np.ones(count), window_s=0.10, polyorder=4, min_observations=5
        )
        assert sigma / 2 < fit.residual_rms < sigma * 2


class TestProperties:
    """Invariants that must hold for any input, checked with Hypothesis.

    These are chosen so that each one fails for a different plausible bug: a
    mis-scaled derivative, a design matrix not re-centred on the evaluation
    point, or a sign error in the local coordinate.
    """

    # Timestamps are built from positive increments rather than sorted from
    # arbitrary floats. Sorting and then filtering out near-duplicates rejects
    # most of what Hypothesis generates, which both slows the search and skews
    # it towards the few shapes that survive.
    _timestamps = st.lists(
        st.floats(min_value=0.004, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=60,
    )
    _values = st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=60,
    )

    @staticmethod
    def _prepare(intervals: list[float], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
        count = min(len(intervals), len(values))
        t = np.cumsum(np.asarray(intervals[:count], dtype=np.float64))
        return t, np.asarray(values[:count], dtype=np.float64)

    @settings(max_examples=50, deadline=None)
    @given(_timestamps, _values, st.floats(min_value=-100, max_value=100, allow_nan=False))
    def test_adding_a_constant_shifts_the_value_and_leaves_derivatives_alone(
        self, times: list[float], values: list[float], offset: float
    ) -> None:
        t, y = self._prepare(times, values)
        kwargs = {"window_s": 1.0, "polyorder": 2, "min_observations": 3}

        base = local_polynomial_fit(t, y, np.ones(t.size), **kwargs)  # type: ignore[arg-type]
        shifted = local_polynomial_fit(t, y + offset, np.ones(t.size), **kwargs)  # type: ignore[arg-type]

        mask = base.supported
        assume(np.any(mask))
        assert np.allclose(shifted.value[mask], base.value[mask] + offset, atol=1e-6)
        assert np.allclose(shifted.velocity[mask], base.velocity[mask], atol=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(_timestamps, _values)
    def test_shifting_time_changes_nothing(self, times: list[float], values: list[float]) -> None:
        """The fit is re-centred on each evaluation point, so absolute time cannot matter."""
        t, y = self._prepare(times, values)
        kwargs = {"window_s": 1.0, "polyorder": 3, "min_observations": 4}

        base = local_polynomial_fit(t, y, np.ones(t.size), **kwargs)  # type: ignore[arg-type]
        moved = local_polynomial_fit(t + 1000.0, y, np.ones(t.size), **kwargs)  # type: ignore[arg-type]

        mask = base.supported
        assume(np.any(mask))
        assert np.allclose(moved.value[mask], base.value[mask], atol=1e-6)
        assert np.allclose(moved.velocity[mask], base.velocity[mask], atol=1e-5)

    @settings(max_examples=50, deadline=None)
    @given(_timestamps, _values)
    def test_reversing_time_negates_velocity_and_preserves_acceleration(
        self, times: list[float], values: list[float]
    ) -> None:
        """Catches a sign error in the local coordinate that nothing else would."""
        t, y = self._prepare(times, values)
        kwargs = {"window_s": 1.0, "polyorder": 3, "min_observations": 4}

        forward = local_polynomial_fit(t, y, np.ones(t.size), **kwargs)  # type: ignore[arg-type]
        reversed_t = -t[::-1]
        backward = local_polynomial_fit(reversed_t, y[::-1], np.ones(t.size), **kwargs)  # type: ignore[arg-type]

        mask = forward.supported & backward.supported[::-1]
        assume(np.any(mask))
        assert np.allclose(backward.value[::-1][mask], forward.value[mask], atol=1e-6)
        assert np.allclose(backward.velocity[::-1][mask], -forward.velocity[mask], atol=1e-5)
        assert np.allclose(backward.acceleration[::-1][mask], forward.acceleration[mask], atol=1e-3)

    @settings(max_examples=50, deadline=None)
    @given(_timestamps, _values)
    def test_never_emits_a_value_without_enough_observations(
        self, times: list[float], values: list[float]
    ) -> None:
        t, y = self._prepare(times, values)
        fit = local_polynomial_fit(
            t, y, np.ones(t.size), window_s=0.2, polyorder=3, min_observations=4
        )
        emitted = np.isfinite(fit.value)
        assert np.all(fit.observations[emitted] >= 4)
