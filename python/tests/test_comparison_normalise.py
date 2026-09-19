"""The phase-relative clock, against warps whose answer is known by construction.

`tests/synthetic.py` parametrises its swing by the three event times and builds
the hand arc from the *fraction* through each phase. Two swings with different
tempos are therefore **the same swing under a piecewise-linear time warp**, and
the normalisation's job is to undo exactly that warp. That is what makes these
assertions measurements rather than regression pins: the residual after
normalising is the error of the map, because the truth is zero.

The nonlinear case is here for the same reason and says the opposite. A warp that
is not linear within a phase is *not* recovered, the residual is bounded rather
than zero, and the bound is measured and asserted rather than described.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.comparison import normalise
from analyzer.contracts.comparison import SWING_POSITION
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.contracts.phases import SwingEvent
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases
from analyzer.phases.signals import swing_signals
from tests.synthetic import SwingShape, swing_sequence
from tests.synthetic_coaching import phases as built_phases

FILTER = FilterConfig(smoothing=SmoothingConfig(window_s=0.12))


def analysed(shape: SwingShape, *, fps: float = 120.0, tail_s: float = 0.4):
    """One synthetic swing, all the way to filtered signals and a detection.

    The clip runs to the shape's own finish plus a tail rather than to a fixed
    length, because a shape whose finish falls past the end of the clip is a
    truncated swing -- a real case, and not the case these tests are about.
    """
    sequence = swing_sequence(duration_s=shape.finish_s + tail_s, fps=fps, shape=shape)
    filtered = filter_sequence(sequence, FILTER, space=LandmarkSpace.FRAME_WIDTHS)
    return filtered, detect_phases(filtered, None), swing_signals(filtered)


# --- the clock ------------------------------------------------------------


def test_clock_puts_the_four_events_on_the_integers() -> None:
    _, detected, _ = analysed(SwingShape())
    clock = normalise.build_clock(detected)

    assert clock.usable
    assert [knot.event for knot in clock.knots] == list(normalise.SWING_ORDER)
    assert [knot.position for knot in clock.knots] == [0.0, 1.0, 2.0, 3.0]
    times = [knot.timestamp_s for knot in clock.knots]
    assert times == sorted(times)


def test_a_clip_with_no_swing_has_no_clock() -> None:
    clock = normalise.build_clock(built_phases(detected=False))

    assert not clock.usable
    assert clock.knots == []
    assert "no swing" in clock.methodology.lower()


def test_a_clip_missing_an_event_names_it() -> None:
    detected = built_phases()
    detected.events = [entry for entry in detected.events if entry.event is not SwingEvent.FINISH]

    clock = normalise.build_clock(detected)

    assert not clock.usable
    assert "finish" in clock.methodology


def test_events_out_of_order_are_refused_rather_than_folded() -> None:
    detected = built_phases(takeaway_frame=10, top_frame=40, impact_frame=35, finish_frame=70)

    clock = normalise.build_clock(detected)

    assert not clock.usable
    assert "increasing" in clock.methodology


def test_a_knot_maps_to_its_own_position_exactly() -> None:
    _, detected, _ = analysed(SwingShape())
    clock = normalise.build_clock(detected)

    times = np.array([knot.timestamp_s for knot in clock.knots])
    positions = normalise.to_position(clock, times)

    assert positions == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=1e-12)


def test_the_map_round_trips() -> None:
    _, detected, _ = analysed(SwingShape())
    clock = normalise.build_clock(detected)

    axis = normalise.positions(97)
    assert normalise.to_position(clock, normalise.to_time(clock, axis)) == pytest.approx(
        axis, abs=1e-12
    )


def test_outside_the_swing_is_refused_rather_than_extrapolated() -> None:
    _, detected, _ = analysed(SwingShape())
    clock = normalise.build_clock(detected)

    outside = normalise.to_time(clock, np.array([-0.5, -1e-9 + 0.0, 3.5]))
    assert np.isnan(outside[0])
    assert np.isnan(outside[2])


# --- known warps ----------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        SwingShape(takeaway_s=0.50, top_s=1.30, impact_s=1.70),
        # Half the backswing, the same descent: a tempo of 1.0 against 2.0.
        SwingShape(takeaway_s=0.50, top_s=0.90, impact_s=1.30),
        # A long, slow swing that starts later.
        SwingShape(takeaway_s=0.80, top_s=2.00, impact_s=2.55),
    ],
)
def test_a_piecewise_linear_warp_is_recovered(shape: SwingShape) -> None:
    """Two tempos of the one swing normalise onto the same curve.

    The fixture builds its arc from the fraction through each phase, so these
    really are one swing warped in time -- and the residual after normalising is
    therefore the map's own error rather than a difference between two motions.

    The tolerance is not zero because the two clips sample that shared curve at
    different instants and each is smoothed over a window in seconds, which is a
    different number of samples on a swing of a different length. What it is not
    is proportional to the tempo difference, which is the thing being asserted.
    """
    baseline = SwingShape()
    _, reference_events, reference_signals = analysed(baseline)
    _, target_events, target_signals = analysed(shape)

    reference_clock = normalise.build_clock(reference_events)
    target_clock = normalise.build_clock(target_events)
    axis = normalise.positions(121)

    left = normalise.resample(reference_clock, reference_signals.t, reference_signals.height, axis)
    right = normalise.resample(target_clock, target_signals.t, target_signals.height, axis)

    both = np.isfinite(left.value) & np.isfinite(right.value)
    assert np.count_nonzero(both) > 100
    residual = np.abs(left.value[both] - right.value[both])
    # Hand height ranges over roughly 0.3 frame widths on this fixture, so 0.01
    # is about 3% of the signal.
    assert float(np.max(residual)) < 0.01


def test_a_warp_inside_a_phase_is_not_recovered_and_the_residual_says_so() -> None:
    """The honest limit of a piecewise-linear map, measured rather than described.

    The knots pin four instants. Between them the map is linear, so a swing that
    reached the top by a different *route* -- the same four instants, a different
    shape in between -- normalises onto a curve that does not coincide, and
    should not. This is the case the whole comparison exists to find, so the test
    asserts that it survives the normalisation instead of being flattened by it.
    """
    baseline = SwingShape()
    steeper = SwingShape(arc_top_angle=1.7)

    _, reference_events, reference_signals = analysed(baseline)
    _, target_events, target_signals = analysed(steeper)

    axis = normalise.positions(121)
    left = normalise.resample(
        normalise.build_clock(reference_events), reference_signals.t, reference_signals.height, axis
    )
    right = normalise.resample(
        normalise.build_clock(target_events), target_signals.t, target_signals.height, axis
    )

    both = np.isfinite(left.value) & np.isfinite(right.value)
    residual = np.abs(left.value[both] - right.value[both])
    assert float(np.max(residual)) > 0.03


def test_normalising_erases_the_timing_difference_it_divided_out() -> None:
    """Which is why the durations are compared as metrics and not read off a plot.

    Two swings whose backswings differ by a factor of two put their tops at the
    same position on the axis, by construction. The difference has not been
    measured away -- it is on the clock's own knots, where a reader can still see
    it.
    """
    _, slow_events, _ = analysed(SwingShape(takeaway_s=0.5, top_s=1.3, impact_s=1.7))
    _, quick_events, _ = analysed(SwingShape(takeaway_s=0.5, top_s=0.9, impact_s=1.3))

    slow = normalise.build_clock(slow_events)
    quick = normalise.build_clock(quick_events)

    assert slow.knot(SwingEvent.TOP) is not None
    assert quick.knot(SwingEvent.TOP) is not None
    assert slow.knot(SwingEvent.TOP).position == quick.knot(SwingEvent.TOP).position

    slow_backswing = (
        slow.knot(SwingEvent.TOP).timestamp_s - slow.knot(SwingEvent.TAKEAWAY).timestamp_s
    )
    quick_backswing = (
        quick.knot(SwingEvent.TOP).timestamp_s - quick.knot(SwingEvent.TAKEAWAY).timestamp_s
    )
    assert slow_backswing > 1.7 * quick_backswing


# --- the ambiguity, and what it does to a value ---------------------------


def test_ambiguity_is_one_frame_interval_by_default() -> None:
    detected = built_phases(interval_s=1.0 / 240.0)
    clock = normalise.build_clock(detected)

    for knot in clock.knots:
        assert knot.ambiguity_s == pytest.approx(1.0 / 240.0)
        assert "one frame" in knot.ambiguity_source.lower()


def test_a_disagreeing_corroboration_widens_the_ambiguity() -> None:
    detected = built_phases(interval_s=1.0 / 240.0)
    impact = next(entry for entry in detected.events if entry.event is SwingEvent.IMPACT)
    impact.corroboration_frame = impact.frame_index + 10
    impact.corroboration_delta_s = 10.0 / 240.0

    clock = normalise.build_clock(detected)
    knot = clock.knot(SwingEvent.IMPACT)

    assert knot is not None
    assert knot.ambiguity_s == pytest.approx(10.0 / 240.0)
    assert "disagree" in knot.ambiguity_source


def test_an_agreeing_corroboration_does_not_narrow_it() -> None:
    """One frame is the floor, not a starting guess to be improved on."""
    detected = built_phases(interval_s=1.0 / 240.0)
    impact = next(entry for entry in detected.events if entry.event is SwingEvent.IMPACT)
    impact.corroboration_frame = impact.frame_index
    impact.corroboration_delta_s = 0.0

    knot = normalise.build_clock(detected).knot(SwingEvent.IMPACT)

    assert knot is not None
    assert knot.ambiguity_s == pytest.approx(1.0 / 240.0)


def test_ambiguity_interpolates_between_the_knots_and_not_the_phases() -> None:
    """The phase duration cancels: normalising moves samples, not certainty.

    Two clips of wildly different lengths with the same clock resolution carry the
    same ambiguity at the same position, which is the property the module docstring
    claims and the reason the bracket can be computed on either clip's own clock.
    """
    short = normalise.build_clock(
        built_phases(
            interval_s=1 / 120, takeaway_frame=10, top_frame=40, impact_frame=52, finish_frame=76
        )
    )
    long = normalise.build_clock(
        built_phases(
            interval_s=1 / 120, takeaway_frame=10, top_frame=100, impact_frame=136, finish_frame=208
        )
    )

    axis = normalise.positions(31)
    assert normalise.ambiguity_at(short, axis) == pytest.approx(
        normalise.ambiguity_at(long, axis), nan_ok=True
    )


def test_the_bracket_is_zero_on_a_flat_signal_and_large_on_a_steep_one() -> None:
    _, detected, signals = analysed(SwingShape())
    clock = normalise.build_clock(detected)
    axis = normalise.positions(61)

    flat = normalise.resample(clock, signals.t, np.zeros_like(signals.t), axis)
    assert np.nanmax(flat.bracket) == pytest.approx(0.0)

    speed = normalise.resample(clock, signals.t, signals.speed, axis)
    # Impact is where the hands are fastest and therefore where the speed curve
    # is locally *flat*: a frame either side of the peak changes the reading very
    # little. Halfway down it is climbing hard. So the bracket is larger in the
    # middle of the descent than at its end, which is the opposite of where a
    # reader would expect the measurement to be least certain.
    mid_descent = int(np.argmin(np.abs(axis - 1.5)))
    at_impact = int(np.argmin(np.abs(axis - 2.0)))
    assert speed.bracket[mid_descent] > speed.bracket[at_impact]


def test_the_bracket_bounds_what_one_frame_of_ambiguity_does() -> None:
    """The claim the bracket makes, checked directly against the signal.

    Shifting the whole clip's clock by one frame is exactly the error the bracket
    exists to absorb, so re-reading the signal from a clock shifted that far must
    move no sample by more than its own bracket.
    """
    _, detected, signals = analysed(SwingShape())
    clock = normalise.build_clock(detected)
    axis = normalise.positions(121)

    honest = normalise.resample(clock, signals.t, signals.speed, axis)

    shifted = clock.model_copy(deep=True)
    for knot in shifted.knots:
        knot.timestamp_s += 1.0 / 120.0
    moved = normalise.resample(shifted, signals.t, signals.speed, axis)

    both = np.isfinite(honest.value) & np.isfinite(moved.value) & np.isfinite(honest.bracket)
    assert np.count_nonzero(both) > 80
    assert np.all(np.abs(moved.value[both] - honest.value[both]) <= honest.bracket[both] + 1e-9)


def test_a_gap_is_not_bridged() -> None:
    """A blocked stretch comes back null rather than as a straight line across it."""
    _, detected, signals = analysed(SwingShape())
    clock = normalise.build_clock(detected)

    holed = np.array(signals.height, dtype=np.float64)
    middle = len(holed) // 2
    holed[middle - 12 : middle + 12] = np.nan

    axis = normalise.positions(121)
    read = normalise.resample(clock, signals.t, holed, axis)

    assert np.any(np.isnan(read.value))
    # And the hole is where the hole is, not spread over the whole clip.
    assert np.count_nonzero(np.isnan(read.value)) < axis.size // 2


def test_resampling_an_unusable_clock_raises() -> None:
    clock = normalise.build_clock(built_phases(detected=False))

    with pytest.raises(normalise.NormalisationError):
        normalise.to_time(clock, np.array([0.5]))


def test_positions_span_takeaway_to_finish_inclusive() -> None:
    axis = normalise.positions(13)

    assert axis[0] == SWING_POSITION[SwingEvent.TAKEAWAY]
    assert axis[-1] == SWING_POSITION[SwingEvent.FINISH]
    assert axis.size == 13
