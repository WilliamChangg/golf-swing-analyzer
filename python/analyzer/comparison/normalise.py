"""Putting two swings on one clock.

Two swings take different amounts of time, and one of them may have been filmed
at four times the frame rate of the other. Neither curve can be laid over the
other against seconds, so both are mapped onto **swing position**: 0 at the
takeaway, 1 at the top, 2 at impact, 3 at the finish, linear in time within each
phase.

## Why piecewise, and why these knots

A single linear stretch -- scaling each clip's takeaway-to-finish interval to
the same length -- is the obvious alternative and is wrong in a way that matters.
The ratio of backswing to downswing is the most quoted number in golf and varies
between swings by a factor of two; a uniform stretch therefore lands one clip's
top a tenth of the axis away from the other's, and every comparison afterwards is
between the top of one swing and a point partway down the other. The four events
are the only instants the two recordings are known to share, so they are the only
places a map between them can be pinned.

Linear **in time** within a phase rather than in frame number, because the
footage is frequently variable-rate and because a slow-motion factor has already
been divided out of the timestamps. Frame numbers index the file; these
timestamps do not.

## What the map destroys, and why that is stated rather than fixed

The knots coincide by construction, so **the map removes every timing difference
between the two swings**. That is what makes the shapes comparable and it is not
a side effect to be worked around: nothing in an overlaid trajectory can say that
one player's backswing was longer, because the normalisation is precisely the
operation that made them the same length. Durations are compared as durations, as
metrics, with brackets of their own, and `PhaseClock` carries its knots so that
what was divided out stays visible.

## The bracket, and why it is a range rather than a derivative

Each knot is located to a frame, so a real instant's position on the axis is
known only as well as the two events bounding it. Propagating that through the
linear map inside a phase is exact: at a fraction `u` between knots whose
ambiguities are `dt_left` and `dt_right`, the instant is ambiguous by
`(1 - u) * dt_left + u * dt_right` -- **in the clip's own seconds**, with the
phase duration cancelling out. Normalising moves where each clip is sampled; it
does not change how well any instant is known.

What that ambiguity does to a *value* is then bounded by looking, rather than by
differentiating: the bracket on a sample is the largest the signal moves anywhere
inside its own ambiguity window. A slope would be a linearisation, and would
under-report exactly at a peak -- which on a hand-speed curve is the middle of
the downswing, the part of the swing anybody comparing two of them cares about.
Taking the range over the window is a bound rather than an estimate, and it costs
one pass over the samples the window covers.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.coaching.bracket import frame_interval_s
from analyzer.contracts.comparison import SWING_POSITION, ClockKnot, PhaseClock
from analyzer.contracts.phases import SwingEvent, SwingPhases

# The four events, in the order they happen. A clock needs all of them.
SWING_ORDER: tuple[SwingEvent, ...] = (
    SwingEvent.TAKEAWAY,
    SwingEvent.TOP,
    SwingEvent.IMPACT,
    SwingEvent.FINISH,
)

FIRST_POSITION = SWING_POSITION[SWING_ORDER[0]]
LAST_POSITION = SWING_POSITION[SWING_ORDER[-1]]

_NO_EVENTS = (
    "This clip did not locate all four swing events, so there is no interval "
    "between known instants to normalise onto. Three knots cannot map the phase "
    "they do not bound, and a fourth invented here would decide where every "
    "sample lands."
)


class NormalisationError(ValueError):
    """Two clips could not be put on one clock."""


@dataclass(frozen=True)
class Resampled:
    """One signal read at a set of normalised positions.

    `value` and `bracket` are NaN wherever the clip supports nothing there --
    outside the tracked range, or across a stretch the filter blocked. They are
    never interpolated across such a gap: an overlay drawn through one would show
    two curves agreeing at a position where one of them was not measured.
    """

    position: NDArray[np.float64]
    value: NDArray[np.float64]
    bracket: NDArray[np.float64]


def build_clock(phases: SwingPhases) -> PhaseClock:
    """The map from this clip's clock onto swing position.

    Returns an unusable clock rather than raising when the clip did not locate
    all four events, so that a caller comparing two clips can report *which* of
    them could not be normalised -- which is the actionable half of the answer.
    """
    interval = frame_interval_s(phases)

    if not phases.detected:
        return PhaseClock(
            knots=[],
            usable=False,
            frame_interval_s=interval,
            slow_motion_factor=phases.slow_motion_factor,
            methodology=(
                "No swing was detected in this clip, so it has no events and no "
                "clock to normalise onto."
            ),
        )

    located = [phases.event(event) for event in SWING_ORDER]
    if any(entry is None for entry in located):
        missing = ", ".join(
            event.value for event, entry in zip(SWING_ORDER, located, strict=True) if entry is None
        )
        return PhaseClock(
            knots=[],
            usable=False,
            frame_interval_s=interval,
            slow_motion_factor=phases.slow_motion_factor,
            methodology=f"{_NO_EVENTS} Missing: {missing}.",
        )

    knots: list[ClockKnot] = []
    for event, entry in zip(SWING_ORDER, located, strict=True):
        assert entry is not None  # noqa: S101 - narrowed by the check above
        ambiguity, source = _ambiguity(entry.corroboration_delta_s, interval)
        knots.append(
            ClockKnot(
                event=event,
                position=SWING_POSITION[event],
                frame_index=entry.frame_index,
                timestamp_s=entry.timestamp_s,
                confidence=entry.confidence.overall,
                ambiguity_s=ambiguity,
                ambiguity_source=source,
            )
        )

    times = [knot.timestamp_s for knot in knots]
    if any(later <= earlier for earlier, later in itertools.pairwise(times)):
        # Detection nests its searches so this cannot happen from the shipped
        # rules; it is checked anyway because a non-increasing knot would make
        # the map non-monotone and silently fold part of one swing onto another.
        return PhaseClock(
            knots=knots,
            usable=False,
            frame_interval_s=interval,
            slow_motion_factor=phases.slow_motion_factor,
            methodology=(
                "This clip's events are not strictly increasing in time, so they "
                "do not divide it into phases and cannot be used as knots."
            ),
        )

    return PhaseClock(
        knots=knots,
        usable=True,
        frame_interval_s=interval,
        slow_motion_factor=phases.slow_motion_factor,
        methodology=(
            "Piecewise-linear in real seconds between the four detected events, "
            "which are the only instants two recordings of a swing are known to "
            "share. Position 0 is the takeaway, 1 the top, 2 impact and 3 the "
            "finish. Every difference in timing between two clips normalised this "
            "way has been divided out by the map itself; the knots above are what "
            "it divided out."
        ),
    )


def _ambiguity(corroboration_delta_s: float | None, interval: float | None) -> tuple[float, str]:
    """How far from its reported instant an event could really be, in real seconds.

    One frame interval is the floor: an event is located to a frame and the true
    instant lies somewhere inside it, which is the same convention
    `coaching.bracket` applies to a duration.

    Where a second, independent estimate of the same instant exists and disagrees
    by more than that, the disagreement is the honest number. Phase 4 corroborates
    impact with the lowest point of the hand arc; two estimates 40 ms apart are
    evidence that the instant is ambiguous by 40 ms, and reporting one frame there
    would be quoting the finer of two numbers that contradict each other.
    """
    if interval is None or not math.isfinite(interval) or interval <= 0.0:
        return 0.0, (
            "This clip's frame interval could not be established from its events, "
            "so nothing here bounds where the instant sits."
        )

    if corroboration_delta_s is not None and abs(corroboration_delta_s) > interval:
        return abs(corroboration_delta_s), (
            "Two independent estimates of this instant disagree by more than one "
            "frame, so their disagreement is the ambiguity rather than the clock."
        )

    return interval, "One frame interval: the instant is located to a frame and lies inside it."


def positions(config_samples: int) -> NDArray[np.float64]:
    """The normalised axis, takeaway to finish inclusive."""
    return np.linspace(FIRST_POSITION, LAST_POSITION, config_samples, dtype=np.float64)


def to_time(clock: PhaseClock, position: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip times for positions on the normalised axis.

    NaN outside [0, 3]: the address phase is bounded by an event on one side
    only, and anything past the finish is past the last knot. Both would be
    extrapolations, and an extrapolated instant is exactly the kind of invented
    number this system refuses elsewhere.
    """
    _require(clock)
    knot_positions = np.array([knot.position for knot in clock.knots], dtype=np.float64)
    knot_times = np.array([knot.timestamp_s for knot in clock.knots], dtype=np.float64)

    inside = (position >= knot_positions[0]) & (position <= knot_positions[-1])
    times = np.full(position.shape, np.nan, dtype=np.float64)
    times[inside] = np.interp(position[inside], knot_positions, knot_times)
    return times


def to_position(clock: PhaseClock, time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Positions on the normalised axis for clip times. The inverse of `to_time`."""
    _require(clock)
    knot_positions = np.array([knot.position for knot in clock.knots], dtype=np.float64)
    knot_times = np.array([knot.timestamp_s for knot in clock.knots], dtype=np.float64)

    inside = (time_s >= knot_times[0]) & (time_s <= knot_times[-1])
    result = np.full(time_s.shape, np.nan, dtype=np.float64)
    result[inside] = np.interp(time_s[inside], knot_times, knot_positions)
    return result


def ambiguity_at(clock: PhaseClock, position: NDArray[np.float64]) -> NDArray[np.float64]:
    """How ambiguous the instant at each position is, in this clip's real seconds.

    Exact first-order propagation through the piecewise-linear map: inside a phase
    the position is `p_left + (t - t_left) / duration`, so a fraction `u` along it
    inherits `(1 - u)` of the left knot's ambiguity and `u` of the right's. The
    phase duration cancels, which is the whole point -- **normalising changes
    where a clip is sampled, not how well any instant in it is known.**
    """
    _require(clock)
    knot_positions = np.array([knot.position for knot in clock.knots], dtype=np.float64)
    ambiguities = np.array([knot.ambiguity_s for knot in clock.knots], dtype=np.float64)

    inside = (position >= knot_positions[0]) & (position <= knot_positions[-1])
    result = np.full(position.shape, np.nan, dtype=np.float64)
    result[inside] = np.interp(position[inside], knot_positions, ambiguities)
    return result


def resample(
    clock: PhaseClock,
    time_s: NDArray[np.float64],
    values: NDArray[np.float64],
    position: NDArray[np.float64],
) -> Resampled:
    """Read one signal at each normalised position, with the bracket it carries.

    The value is linearly interpolated between the two clip samples bracketing the
    instant, and is NaN when either of them is -- never across a gap. The bracket
    is the largest the signal moves anywhere inside that instant's own ambiguity
    window, which is a bound rather than a linearisation; see the module
    docstring.
    """
    times = to_time(clock, position)
    slack = ambiguity_at(clock, position)

    value = np.full(position.shape, np.nan, dtype=np.float64)
    bracket = np.full(position.shape, np.nan, dtype=np.float64)

    order = np.argsort(time_s)
    clock_t = np.asarray(time_s, dtype=np.float64)[order]
    signal = np.asarray(values, dtype=np.float64)[order]
    if clock_t.size < 2:
        return Resampled(position=position, value=value, bracket=bracket)

    for index, (instant, window) in enumerate(zip(times, slack, strict=True)):
        if not math.isfinite(instant):
            continue
        here = _interpolate(clock_t, signal, instant)
        if not math.isfinite(here):
            continue
        value[index] = here
        bracket[index] = _range_within(clock_t, signal, instant, window, here)

    return Resampled(position=position, value=value, bracket=bracket)


def _interpolate(
    clock_t: NDArray[np.float64], signal: NDArray[np.float64], instant: float
) -> float:
    """The signal at one instant, refusing to bridge a gap.

    `np.interp` would happily draw a straight line across a stretch the filter
    blocked, which is the one thing an overlay must not do: it would show two
    curves agreeing at a position where one of them was never measured.
    """
    if instant < clock_t[0] or instant > clock_t[-1]:
        return float("nan")

    upper = int(np.searchsorted(clock_t, instant, side="left"))
    if upper == 0:
        return float(signal[0])

    lower = upper - 1
    if upper >= clock_t.size:
        return float(signal[-1])

    left, right = float(signal[lower]), float(signal[upper])
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")

    span = float(clock_t[upper] - clock_t[lower])
    if span <= 0.0:
        return left
    weight = (instant - float(clock_t[lower])) / span
    return left + weight * (right - left)


def _range_within(
    clock_t: NDArray[np.float64],
    signal: NDArray[np.float64],
    instant: float,
    window: float,
    here: float,
) -> float:
    """How far the signal moves inside an instant's ambiguity window.

    The window is widened by one sample at each end before the range is taken, and
    that is not slack. A reading anywhere inside the window is interpolated
    between the two clip samples bracketing it, and the sample on the far side of
    an edge is one of that pair -- so a range over the strictly-interior samples
    alone is not a bound on what a shifted reading can return. Including the
    hull's endpoints makes it one, which a test asserts directly by re-reading
    the signal from a clock shifted by a whole frame.

    NaN when the window holds no finite sample at all, which happens at the very
    ends of a clip. Reported as unknown rather than as zero: a bracket of zero
    would let any difference count as resolved, exactly where the evidence is
    thinnest.
    """
    if not math.isfinite(window) or window <= 0.0:
        return 0.0

    low = max(int(np.searchsorted(clock_t, instant - window, side="left")) - 1, 0)
    high = min(int(np.searchsorted(clock_t, instant + window, side="right")) + 1, clock_t.size)
    if high <= low:
        return float("nan")

    inside = signal[low:high]
    finite = inside[np.isfinite(inside)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite - here)))


def _require(clock: PhaseClock) -> None:
    if not clock.usable or len(clock.knots) != len(SWING_ORDER):
        raise NormalisationError(clock.methodology)
