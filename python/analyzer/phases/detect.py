"""Locating swing events in the hand signal.

Deterministic and rule-based, with no model and no training. That is a choice
rather than a limitation of ambition: the events are defined by the shape of the
hand-speed signal, the shape is not subtle, and a rule that can be read is a
rule that can be argued with. Phase 12 trains a learned detector and compares it
against this one on the same held-out clips, which is only a meaningful
comparison because this one exists first.

## The structure being looked for

    speed  |          ___                    /\\
           |      ___/   \\__  <- top        /  \\   <- impact
           |  ___/          \\__          __/    \\___
           |_/                  \\_______/           \\____
           +--------------------------------------------------
            address   backswing      downswing   follow-through

One minimum between two maxima, with the minimum at the highest point the hands
reach. Everything below is that sentence written carefully.

## Order matters

Events are not found left to right. Impact is found first because it is the
clearest feature in the signal -- the global speed maximum, several times
anything else. The top is then found *before* impact, the takeaway *before* the
top, and the finish *after* impact. Each search is bounded by an event already
located, so a rule can never place the top after impact or the takeaway after
the top; those orderings are impossible by construction rather than checked
afterwards.

## What it refuses

A clip with no swing in it produces no events, not events at frame zero. The
gate is how far the hands ranged measured in the subject's own torso lengths,
rather than a threshold on an absolute distance, which would mean different
things at every framing. The same test is applied again to the backswing and the
downswing separately, because a clip that is still except for one twitch can
otherwise be organised into a swing shape made almost entirely of noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks  # type: ignore[import-untyped]

from analyzer.contracts.phases import (
    DetectedEvent,
    DetectedPhase,
    EventConfidence,
    HandSignalInfo,
    PhaseConfig,
    SwingEvent,
    SwingPhase,
    SwingPhases,
)
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.phases.signals import SwingSignals, swing_signals

# A second speed peak this close to the first in height is treated as another
# swing-like motion rather than as part of the same swing.
_SECOND_SWING_FRACTION = 0.5

# Resolution scoring: a feature of duration D is fully resolved by a smoothing
# window of D/3 or less, and not resolved at all once the window reaches D.
_FULLY_RESOLVED_RATIO = 3.0


def _nan_argmax(values: NDArray[np.float64], start: int, stop: int) -> int | None:
    """Index of the largest finite value in [start, stop), or None if there is none."""
    if stop <= start:
        return None
    window = values[start:stop]
    if not np.any(np.isfinite(window)):
        return None
    return start + int(np.nanargmax(window))


def _nan_argmin(values: NDArray[np.float64], start: int, stop: int) -> int | None:
    if stop <= start:
        return None
    window = values[start:stop]
    if not np.any(np.isfinite(window)):
        return None
    return start + int(np.nanargmin(window))


def _window_bounds(t: NDArray[np.float64], index: int, span_s: float) -> tuple[int, int]:
    """Half-open index range within `span_s`/2 either side of `index`."""
    half = span_s / 2.0
    start = int(np.searchsorted(t, t[index] - half, side="left"))
    stop = int(np.searchsorted(t, t[index] + half, side="right"))
    return start, stop


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal half-open [start, stop) runs of True."""
    if mask.size == 0 or not bool(mask.any()):
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _visibility_score(signals: SwingSignals, index: int, span_s: float) -> float:
    """Mean reported visibility of the hand landmark around an event."""
    start, stop = _window_bounds(signals.t, index, span_s)
    window = signals.hand.visibility[start:stop]
    finite = window[np.isfinite(window)]
    return float(np.clip(np.mean(finite), 0.0, 1.0)) if finite.size else 0.0


def _resolution_score(duration_s: float, window_s: float) -> float:
    """Whether a feature of this duration survives a smoothing window of that width.

    Full marks when the window is a third of the feature or less; zero once the
    window is as long as the feature, because by then the filter has averaged the
    event away and whatever peak remains is an artefact of the smoothing rather
    than a measurement of the motion.

    This is the factor that stops a 24 fps clip reporting a confident impact. Its
    frame rate forces a window about as long as a downswing, and no amount of
    clarity in the resulting curve makes that measurable.
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or window_s <= 0:
        return 0.0
    fully = duration_s / _FULLY_RESOLVED_RATIO
    if window_s <= fully:
        return 1.0
    if window_s >= duration_s:
        return 0.0
    return float((duration_s - window_s) / (duration_s - fully))


def _crossing_margin(
    speed: NDArray[np.float64],
    t: NDArray[np.float64],
    index: int,
    threshold: float,
    span_s: float,
    *,
    rising: bool,
) -> float:
    """How cleanly a speed threshold crossing separates still from moving.

    The fraction of nearby frames on the still side that really are still, and on
    the moving side that really are moving, averaged. A crisp takeaway scores 1;
    a threshold clipped repeatedly by noise scores near 0.5, which is what an
    arbitrary crossing deserves.
    """
    start, stop = _window_bounds(t, index, span_s * 2)
    before, after = speed[start:index], speed[index:stop]
    quiet, moving = (before, after) if rising else (after, before)

    scores = []
    quiet_finite = quiet[np.isfinite(quiet)]
    if quiet_finite.size:
        scores.append(float(np.mean(quiet_finite < threshold)))
    moving_finite = moving[np.isfinite(moving)]
    if moving_finite.size:
        scores.append(float(np.mean(moving_finite >= threshold)))
    return float(np.mean(scores)) if scores else 0.0


def _span_travel(signals: SwingSignals, start: int, stop: int) -> float:
    """Straight-line distance the hands covered between two frames."""
    position, height = signals.hand.position, signals.height
    for index in (start, stop):
        if not (np.isfinite(position[index, 0]) and np.isfinite(height[index])):
            return float("nan")
    return float(np.hypot(position[stop, 0] - position[start, 0], height[stop] - height[start]))


def _empty_result(signals: SwingSignals, config: PhaseConfig, warnings: list[str]) -> SwingPhases:
    return SwingPhases(
        detected=False,
        hand=_hand_info(signals),
        frames=len(signals),
        slow_motion_factor=signals.slow_motion_factor,
        config=config,
        warnings=warnings,
    )


def _hand_info(signals: SwingSignals) -> HandSignalInfo:
    return HandSignalInfo(
        source=signals.hand.source,
        valid_frames=int(np.count_nonzero(signals.hand.valid)),
        total_frames=len(signals),
        peak_speed=signals.peak_speed,
        travel=signals.travel,
        torso_length=signals.torso_length,
        travel_ratio=signals.travel_ratio,
    )


def detect_phases(filtered: FilteredSequence, config: PhaseConfig | None = None) -> SwingPhases:
    """Locate the four swing events in a filtered pose sequence."""
    resolved = config or PhaseConfig()
    signals = swing_signals(filtered)
    window_s = filtered.report.config.smoothing.window_s
    warnings: list[str] = []

    speed, t = signals.speed, signals.t
    if not np.any(np.isfinite(speed)):
        return _empty_result(
            signals,
            resolved,
            [
                "The hands were never tracked, so there is no signal to find a swing in. "
                "Check the filtering report: the landmarks may have been gated out, or "
                "the clip's frame rate may be too low for the smoothing window."
            ],
        )

    peak_speed = signals.peak_speed
    if not np.isfinite(signals.torso_length) or signals.torso_length <= 0:
        return _empty_result(
            signals,
            resolved,
            [
                "The subject's torso was never tracked, so there is no scale to judge the "
                "hand motion against and no swing can be confirmed. The capture protocol "
                "asks for the whole body in frame throughout, which is what makes that "
                "scale available."
            ],
        )

    ratio = signals.travel_ratio
    if ratio < resolved.min_travel_ratio:
        return _empty_result(
            signals,
            resolved,
            [
                f"No swing detected. The hands ranged over {ratio:.2f} torso lengths, below "
                f"the {resolved.min_travel_ratio:g} a swing requires. Either the clip "
                "contains no swing, or the hands were not tracked through the part where "
                "one happened."
            ],
        )

    threshold = resolved.moving_fraction * peak_speed

    # --- impact: the clearest feature in the signal, so found first ----------
    impact = _nan_argmax(speed, 0, len(speed))
    assert impact is not None  # noqa: S101 - a finite value exists, checked above

    # --- top: the speed minimum at the highest point before impact ----------
    highest = _nan_argmax(signals.height, 0, impact)
    if highest is None:
        return _empty_result(
            signals,
            resolved,
            [
                "The hands were never tracked before the fastest frame, so no backswing "
                "could be located."
            ],
        )

    search_start, search_stop = _window_bounds(t, highest, resolved.transition_search_s * 2)
    top = _nan_argmin(speed, search_start, min(search_stop, impact))
    if top is None:
        top = highest

    # --- takeaway: the end of the last still stretch before the backswing ---
    #
    # Bounded by the backswing's own speed peak, not by the top. The hands slow
    # almost to rest at the top, so a search bounded there finds the transition
    # pause and reports the takeaway in the middle of the swing. The backswing
    # peak always lies between the real takeaway and that pause, which separates
    # the two without needing to tell them apart.
    #
    # Stillness must also be sustained. A single frame dipping under the
    # threshold is noise, and treating it as the hands having stopped is the
    # same failure in miniature.
    quiet = np.isfinite(speed) & (speed < threshold)
    sustained = [run for run in _runs(quiet) if t[run[1] - 1] - t[run[0]] >= resolved.min_still_s]

    backswing_peak = _nan_argmax(speed, 0, top)
    limit = backswing_peak if backswing_peak is not None else top
    before_backswing = [run for run in sustained if run[0] < limit]
    if before_backswing:
        takeaway = min(before_backswing[-1][1], top)
    else:
        takeaway = int(np.argmax(np.isfinite(speed)))
        warnings.append(
            "The hands are already moving in the first tracked frame, so the takeaway is "
            "reported at the start of the clip rather than located. Recording from a still "
            "address would place it properly."
        )

    # --- finish: the first sustained still stretch after impact -------------
    after_impact = [run for run in sustained if run[0] > impact]
    truncated = not after_impact
    if truncated:
        finish = int(len(speed) - 1 - np.argmax(np.isfinite(speed)[::-1]))
        # Two different things cut a follow-through short, and they call for
        # different fixes: keep recording, or fix the capture so the hands stay
        # trackable. Saying "the clip ended" when the camera was still rolling
        # sends the reader after the wrong one.
        ran_out_of_clip = finish >= len(speed) - 1
        cause = (
            "the clip ended first"
            if ran_out_of_clip
            else f"tracking was lost at {t[finish]:.2f} s, with {t[-1] - t[finish]:.2f} s of "
            "clip still to run"
        )
        warnings.append(
            f"The hands had not come to rest before {cause}, so the finish is reported at "
            "the last tracked frame rather than located. Its confidence is reduced "
            "accordingly."
        )
    else:
        finish = after_impact[0][0]

    backswing_s = float(t[top] - t[takeaway])
    downswing_s = float(t[impact] - t[top])

    # Each half of the swing must itself be real motion rather than noise the
    # rules happened to organise. Without this, a clip that is still except for
    # one brief twitch yields a "top" at whichever still frame noise made
    # highest and a backswing of nearly a second containing no movement at all --
    # a swing in every structural respect except having happened.
    floor = resolved.min_phase_travel_ratio * signals.torso_length
    backswing_travel = _span_travel(signals, takeaway, top)
    downswing_travel = _span_travel(signals, top, impact)
    if not (backswing_travel > floor and downswing_travel > floor):
        return _empty_result(
            signals,
            resolved,
            [
                f"No swing detected. The hands moved {backswing_travel / signals.torso_length:.2f} "
                f"torso lengths from the takeaway to the top and "
                f"{downswing_travel / signals.torso_length:.2f} from there to the fastest "
                f"frame; both must exceed {resolved.min_phase_travel_ratio:g} to be motion "
                "rather than noise. The clip has a burst of movement without a swing's "
                "shape around it.",
                *_tracking_gap_warnings(signals, resolved),
            ],
        )

    if backswing_s < resolved.min_backswing_s or not (
        resolved.min_downswing_s <= downswing_s <= resolved.max_downswing_s
    ):
        if backswing_s < resolved.min_backswing_s:
            fault = (
                f"a backswing of {backswing_s:.2f} s, under the "
                f"{resolved.min_backswing_s:g} s minimum"
            )
        elif downswing_s > resolved.max_downswing_s:
            fault = (
                f"a descent taking {downswing_s:.2f} s, over the "
                f"{resolved.max_downswing_s:g} s a downswing can last -- the club would "
                "fall faster than that unaided, so this is someone lowering it"
            )
        else:
            fault = (
                f"a downswing of {downswing_s:.2f} s, under the "
                f"{resolved.min_downswing_s:g} s minimum"
            )
        return _empty_result(
            signals,
            resolved,
            [
                f"No swing detected. The strongest motion in this clip has {fault}. "
                "Those are not a swing's proportions, so no events are reported.",
                *_slow_motion_warning(signals, resolved, backswing_s, downswing_s),
                *_tracking_gap_warnings(signals, resolved),
            ],
        )

    warnings.extend(_structural_warnings(speed, peak_speed, t, resolved, window_s, downswing_s))
    warnings.extend(_tracking_gap_warnings(signals, resolved))

    events = _build_events(
        signals, resolved, window_s, threshold, takeaway, top, impact, finish, truncated
    )
    return SwingPhases(
        detected=True,
        events=events,
        phases=_build_phases(signals, events, takeaway, top, impact, finish),
        hand=_hand_info(signals),
        frames=len(signals),
        slow_motion_factor=signals.slow_motion_factor,
        config=resolved,
        warnings=warnings,
    )


def _slow_motion_warning(
    signals: SwingSignals, config: PhaseConfig, backswing_s: float, downswing_s: float
) -> list[str]:
    """Say so when a motion has a swing's shape but not a swing's timing.

    Every rule about duration in this module is stated in **real** seconds, and
    reads them off the container's presentation timestamps. That is right for an
    ordinary recording and wrong for a slow-motion one, where the clock runs at
    the playback rate and every phase comes out stretched by the same factor.

    Nothing in a conformed slow-motion file records that factor, so it cannot be
    measured here and is not guessed. What *can* be said is that the shape checks
    passed -- the hands travelled a swing's distance, in a swing's pattern -- and
    only the durations failed, all of them long. That is the signature, and
    reporting it with the smallest factor that would bring the clip inside the
    bounds turns "no swing detected" into something the person holding the camera
    can act on.

    Only ever a suggestion. The factor named is a lower bound derived from the
    configured limit, not an estimate of the clip's real speed, and no metric is
    computed from it.
    """
    if signals.slow_motion_factor != 1.0:
        # Already being reinterpreted; a second suggestion would be noise.
        return []
    if downswing_s <= config.max_downswing_s and backswing_s >= config.min_backswing_s:
        return []
    if downswing_s <= config.max_downswing_s:
        # Too *short*, which slowing the clock down cannot explain.
        return []

    minimum = downswing_s / config.max_downswing_s
    return [
        f"The motion has a swing's shape but not a swing's timing: every phase is long, "
        f"in the same proportion. If this is slow-motion footage, re-run it with a "
        f"slow-motion factor of at least {minimum:.1f} and the durations fall inside the "
        "bounds. Phone slow motion is usually 4x or 8x, from a 120 or 240 fps capture "
        "conformed to 30. Nothing in the file records which, so the factor has to be "
        "supplied rather than measured, and it does change the result: it sets how many "
        "samples the smoothing window holds."
    ]


def _tracking_gap_warnings(signals: SwingSignals, config: PhaseConfig) -> list[str]:
    """Flag a stretch long enough to have hidden the swing.

    Worth its own diagnostic because of how the failure presents. A swing fast
    enough to blur at the capture's shutter speed is a swing the pose estimator
    loses exactly where it matters, and what reaches this layer is not an error
    but a quiet clip with a hole in the middle. Saying "no swing detected"
    without saying "and the hands vanished for half a second right here" points
    the reader at the wrong problem.
    """
    # Keyed on whether the estimator produced a usable observation, not on
    # whether the filter emitted a value. The two differ at the ends of every
    # clip, where the hand was seen perfectly well but the fit had too little
    # support; conflating them would put a "tracking was lost" warning on
    # healthy footage and point the reader at a camera problem that is really a
    # window-width one.
    untracked = ~signals.hand.observed
    longest_s = 0.0
    start_s = 0.0
    for start, stop in _runs(untracked):
        last = min(stop, len(signals.t) - 1)
        span = float(signals.t[last] - signals.t[start])
        if span > longest_s:
            longest_s, start_s = span, float(signals.t[start])

    if longest_s < config.min_tracking_gap_s:
        return []

    return [
        f"The hands were untracked for {longest_s:.2f} s from {start_s:.2f} s -- long enough "
        "to contain an entire downswing. Pose estimation loses a landmark when motion blur "
        "smears it, which happens first at exactly the moment a swing is fastest. A shorter "
        "shutter and a higher frame rate are what make that stretch trackable."
    ]


def _structural_warnings(
    speed: NDArray[np.float64],
    peak_speed: float,
    t: NDArray[np.float64],
    config: PhaseConfig,
    window_s: float,
    downswing_s: float,
) -> list[str]:
    """Facts about the clip that change how much the events are worth."""
    warnings: list[str] = []

    if window_s >= downswing_s / _FULLY_RESOLVED_RATIO:
        warnings.append(
            f"The smoothing window is {window_s:.3f} s and the downswing lasts "
            f"{downswing_s:.3f} s, so the filter averages over "
            f"{window_s / downswing_s:.0%} of the event being measured. Impact timing "
            "and peak speed are both flattened by that; a higher capture frame rate "
            "allows a narrower window."
        )

    # Other swing-like motions in the same clip. Ignoring them would silently
    # analyse whichever happened to be fastest and report it as the swing.
    filled = np.nan_to_num(speed, nan=-1.0)
    minimum_separation = config.min_backswing_s + config.min_downswing_s
    interval = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    distance = max(1, int(minimum_separation / interval)) if interval > 0 else 1
    peaks, _ = find_peaks(filled, prominence=_SECOND_SWING_FRACTION * peak_speed, distance=distance)
    if len(peaks) > 1:
        times = ", ".join(f"{t[index]:.2f} s" for index in peaks)
        warnings.append(
            f"{len(peaks)} swing-like motions were found in this clip (peaks at {times}). "
            "Only the fastest is reported. Trim the clip to one swing, because nothing "
            "downstream can tell which of them a metric refers to."
        )

    return warnings


def _build_events(
    signals: SwingSignals,
    config: PhaseConfig,
    window_s: float,
    threshold: float,
    takeaway: int,
    top: int,
    impact: int,
    finish: int,
    truncated: bool,
) -> list[DetectedEvent]:
    t, speed = signals.t, signals.speed
    span = config.confidence_window_s

    backswing_s = float(t[top] - t[takeaway])
    downswing_s = float(t[impact] - t[top])
    follow_s = float(t[finish] - t[impact])
    address_s = float(t[takeaway] - t[0])

    def confidence(index: int, margin: float, duration_s: float) -> EventConfidence:
        visibility = _visibility_score(signals, index, span)
        resolution = _resolution_score(duration_s, window_s)
        margin = float(np.clip(margin, 0.0, 1.0))
        return EventConfidence(
            overall=float(np.clip(margin * visibility * resolution, 0.0, 1.0)),
            margin=margin,
            visibility=visibility,
            resolution=resolution,
        )

    # --- takeaway ---
    takeaway_margin = _crossing_margin(speed, t, takeaway, threshold, span, rising=True)

    # --- top: how deep the speed dip is, against the motion either side of it.
    # Normalised by the local maximum rather than by the swing's peak, because
    # the question is whether the hands paused here, not how this compares with
    # the downswing they have not yet made.
    dip_start, dip_stop = _window_bounds(t, top, config.transition_search_s * 2)
    left = _nan_argmax(speed, dip_start, top)
    right = _nan_argmax(speed, top + 1, dip_stop)
    # The weaker shoulder of the two, so a dip that is only clear from one side
    # is scored by the side that makes it least convincing.
    surrounding = [speed[i] for i in (left, right) if i is not None and np.isfinite(speed[i])]
    local_max = min(surrounding) if surrounding else float("nan")
    top_margin = (
        (local_max - speed[top]) / local_max if np.isfinite(local_max) and local_max > 0 else 0.0
    )

    # --- impact: prominence of the global peak, against the peak itself ---
    filled = np.nan_to_num(speed, nan=-1.0)
    peaks, properties = find_peaks(filled, prominence=0.0)
    impact_margin = 0.0
    if len(peaks):
        nearest = int(np.argmin(np.abs(peaks - impact)))
        impact_margin = float(properties["prominences"][nearest] / signals.peak_speed)

    # --- impact corroboration: the lowest the hands go after the top ---
    lowest = _nan_argmin(signals.height, top, len(signals.height))
    corroboration = lowest if lowest is not None else None
    delta = float(t[corroboration] - t[impact]) if corroboration is not None else None

    # --- finish ---
    finish_margin = (
        0.0 if truncated else _crossing_margin(speed, t, finish, threshold, span, rising=False)
    )

    return [
        DetectedEvent(
            event=SwingEvent.TAKEAWAY,
            frame_index=takeaway,
            timestamp_s=float(t[takeaway]),
            confidence=confidence(takeaway, takeaway_margin, max(address_s, backswing_s)),
            methodology=(
                "End of the last stretch in which hand speed stayed below "
                f"{config.moving_fraction:.0%} of its peak before the top of the backswing."
            ),
        ),
        DetectedEvent(
            event=SwingEvent.TOP,
            frame_index=top,
            timestamp_s=float(t[top]),
            confidence=confidence(top, top_margin, min(backswing_s, downswing_s)),
            methodology=(
                "Minimum hand speed within "
                f"{config.transition_search_s:g} s of the highest point the hands reached "
                "before impact."
            ),
        ),
        DetectedEvent(
            event=SwingEvent.IMPACT,
            frame_index=impact,
            timestamp_s=float(t[impact]),
            confidence=confidence(impact, impact_margin, downswing_s),
            methodology=(
                "Maximum hand speed. A kinematic estimate: nothing here observes the ball "
                "or the club, and hand speed peaks slightly before the club reaches the "
                "ball, so this runs marginally early. Phases 10 and 11 supply independent "
                "evidence."
            ),
            corroboration_frame=corroboration,
            corroboration_delta_s=delta,
        ),
        DetectedEvent(
            event=SwingEvent.FINISH,
            frame_index=finish,
            timestamp_s=float(t[finish]),
            confidence=confidence(finish, finish_margin, follow_s),
            methodology=(
                "Start of the first stretch after impact in which hand speed stayed below "
                f"{config.moving_fraction:.0%} of its peak."
                + (" Not reached: the clip ended first." if truncated else "")
            ),
        ),
    ]


def _build_phases(
    signals: SwingSignals,
    events: list[DetectedEvent],
    takeaway: int,
    top: int,
    impact: int,
    finish: int,
) -> list[DetectedPhase]:
    """Tile the clip with the four phases the events bound."""
    t = signals.t
    by_event = {entry.event: entry.confidence.overall for entry in events}

    spans = (
        (SwingPhase.ADDRESS, 0, takeaway, by_event[SwingEvent.TAKEAWAY]),
        (
            SwingPhase.BACKSWING,
            takeaway,
            top,
            min(by_event[SwingEvent.TAKEAWAY], by_event[SwingEvent.TOP]),
        ),
        (
            SwingPhase.DOWNSWING,
            top,
            impact,
            min(by_event[SwingEvent.TOP], by_event[SwingEvent.IMPACT]),
        ),
        (
            SwingPhase.FOLLOW_THROUGH,
            impact,
            finish + 1,
            min(by_event[SwingEvent.IMPACT], by_event[SwingEvent.FINISH]),
        ),
    )

    phases: list[DetectedPhase] = []
    for phase, start, stop, confidence in spans:
        if stop <= start:
            # A zero-length phase is not reported as an empty interval: it would
            # read as a measured duration of zero rather than as an absence.
            continue
        last = min(stop, len(t)) - 1
        phases.append(
            DetectedPhase(
                phase=phase,
                start_frame=start,
                end_frame=min(stop, len(t)),
                start_s=float(t[start]),
                end_s=float(t[last]),
                duration_s=float(t[last] - t[start]),
                confidence=confidence,
            )
        )
    return phases
