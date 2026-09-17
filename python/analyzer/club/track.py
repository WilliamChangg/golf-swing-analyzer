"""Choosing one shaft per frame, and refusing where nothing can be chosen.

The detector offers candidates; this decides. It is the only part of the phase
that sees more than one frame, and everything it does that a per-frame detector
could not comes from that.

## Seed at the best evidence, grow outward

A live tracker starts at frame zero and commits. This one runs over a clip that
already exists, so it starts wherever the evidence is strongest -- which on a
golf swing is address or the top, where the club is nearly still and its edges
are sharp -- and grows outward into the downswing, where the club is fastest and
the evidence is worst. Those are the frames worth measuring and they are the ones
a forward-only tracker reaches in the worst condition, having accumulated
whatever it picked up on the way.

This is Phase 4's argument in a different medium. Detection there finds impact
first, because it is the clearest feature in the hand-speed signal, then searches
for the top *before* it and the takeaway *before* that -- each search bounded by
something already located. Here each frame is decided against a neighbour already
decided, and the order the frames are decided in is chosen by the evidence rather
than by the clock.

When a run stops -- the club blurs away, or goes behind the player -- the
remaining frames are seeded again at the best evidence among them and grown the
same way. A clip therefore ends up as several runs rather than one track, and the
gaps between them are the honest shape of the result.

## What the prediction may and may not do

The predicted direction **scores** candidates. It never **supplies** one.

That distinction is the difference between a tracker and a fabricator, and it is
the same rule Phase 3 states about its gap policy: bridging a short absence is
interpolation between observations, and a value produced where there is no
observation is an invention however smooth it looks. A frame whose candidates all
score badly against the prediction is refused, and the refusal leaves a visible
hole. Nothing here ever writes the prediction into the output.

It also has a failure mode worth naming, because it is the one that makes
classical trackers untrustworthy: a search *narrowed* by the current belief finds
what it expects, confirms it, and narrows further. The detector in this package
is never told what the tracker believes, so its candidate list is the same
whether the tracker is right or lost -- and a lost tracker's candidates therefore
still contain the club.

## Three factors, and why margin is measured on the combined score

`support` is the winner's own evidence, `continuity` is how well it agrees with
the prediction, and `margin` is how far it stands above the best alternative.

Margin is computed on `support x continuity`, the same quantity the choice was
made on, and not on support alone. The question it answers is "given everything
known at this frame, was the choice clear" -- and at a frame where the club is
smeared to a support of 0.6 while a door frame behind the player sits at 0.95,
the choice *is* clear, because the door frame is stationary and scores almost
nothing on continuity. Scoring the margin on evidence alone would report that
frame as ambiguous and refuse a correct detection precisely where this phase is
hardest.

Where there is no prediction -- the first frame of a run -- every candidate takes
a continuity of 1.0 and the margin reduces to the evidence margin, which is
right: with no temporal information the choice rests on the image alone. Those
frames are counted as `unanchored`, because a continuity of 1.0 there is absence
of evidence against rather than agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from analyzer.club.geometry import angle_difference_deg, unwrap_deg
from analyzer.contracts.club import (
    ClubConfig,
    ClubFrame,
    ClubImpactEstimate,
    ClubTrackingQuality,
    PhaseCoverage,
    ShaftConfidence,
    ShaftObservation,
    ShaftRefusal,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase, SwingPhases

METHODOLOGY = (
    "One shaft per frame, chosen from the candidate lines a Canny/Hough detector "
    "offered. Confidence is support x margin x continuity: the fraction of the ray "
    "from the hands that lies on an edge, how far the winner stood above the best "
    "alternative on the same score, and how well its direction agreed with the one "
    "predicted from tracked neighbours. Runs are seeded at the frames with the "
    "strongest evidence and grown outward; a prediction scores candidates and never "
    "supplies one, so a frame with nothing acceptable emits nothing."
)

# How many tracked frames must lie after the top before a club-head impact
# estimate is offered at all, and how far from the ends of that set the minimum
# must sit. A minimum at the edge of a searched interval is not a minimum, it is
# where the search stopped -- the same reason Phase 4 scores a finish at zero on
# a clip that ends before the hands do.
_MIN_HEAD_FRAMES_FOR_IMPACT = 5
_IMPACT_INTERIOR_MARGIN = 1

# And how close to the minimum the frames either side of it must actually be.
#
# Interiority in the *observed set* is not enough, which the synthetic swing
# demonstrated rather than argued: at a 180-degree shutter the club head blurs
# away through impact itself, so the frames around the true minimum are missing
# and the lowest *observed* point sits comfortably inside a set whose middle has
# a hole in it. That put impact 29 frames early with nothing in the output
# suggesting a problem. Requiring the neighbours to be within two frames means
# the minimum is located from an arc that was actually sampled near it, and turns
# that case into a refusal.
_IMPACT_MAX_NEIGHBOUR_GAP = 2

# And the phase's own gate, applied to its own derived quantity: an impact
# located from the club is worth nothing if the downswing was not tracked.
#
# Also measured rather than argued. At a 360-degree shutter the club head blurs
# away for the whole of impact, the lowest *observed* head lands in the
# follow-through with two perfectly adjacent neighbours, and both of the checks
# above pass -- putting impact 38 frames late. Nothing local to the minimum can
# see that; only the coverage of the phase the minimum was supposed to fall in.
_MIN_DOWNSWING_COVERAGE_FOR_IMPACT = 0.5

# Percentile of the tracked shaft lengths used as the clip's own estimate of how
# long the club is in the picture. The maximum would be set by a single
# over-long false positive; a high percentile is the same statement made robustly.
_LENGTH_SCALE_PERCENTILE = 95.0


@dataclass(frozen=True)
class CandidateView:
    """One candidate expressed in the frame everything above Phase 3 measures in.

    The pixel tip is carried alongside the frame-widths one because image speed
    is the quantity that explains motion blur, and blur is a property of the
    sensor grid rather than of any normalised frame. Converting it back would
    reintroduce the frame width the reader then has to divide out again.
    """

    tip: tuple[float, float]
    tip_px: tuple[float, float]
    angle_deg: float
    length: float
    length_torso: float
    support: float


@dataclass(frozen=True)
class FrameEvidence:
    """Everything known about one frame before any temporal reasoning.

    `refusal` is the detector's own verdict and is set exactly when `candidates`
    is empty -- no hands to anchor on, no edges in the region, or no line in it
    that could be a club. The tracker adds the three reasons a *candidate* can
    still be rejected, which it alone can see.
    """

    frame_index: int
    timestamp_s: float
    grip: tuple[float, float] | None
    candidates: tuple[CandidateView, ...] = ()
    refusal: ShaftRefusal | None = None


@dataclass
class _Chosen:
    """One accepted frame, before the derived per-frame quantities are filled in."""

    candidate: CandidateView
    support: float
    margin: float
    continuity: float
    runner_up_angle_deg: float | None
    candidates: int
    anchored: bool


@dataclass(frozen=True)
class TrackResult:
    """What the tracker concluded, ready to be assembled into a report."""

    frames: list[ClubFrame]
    quality: ClubTrackingQuality | None
    phase_coverage: list[PhaseCoverage]
    impact: ClubImpactEstimate | None
    refusals: dict[ShaftRefusal, int]
    tracked_frames: int
    unanchored_frames: int
    warnings: list[str] = field(default_factory=list)


def track_shafts(
    evidence: list[FrameEvidence],
    config: ClubConfig,
    *,
    phases: SwingPhases | None = None,
    frame_interval_s: float | None = None,
) -> TrackResult:
    """Decide one shaft per frame, or a named reason there is none."""
    chosen = _grow_runs(evidence, config)
    chosen = _prune_short_runs(chosen, config.min_track_frames)

    times = np.array([entry.timestamp_s for entry in evidence], dtype=np.float64)
    rates = _angular_rates(evidence, chosen, times, config)
    speeds = _tip_speeds(evidence, chosen, times, config)
    head = _head_support(chosen, config)

    frames = _assemble(evidence, chosen, rates, speeds, head, phases, times, config)
    refusals = _count_refusals(frames)
    coverage = _phase_coverage(frames, phases)
    quality = _quality(frames, chosen, head, frame_interval_s)
    impact = _club_head_impact(frames, phases, coverage)

    return TrackResult(
        frames=frames,
        quality=quality,
        phase_coverage=coverage,
        impact=impact,
        refusals=refusals,
        tracked_frames=len(chosen),
        unanchored_frames=sum(1 for entry in chosen.values() if not entry.anchored),
        warnings=_warnings(frames, chosen, coverage, quality),
    )


# --- growth ---------------------------------------------------------------


def _grow_runs(evidence: list[FrameEvidence], config: ClubConfig) -> dict[int, _Chosen]:
    """Seed at the strongest remaining evidence and grow outward, repeatedly.

    Each pass consumes one run. The loop ends when no unassigned frame can be
    seeded, which is when what is left is either empty or too weak to start from
    -- and a seed too weak to start from is exactly a frame whose evidence alone
    does not identify a club, since a seed has no prediction to lean on.
    """
    assigned: dict[int, _Chosen] = {}
    times = [entry.timestamp_s for entry in evidence]

    while True:
        seed = _best_seed(evidence, times, assigned, config)
        if seed is None:
            return assigned
        index, chosen = seed
        assigned[index] = chosen
        for direction in (1, -1):
            _walk(evidence, times, assigned, index, direction, config)


def _best_seed(
    evidence: list[FrameEvidence],
    times: list[float],
    assigned: dict[int, _Chosen],
    config: ClubConfig,
) -> tuple[int, _Chosen] | None:
    """The unassigned, **unreachable** frame whose image alone identifies a club best.

    Unreachable is the load-bearing word, and leaving it out was a real defect
    rather than an inefficiency. A frame with a tracked neighbour inside the
    prediction window has *already* been judged against that neighbour by the
    walk that reached it; if it is still unassigned, that judgement was a
    rejection. Seeding it in a later pass evaluates it with no prediction at all
    -- and a seed is accepted on evidence alone -- so the frame comes back
    holding exactly the candidate the temporal check threw out.

    Caught by `test_an_impossible_rotation_is_rejected_as_a_different_object`,
    where a shaft jumping 157 degrees in one frame at 120 fps was refused as
    `DISCONTINUOUS` by the walk and then re-admitted by the next seed with a
    confidence of 1.0.
    """
    best: tuple[float, int, _Chosen] | None = None
    for position, entry in enumerate(evidence):
        if position in assigned or not entry.candidates:
            continue
        if _has_tracked_neighbour(times, assigned, position, config.max_prediction_window_s):
            continue
        outcome = _evaluate(entry, prediction=None, tolerance_deg=None, config=config)
        if not isinstance(outcome, _Chosen):
            continue
        score = outcome.support * outcome.margin
        if best is None or score > best[0]:
            best = (score, position, outcome)
    if best is None:
        return None
    return best[1], best[2]


def _has_tracked_neighbour(
    times: list[float], assigned: dict[int, _Chosen], position: int, window_s: float
) -> bool:
    """Whether any accepted frame sits close enough in time to have predicted this one."""
    return any(abs(times[position] - times[index]) <= window_s for index in assigned)


def _walk(
    evidence: list[FrameEvidence],
    times: list[float],
    assigned: dict[int, _Chosen],
    start: int,
    direction: int,
    config: ClubConfig,
) -> None:
    """Extend an accepted frame in one direction for as long as a prediction holds.

    A refused frame does not stop the walk: the club goes behind the player for a
    few frames and comes back, and the prediction is still the best thing
    available on the other side. What stops it is time -- once the gap to the last
    *accepted* frame exceeds `max_prediction_window_s`, the prediction is
    extrapolating across most of a downswing, and a frame decided against it
    would be decided against a guess. Those frames are left for a later seed,
    which may reach them from the other side.
    """
    anchors: list[int] = [start]
    position = start + direction

    while 0 <= position < len(evidence):
        if position in assigned:
            return
        last = anchors[-1]
        gap = abs(times[position] - times[last])
        if gap > config.max_prediction_window_s:
            return

        prediction = _predict(times, anchors, assigned, times[position])
        tolerance = min(180.0, config.max_angular_rate_deg_s * max(gap, 1e-9))
        outcome = _evaluate(
            evidence[position],
            prediction=prediction,
            tolerance_deg=tolerance,
            config=config,
            rate_reference=(assigned[last].candidate.angle_deg, gap),
        )
        if isinstance(outcome, _Chosen):
            assigned[position] = outcome
            anchors.append(position)
        position += direction


def _predict(
    times: list[float], anchors: list[int], assigned: dict[int, _Chosen], at_s: float
) -> float:
    """Where the shaft should point, from the two most recent accepted frames.

    Two frames give a rate and the prediction follows it, which matters through
    the downswing: at 120 fps a club turning at 2,000 deg/s moves 17 degrees
    between frames, so a constant-angle prediction would score a correct
    detection at half continuity there. One frame gives a constant angle, which
    is the honest thing to extrapolate from a single observation.
    """
    last = anchors[-1]
    latest = assigned[last].candidate.angle_deg
    if len(anchors) < 2:
        return latest

    previous = anchors[-2]
    span = times[last] - times[previous]
    if abs(span) < 1e-9:
        return latest
    rate = angle_difference_deg(latest, assigned[previous].candidate.angle_deg) / span
    return latest + rate * (at_s - times[last])


def _evaluate(
    entry: FrameEvidence,
    *,
    prediction: float | None,
    tolerance_deg: float | None,
    config: ClubConfig,
    rate_reference: tuple[float, float] | None = None,
) -> _Chosen | ShaftRefusal:
    """Choose among one frame's candidates, or say why none of them will do."""
    if entry.grip is None:
        return ShaftRefusal.NO_GRIP
    if not entry.candidates:
        return entry.refusal or ShaftRefusal.NO_CANDIDATE

    pool = list(entry.candidates)
    if rate_reference is not None:
        # A hard bound rather than a penalty. A candidate implying a rotation
        # faster than a club can turn is not a worse explanation of the same
        # object, it is a different object, and letting it compete on a score
        # would let a strong enough background line outvote the physics.
        reference_angle, gap = rate_reference
        if gap > 0.0:
            pool = [
                candidate
                for candidate in pool
                if abs(angle_difference_deg(candidate.angle_deg, reference_angle)) / gap
                <= config.max_angular_rate_deg_s
            ]
        if not pool:
            return ShaftRefusal.DISCONTINUOUS

    scores = [
        (candidate, _continuity(candidate.angle_deg, prediction, tolerance_deg))
        for candidate in pool
    ]
    ranked = sorted(scores, key=lambda pair: pair[0].support * pair[1], reverse=True)
    winner, continuity = ranked[0]
    best_score = winner.support * continuity

    if best_score <= 0.0:
        return ShaftRefusal.LOW_CONFIDENCE

    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (
        1.0
        if runner_up is None
        else max(0.0, (best_score - runner_up[0].support * runner_up[1]) / best_score)
    )

    if margin < config.min_margin:
        return ShaftRefusal.AMBIGUOUS
    if winner.support < config.min_support:
        return ShaftRefusal.LOW_CONFIDENCE
    if winner.support * margin * continuity < config.min_confidence:
        return ShaftRefusal.LOW_CONFIDENCE

    return _Chosen(
        candidate=winner,
        support=winner.support,
        margin=margin,
        continuity=continuity,
        runner_up_angle_deg=None if runner_up is None else runner_up[0].angle_deg,
        candidates=len(entry.candidates),
        anchored=prediction is not None,
    )


def _continuity(
    angle_deg_value: float, prediction: float | None, tolerance_deg: float | None
) -> float:
    """How well one candidate agrees with the predicted direction.

    Linear in the disagreement and zero at the tolerance, which is the angle the
    shaft could have turned through in the elapsed time at its physical limit. A
    candidate at the limit is therefore scored at zero rather than merely low: it
    is as far from the prediction as anything can be and still be the same object.
    """
    if prediction is None or tolerance_deg is None or tolerance_deg <= 0.0:
        return 1.0
    disagreement = abs(angle_difference_deg(angle_deg_value, prediction))
    return max(0.0, min(1.0, 1.0 - disagreement / tolerance_deg))


def _prune_short_runs(chosen: dict[int, _Chosen], minimum: int) -> dict[int, _Chosen]:
    """Discard runs of fewer than `minimum` consecutive accepted frames.

    One frame that happens to hold a line near the hands is not evidence that
    anything was followed, and nothing in a single frame distinguishes it from a
    lucky background match -- which is the whole reason the tracker exists.
    """
    if minimum <= 1 or not chosen:
        return chosen

    kept: dict[int, _Chosen] = {}
    ordered = sorted(chosen)
    run: list[int] = []
    for position in ordered:
        if run and position == run[-1] + 1:
            run.append(position)
            continue
        if len(run) >= minimum:
            kept.update({index: chosen[index] for index in run})
        run = [position]
    if len(run) >= minimum:
        kept.update({index: chosen[index] for index in run})
    return kept


# --- derived per-frame quantities -----------------------------------------


def _adjacent_pair(chosen: dict[int, _Chosen], position: int) -> tuple[int | None, int | None]:
    """The tracked frames **immediately** either side, if they were tracked too.

    Immediately adjacent rather than nearest-tracked-within-a-window, and the
    difference is not fussiness. A rate taken across a gap has to assume how many
    half-turns the shaft made while nothing was watching, which is exactly the
    assumption `unwrap_deg` refuses to make across a break in its own series --
    and measured on the synthetic swing, taking the nearest tracked neighbour
    across a three-frame hole reported 8,050 deg/s on a club turning at 700.

    So a frame beside a gap gets a one-sided rate, and a frame alone between two
    gaps gets none. Both are the honest answer, and both say so by reporting
    None rather than a number.
    """
    before = position - 1 if (position - 1) in chosen else None
    after = position + 1 if (position + 1) in chosen else None
    return before, after


def _angular_rates(
    evidence: list[FrameEvidence],
    chosen: dict[int, _Chosen],
    times: NDArray[np.float64],
    config: ClubConfig,
) -> dict[int, float]:
    """Rate of change of the shaft direction at each tracked frame.

    Taken from the **unwrapped** angle series. A swing carries the shaft through
    more than a full turn, so differencing the reported angles directly would
    report one 360-degree jump somewhere in the middle of every clip -- and it
    would land in the downswing, which is the only place the rate is interesting.

    Central where both neighbours exist, one-sided where one does. Unlike Phase
    3's derivatives, which come from a fitted polynomial, this is a difference of
    measurements: there is no fit here to take a coefficient from, and saying so
    is better than implying the two are the same kind of number.
    """
    angles = np.full(len(evidence), np.nan, dtype=np.float64)
    for position, entry in chosen.items():
        angles[position] = entry.candidate.angle_deg
    unwrapped = unwrap_deg(angles)

    rates: dict[int, float] = {}
    for position in chosen:
        before, after = _adjacent_pair(chosen, position)
        left = before if before is not None else position
        right = after if after is not None else position
        span = float(times[right] - times[left])
        if right == left or abs(span) < 1e-9:
            continue
        if not (np.isfinite(unwrapped[left]) and np.isfinite(unwrapped[right])):
            continue
        rates[position] = float((unwrapped[right] - unwrapped[left]) / span)
    return rates


def _tip_speeds(
    evidence: list[FrameEvidence],
    chosen: dict[int, _Chosen],
    times: NDArray[np.float64],
    config: ClubConfig,
) -> dict[int, float]:
    """How fast the tip crossed the image, in pixels per second.

    Measured in pixels rather than in frame widths on purpose. This is the number
    that decides whether a frame could have been sharp, and blur happens on the
    sensor grid: a smear is so many pixels long, and the same motion on a 4K
    sensor and a 720p one produces different amounts of it.
    """
    speeds: dict[int, float] = {}
    for position, _entry in chosen.items():
        before, after = _adjacent_pair(chosen, position)
        left = before if before is not None else position
        right = after if after is not None else position
        span = float(times[right] - times[left])
        if right == left or abs(span) < 1e-9:
            continue
        start = chosen[left].candidate.tip_px
        end = chosen[right].candidate.tip_px
        speeds[position] = float(np.hypot(end[0] - start[0], end[1] - start[1]) / abs(span))
    return speeds


def _head_support(chosen: dict[int, _Chosen], config: ClubConfig) -> dict[int, bool]:
    """Which frames' evidence ran all the way to the end of the club.

    Judged against the clip's own longest observed shaft rather than against an
    assumed club length, because no clip records what club was used and the
    *projection* of one varies with where it points. What that makes this is a
    relative statement -- this frame saw as much of the club as the best frame in
    the clip did -- which is the strongest thing one view supports.
    """
    if not chosen:
        return {}
    lengths = np.array([entry.candidate.length_torso for entry in chosen.values()])
    scale = float(np.percentile(lengths, _LENGTH_SCALE_PERCENTILE))
    if not np.isfinite(scale) or scale <= 0.0:
        return dict.fromkeys(chosen, False)
    threshold = config.min_head_length_fraction * scale
    return {
        position: bool(entry.candidate.length_torso >= threshold)
        for position, entry in chosen.items()
    }


def _assemble(
    evidence: list[FrameEvidence],
    chosen: dict[int, _Chosen],
    rates: dict[int, float],
    speeds: dict[int, float],
    head: dict[int, bool],
    phases: SwingPhases | None,
    times: NDArray[np.float64],
    config: ClubConfig,
) -> list[ClubFrame]:
    """One `ClubFrame` per video frame, tracked or refused, in frame order."""
    frames: list[ClubFrame] = []
    for position, entry in enumerate(evidence):
        phase = phases.phase_at(entry.frame_index) if phases is not None else None
        picked = chosen.get(position)
        if picked is None:
            frames.append(
                ClubFrame(
                    frame_index=entry.frame_index,
                    timestamp_s=entry.timestamp_s,
                    detected=False,
                    refusal=_explain(entry, chosen, position, times, config),
                    phase=phase,
                )
            )
            continue

        assert entry.grip is not None  # noqa: S101 - a chosen frame has an anchor by construction
        candidate = picked.candidate
        frames.append(
            ClubFrame(
                frame_index=entry.frame_index,
                timestamp_s=entry.timestamp_s,
                detected=True,
                phase=phase,
                shaft=ShaftObservation(
                    grip_x=entry.grip[0],
                    grip_y=entry.grip[1],
                    tip_x=candidate.tip[0],
                    tip_y=candidate.tip[1],
                    angle_deg=candidate.angle_deg,
                    length=candidate.length,
                    length_torso=candidate.length_torso,
                    reaches_head=head.get(position, False),
                    confidence=ShaftConfidence(
                        overall=picked.support * picked.margin * picked.continuity,
                        support=picked.support,
                        margin=picked.margin,
                        continuity=picked.continuity,
                    ),
                    runner_up_angle_deg=picked.runner_up_angle_deg,
                    candidates=picked.candidates,
                    angular_rate_deg_s=rates.get(position),
                    tip_speed_px_s=speeds.get(position),
                ),
            )
        )
    return frames


def _explain(
    entry: FrameEvidence,
    chosen: dict[int, _Chosen],
    position: int,
    times: NDArray[np.float64],
    config: ClubConfig,
) -> ShaftRefusal:
    """Why one untracked frame carries nothing.

    Re-evaluated here rather than recorded during growth, and the reason is that
    growth may visit a frame more than once -- from the left, and again from the
    right after a later seed -- and the interesting answer is the one that holds
    against the track as it finally stands. A frame refused from a run that was
    itself later pruned should not be explained by that run.

    It re-evaluates against the **same** prediction and the same hard rate bound
    the walk applied, which is what makes the reason the real one. An earlier
    version explained every frame with a 180-degree tolerance and no rate bound,
    and reported a shaft that had jumped a third of a turn in one frame as
    `LOW_CONFIDENCE` -- true, in that its score was low, and useless next to
    "whatever this is, it is not the object the neighbouring frames were
    following".

    A frame that was accepted and then pruned for sitting in too short a run is
    `ISOLATED`, which is a different fact from anything the evaluation can
    report: nothing was wrong with it except that nothing around it agreed.
    """
    if entry.grip is None:
        return ShaftRefusal.NO_GRIP
    if not entry.candidates:
        return entry.refusal or ShaftRefusal.NO_CANDIDATE

    nearest = min(chosen, key=lambda index: abs(index - position), default=None)
    gap = 0.0 if nearest is None else abs(float(times[position] - times[nearest]))
    if nearest is None or gap > config.max_prediction_window_s:
        outcome = _evaluate(entry, prediction=None, tolerance_deg=None, config=config)
    else:
        reference = chosen[nearest].candidate.angle_deg
        outcome = _evaluate(
            entry,
            prediction=reference,
            tolerance_deg=min(180.0, config.max_angular_rate_deg_s * max(gap, 1e-9)),
            config=config,
            rate_reference=(reference, gap),
        )
    return outcome if isinstance(outcome, ShaftRefusal) else ShaftRefusal.ISOLATED


def _count_refusals(frames: list[ClubFrame]) -> dict[ShaftRefusal, int]:
    counts: dict[ShaftRefusal, int] = {}
    for frame in frames:
        if frame.refusal is not None:
            counts[frame.refusal] = counts.get(frame.refusal, 0) + 1
    return counts


# --- summaries ------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _phase_coverage(frames: list[ClubFrame], phases: SwingPhases | None) -> list[PhaseCoverage]:
    """Coverage per swing phase. Empty when no swing was detected.

    **The gate**, and the reason `ClubTrackingReport.coverage` is documented as
    the number not to read on its own. Address and the follow-through are most of
    a clip and the club is nearly still through both, so the clip-wide rate is an
    average over the easy frames.
    """
    if phases is None or not phases.detected:
        return []

    by_index = {frame.frame_index: frame for frame in frames}
    coverage: list[PhaseCoverage] = []
    for interval in phases.phases:
        span = [
            by_index[index]
            for index in range(interval.start_frame, interval.end_frame)
            if index in by_index
        ]
        tracked = [frame for frame in span if frame.shaft is not None]
        confidences = [frame.shaft.confidence.overall for frame in tracked if frame.shaft]
        speeds = [
            frame.shaft.tip_speed_px_s
            for frame in tracked
            if frame.shaft and frame.shaft.tip_speed_px_s is not None
        ]
        coverage.append(
            PhaseCoverage(
                phase=interval.phase,
                frames=len(span),
                tracked=len(tracked),
                coverage=len(tracked) / len(span) if span else 0.0,
                median_confidence=_median(confidences),
                median_tip_speed_px_s=_median(speeds),
            )
        )
    return coverage


def _quality(
    frames: list[ClubFrame],
    chosen: dict[int, _Chosen],
    head: dict[int, bool],
    frame_interval_s: float | None,
) -> ClubTrackingQuality | None:
    """The fit, the check and the capture, each reported separately."""
    shafts = [frame.shaft for frame in frames if frame.shaft is not None]
    if not shafts:
        return None

    rates = [abs(s.angular_rate_deg_s) for s in shafts if s.angular_rate_deg_s is not None]
    speeds = [s.tip_speed_px_s for s in shafts if s.tip_speed_px_s is not None]
    lengths = [s.length_torso for s in shafts]
    head_frames = sum(1 for value in head.values() if value)
    max_speed = max(speeds) if speeds else None

    return ClubTrackingQuality(
        median_support=_median([s.confidence.support for s in shafts]),
        median_margin=_median([s.confidence.margin for s in shafts]),
        median_continuity=_median([s.confidence.continuity for s in shafts]),
        median_confidence=_median([s.confidence.overall for s in shafts]),
        median_angular_rate_deg_s=_median(rates),
        max_angular_rate_deg_s=max(rates) if rates else None,
        median_tip_speed_px_s=_median(speeds),
        max_tip_speed_px_s=max_speed,
        # The exposure cannot be longer than the interval between frames, so this
        # is the smear at a 360-degree shutter: the worst any camera does, and an
        # upper bound on what this clip's club head actually carried. Nothing in
        # the file records the exposure, so it is the most that can be said.
        max_blur_px=(
            None if max_speed is None or frame_interval_s is None else max_speed * frame_interval_s
        ),
        median_length_torso=_median(lengths),
        max_length_torso=max(lengths) if lengths else None,
        head_frames=head_frames,
        head_fraction=head_frames / len(chosen) if chosen else 0.0,
        methodology=METHODOLOGY,
    )


def _club_head_impact(
    frames: list[ClubFrame],
    phases: SwingPhases | None,
    coverage: list[PhaseCoverage],
) -> ClubImpactEstimate | None:
    """The lowest point the club head reached after the top, where it was observed.

    Independent of Phase 4 in the way that matters: Phase 4 reads the *hands*,
    and the hands reach their lowest point and their highest speed before the
    club head arrives at the ball. This reads the head.

    Emitted only from frames where the head was genuinely observed -- which are
    the blurriest frames in the clip, so this refuses far more often than it
    answers -- and only when the minimum sits strictly inside the observed set. A
    minimum at the edge of an interval is where the search stopped rather than
    where the club was lowest, which is the same reason Phase 4 scores a finish
    at zero on a clip that ends before the hands do.
    """
    if phases is None or not phases.detected:
        return None
    top = phases.event(SwingEvent.TOP)
    if top is None:
        return None

    downswing = next((entry for entry in coverage if entry.phase is SwingPhase.DOWNSWING), None)
    if downswing is None or downswing.coverage < _MIN_DOWNSWING_COVERAGE_FOR_IMPACT:
        return None

    observed = [
        frame
        for frame in frames
        if frame.shaft is not None
        and frame.shaft.reaches_head
        and frame.frame_index > top.frame_index
    ]
    if len(observed) < _MIN_HEAD_FRAMES_FOR_IMPACT:
        return None

    heights = [frame.shaft.tip_y for frame in observed if frame.shaft]
    lowest = int(np.argmin(heights))
    if not (_IMPACT_INTERIOR_MARGIN <= lowest < len(observed) - _IMPACT_INTERIOR_MARGIN):
        return None

    at = observed[lowest]
    before = at.frame_index - observed[lowest - 1].frame_index
    after = observed[lowest + 1].frame_index - at.frame_index
    if max(before, after) > _IMPACT_MAX_NEIGHBOUR_GAP:
        return None

    kinematic = phases.event(SwingEvent.IMPACT)
    return ClubImpactEstimate(
        frame_index=at.frame_index,
        timestamp_s=at.timestamp_s,
        methodology=(
            "Lowest observed club-head position after the top, over the frames whose "
            "edge evidence ran to the end of the club. Independent of the hand-speed "
            "peak Phase 4 uses, and deliberately does not replace it."
        ),
        head_frames_searched=len(observed),
        kinematic_frame=None if kinematic is None else kinematic.frame_index,
        delta_s=None if kinematic is None else at.timestamp_s - kinematic.timestamp_s,
    )


def _warnings(
    frames: list[ClubFrame],
    chosen: dict[int, _Chosen],
    coverage: list[PhaseCoverage],
    quality: ClubTrackingQuality | None,
) -> list[str]:
    """What a reader should know before believing any of the numbers above."""
    notes: list[str] = []
    if not chosen:
        return notes

    downswing = next((entry for entry in coverage if entry.phase is SwingPhase.DOWNSWING), None)
    if downswing is not None and downswing.coverage < 0.5:
        notes.append(
            f"The club was tracked in {downswing.tracked} of {downswing.frames} downswing "
            f"frames ({downswing.coverage:.0%}). The clip-wide coverage is not a substitute "
            "for this: address and the follow-through are most of a clip and the club is "
            "nearly still through both, so a high overall rate can contain no downswing at all."
        )

    if quality is not None and quality.max_blur_px is not None and quality.max_blur_px > 20.0:
        notes.append(
            f"At its fastest the club head crossed {quality.max_tip_speed_px_s:.0f} px/s, which "
            f"is up to {quality.max_blur_px:.0f} px of smear in one exposure at a 360-degree "
            "shutter. Nothing in a video file records the shutter, so that is an upper bound "
            "rather than a measurement -- a shutter n times faster than the frame interval "
            "divides it by n. `data/README.md` asks for 1/1000 s for this reason."
        )

    unanchored = sum(1 for entry in chosen.values() if not entry.anchored)
    if unanchored > 1:
        notes.append(
            f"{unanchored} tracked frames had no tracked neighbour to be predicted from, so "
            "their continuity factor is 1.0 by absence of evidence rather than by agreement. "
            "Each is the start of a separate run; a track in many pieces is weaker evidence "
            "than the same number of frames in one."
        )

    if quality is not None and quality.head_fraction < 0.5:
        notes.append(
            f"The evidence reached the end of the club in {quality.head_fraction:.0%} of "
            "tracked frames, so most observations are a shaft *direction* with no club-head "
            "position. Two things shorten a segment and one view cannot separate them: the "
            "head was smeared, or the club was pointing at the camera."
        )

    return notes
