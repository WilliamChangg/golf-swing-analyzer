"""Deciding which candidate is the ball, and the frame at which it stops being there.

The detector offers candidates; this decides. It is the only part of the phase
that sees more than one frame, and everything it does that a per-frame detector
could not comes from that -- which here is everything, because neither question
this phase asks exists inside a single frame. Which blob is the ball is not
visible in a still picture, and "the frame the ball left" is not a property of any
frame at all.

## Establish, then watch

Two stages, and they answer different questions from different evidence.

**Establish.** Over the frames before the top of the backswing -- where the ball
is still, unoccluded and has been sitting for longer than any event in the clip
lasts -- every candidate position from every frame is clustered. A cluster is a
place that repeatedly held something ball-shaped. The ball is one of them; so are
a tee marker, a white shoe that did not move, a bright mat seam and a daisy.

**Watch.** Each surviving cluster is then followed through the whole clip and
asked whether it empties. Exactly one thing in a golf frame is a small still
round object that disappears once and never returns, so that is the question that
picks the ball out of the stationary set.

This is Phase 10's argument in a different medium, again. Its tracker seeds where
the evidence is strongest and grows outward rather than starting at frame zero and
committing; this identifies from the stretch where the evidence is strongest and
only then looks at the stretch where the measurement is. In both cases the point
is that an offline pass over a clip that already exists is not obliged to make its
decisions in the order the frames arrive.

## The identification is circular, and the circle is closed rather than hidden

Picking the ball partly by the fact that it left, and then reporting the instant
it left, is an argument with itself if the confidence in that instant is also
scored on the departure.

So the evidence is partitioned. `BallConfidence` scores one frame's observation
from the image at that frame and the position agreed across the clip -- contrast,
a co-located competitor, drift -- and reads nothing about any departure.
`DepartureConfidence` scores the instant from what happened either side of it --
how much of the lead-in carried a ball, how strong the final sightings were,
whether the absence lasted -- and reads nothing about how much the tracker liked
its candidate. Neither can flatter the other.

What is left over is the one thing a circular identification genuinely cannot
check, and it is reported rather than solved: if *two* stationary candidates
depart, the clip cannot say which was the ball. That is
`EstablishedBall.margin`'s job, and a low value there is the failure mode that
looks perfect from every frame's point of view.

## Where the departure is taken from

Not the last frame carrying a ball, which a single false positive in the
follow-through moves by a hundred frames. The end of the **last run of presence
long enough to be a track** -- `BallConfig.min_establishment_frames` -- which a
one-frame speck cannot be, and which correctly prefers the later of two runs when
the ball was occluded partway through and came back.

Permanence is then a real check rather than a tautology: had the departure been
defined as the last sighting, nothing could ever come back after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from analyzer.contracts.ball import (
    BallConfidence,
    BallConfig,
    BallDeparture,
    BallFrame,
    BallObservation,
    BallQuality,
    BallRefusal,
    DepartureConfidence,
    EstablishedBall,
)
from analyzer.contracts.phases import SwingEvent, SwingPhases

METHODOLOGY = (
    "One ball per frame, chosen from the candidate regions a top-hat detector "
    "offered. Confidence is contrast x margin x stillness: how far the region stood "
    "from the ring around it, how far it stood above any other candidate in the same "
    "place, and how close it sat to the position established across the clip. The "
    "position is established by clustering candidates over the frames before the top "
    "of the backswing and keeping the cluster that later empties."
)

ESTABLISHMENT_METHODOLOGY = (
    "Candidate positions from the frames before the top of the backswing, clustered "
    "within the drift bound, ranked by how many frames agreed and how strongly. Among "
    "clusters meeting the establishment bounds, the one that departs permanently is "
    "the ball; where none departs, the best-established is reported and no instant is."
)

DEPARTURE_METHODOLOGY = (
    "The frame after the end of the last run of ball observations long enough to be a "
    "track. Impact lies between that run's final frame and this one -- the ball was "
    "present at one and absent at the other -- so the interval between them is a "
    "bracket rather than a deviation, and it is one frame interval wide whatever the "
    "footage."
)

# How far a candidate may sit from a cluster's running centre and still join it,
# as a multiple of `BallConfig.max_drift_torso`.
#
# One would make clustering and the per-frame drift check the same bound, which
# sounds tidy and loses balls: the drift check measures against a settled position
# and this measures against a centre that starts at whichever candidate happened
# to arrive first. Slightly looser here, and the settled position is then the
# median of what gathered -- which is what the per-frame check uses.
_CLUSTER_RADIUS_FACTOR = 1.5

# Clusters kept before any of them is followed through the clip. A cluttered
# driving range offers a great many stationary bright specks; the ones past this
# are the ones fewest frames agreed on, which the establishment bounds would
# reject anyway.
_MAX_CLUSTERS = 16

# How much of the way towards the latest sighting the drift reference moves each
# time the ball is seen. Small: the thing being followed is a camera drifting over
# hundreds of frames, not a ball moving, and a large weight would let the
# detector's own centroid noise walk the reference off the ball. See `_follow`.
_DRIFT_FOLLOW_WEIGHT = 0.2

# Below this, the ball was dimming before it went and the reported instant runs
# early. See the measured table where it is used.
_ABRUPTNESS_WARNING = 0.85


@dataclass(frozen=True)
class BallCandidateView:
    """One candidate expressed in the frame everything above Phase 3 measures in.

    The pixel radius is carried alongside the frame-widths one because whether a
    ball can be assessed for roundness at all is a fact about the sensor grid,
    the way blur is in Phase 10. Converting it back would reintroduce the frame
    width a reader then has to divide out again.
    """

    x: float
    y: float
    radius: float
    radius_px: float
    radius_torso: float
    contrast: float
    circularity: float


@dataclass(frozen=True)
class BallEvidence:
    """Everything known about one frame before any temporal reasoning.

    `refusal` is the detector's own verdict and is set exactly when `candidates`
    is empty -- no hands to anchor on, a region outside the picture, or nothing in
    it of a ball's size and shape. The tracker adds the reasons a *candidate* can
    still be rejected, and the one reason that is not a rejection at all: `GONE`.
    """

    frame_index: int
    timestamp_s: float
    candidates: tuple[BallCandidateView, ...] = ()
    refusal: BallRefusal | None = None


@dataclass
class _Cluster:
    """A place that repeatedly held something ball-shaped."""

    x: float
    y: float
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    contrasts: list[float] = field(default_factory=list)

    def add(self, frame_index: int, candidate: BallCandidateView) -> None:
        self.frames.append(frame_index)
        self.xs.append(candidate.x)
        self.ys.append(candidate.y)
        self.radii.append(candidate.radius)
        self.contrasts.append(candidate.contrast)
        # A running mean, so a cluster that starts on the rim of a ball migrates
        # onto it rather than anchoring wherever the first candidate landed.
        self.x = float(np.mean(self.xs))
        self.y = float(np.mean(self.ys))

    @property
    def score(self) -> float:
        """How good a candidate for "the ball" this position is, before departure.

        Frames first and contrast second, deliberately in that order of
        magnitude: a hundred agreeing frames is a far stronger statement about a
        stationary object than a bright one, and a bright thing seen six times is
        the failure this is guarding against.
        """
        return len(self.frames) * float(np.median(self.contrasts))


@dataclass(frozen=True)
class BallTrackResult:
    """What the tracker concluded, ready to be assembled into a report."""

    frames: list[BallFrame]
    established: EstablishedBall | None
    departure: BallDeparture | None
    quality: BallQuality | None
    refusals: dict[BallRefusal, int]
    observed_frames: int
    pre_departure_coverage: float
    warnings: list[str] = field(default_factory=list)


def track_ball(
    evidence: list[BallEvidence],
    config: BallConfig,
    *,
    phases: SwingPhases | None = None,
    torso_length: float = 1.0,
) -> BallTrackResult:
    """Identify the ball across a clip and locate the frame it stopped being there.

    `torso_length` is in **frame widths**, and converting `max_drift_torso` with
    it once here is not a convenience. Every position in this module is in frame
    widths and every bound in `BallConfig` is in torso lengths, so comparing one
    against the other directly is wrong by the factor between them -- about four
    on ordinary framing, which is loose enough to merge a teed ball with a tee
    marker beside it into a single object that then never departs. That is the
    mistake `analyzer/coordinates.py` exists to make impossible for coordinate
    frames, and it has the same shape one level up in units.
    """
    warnings: list[str] = []
    drift = config.max_drift_torso * (torso_length if torso_length > 0 else 1.0)
    established, runner_up, departure_index = _establish(evidence, config, drift, phases, warnings)

    if established is None:
        # No position in this clip held something ball-shaped often enough to be
        # called a ball. Every frame is refused with whatever the detector said
        # about it, which is what a reader needs in order to tell a framing
        # problem from a contrast one.
        refused = [_refused(entry, entry.refusal or BallRefusal.NO_CANDIDATE) for entry in evidence]
        return BallTrackResult(
            frames=refused,
            established=None,
            departure=None,
            quality=None,
            refusals=_count_refusals(refused),
            observed_frames=0,
            pre_departure_coverage=0.0,
            warnings=warnings,
        )

    chosen = _assign(evidence, established, config, drift)
    departure = _departure(evidence, chosen, config, phases, warnings, established, drift)

    frames = _assemble(evidence, chosen, departure, phases)
    observed = sum(1 for entry in frames if entry.detected)
    return BallTrackResult(
        frames=frames,
        established=_established_model(
            established, runner_up, torso_length, departed=departure_index is not None
        ),
        departure=departure,
        quality=_quality(frames, chosen, established, torso_length),
        refusals=_count_refusals(frames),
        observed_frames=observed,
        pre_departure_coverage=_pre_departure_coverage(frames, departure),
        warnings=warnings + _warnings(established, departure, config, drift),
    )


# --- establishment --------------------------------------------------------


@dataclass(frozen=True)
class _Established:
    """The agreed ball position, in frame widths, and what backs it."""

    x: float
    y: float
    radius: float
    frames: int
    searched: int
    spread: float
    median_contrast: float
    score: float


def _establishment_span(
    evidence: list[BallEvidence], phases: SwingPhases | None
) -> tuple[int, int]:
    """Which frames the identification is made from.

    Everything before the top of the backswing: address and the backswing, where
    the ball is still, in plain view, and the club is nowhere near it. That is
    hundreds of frames of the easiest evidence in the clip, and using the whole of
    it rather than address alone matters on a clip that starts with the player
    already taking the club back.

    Falls back to the whole clip when no swing was detected. The frames after
    impact then dilute the agreement, which is the correct direction for the
    error to run: it lowers the establishment coverage and can only make the
    identification refuse, never make it confident about the wrong thing.
    """
    if phases is not None and phases.detected:
        top = phases.event(SwingEvent.TOP)
        if top is not None and top.frame_index > 0:
            return 0, min(top.frame_index, len(evidence))
    return 0, len(evidence)


def _cluster(evidence: list[BallEvidence], span: tuple[int, int], drift: float) -> list[_Cluster]:
    """Group candidate positions across frames into places that kept holding one."""
    radius = _CLUSTER_RADIUS_FACTOR * drift
    clusters: list[_Cluster] = []

    for entry in evidence[span[0] : span[1]]:
        for candidate in entry.candidates:
            nearest: _Cluster | None = None
            best = radius
            for cluster in clusters:
                distance = math.hypot(candidate.x - cluster.x, candidate.y - cluster.y)
                if distance <= best:
                    nearest, best = cluster, distance
            if nearest is None:
                clusters.append(_Cluster(x=candidate.x, y=candidate.y))
                nearest = clusters[-1]
            # One candidate per frame per cluster. Without this a ball and its
            # shadow, landing in the same cluster, would count that frame twice
            # and inflate the agreement the identification rests on.
            if nearest.frames and nearest.frames[-1] == entry.frame_index:
                continue
            nearest.add(entry.frame_index, candidate)

    clusters.sort(key=lambda item: item.score, reverse=True)
    return clusters[:_MAX_CLUSTERS]


def _qualifies(cluster: _Cluster, searched: int, config: BallConfig) -> bool:
    """Whether a cluster has enough behind it to be called a stationary object."""
    if len(cluster.frames) < config.min_establishment_frames:
        return False
    return searched > 0 and len(cluster.frames) / searched >= config.min_establishment_coverage


def _establish(
    evidence: list[BallEvidence],
    config: BallConfig,
    drift: float,
    phases: SwingPhases | None,
    warnings: list[str],
) -> tuple[_Established | None, _Established | None, int | None]:
    """Decide which position is the ball, and whether it ever emptied.

    Returns the chosen position, the best rival, and the index at which the
    chosen one departed -- the last of which is only used to record *that* it
    departed. The instant itself is located afterwards, from the per-frame
    assignment rather than from this pass, so that the frame reported is the one a
    reader can check against the frames in the report.
    """
    start, stop = _establishment_span(evidence, phases)
    searched = max(0, stop - start)
    if searched == 0:
        return None, None, None

    clusters = _cluster(evidence, (start, stop), drift)
    qualified = [cluster for cluster in clusters if _qualifies(cluster, searched, config)]
    if not qualified:
        return None, None, None

    # Among the stationary objects, the ball is the one that leaves **at a moment
    # a ball can be struck at**. Checked here on presence alone -- a full per-frame
    # assignment for every rival would cost the same work several times over, and
    # the question at this stage is only which cluster to spend it on.
    #
    # The window is part of the identification and not a veto applied afterwards,
    # which is how it was written first and which fails on real footage rather
    # than in principle. On `data/face-on/rory_face_on.mp4` three stationary
    # candidates empty and stay empty; the best-supported of them goes at frame
    # 193, in the middle of the backswing, where it is something the player's own
    # body moved in front of. Choosing on support and then vetoing the winner
    # threw the clip away while the real ball sat in the candidate list, departing
    # at the right moment, never considered.
    departing = [
        (cluster, index)
        for cluster, index in (
            (item, _departure_index(evidence, item, config, drift)) for item in qualified
        )
        if index is not None and _within_strike_window(index, phases)
    ]

    if departing:
        departing.sort(key=lambda pair: pair[0].score, reverse=True)
        chosen, departure_index = departing[0]
        if len(departing) > 1:
            other = departing[1][0]
            warnings.append(
                f"{len(departing)} stationary candidates in this clip emptied and stayed empty. "
                f"The one reported sat {_distance(chosen, other):.2f} frame widths from the next, "
                "which was agreed on by "
                f"{len(other.frames)} frames against {len(chosen.frames)}. Only one of them is the "
                "ball and nothing in a single view says which; the established margin is what "
                "that uncertainty is reported as."
            )
    else:
        # Nothing departed. A practice swing, a clip that ends before impact, or a
        # ball the detector never lost sight of. The identification still stands
        # and no instant is reported from it.
        chosen, departure_index = qualified[0], None

    # The margin is measured against what the choice was actually made over, which
    # is the departing clusters when any departed and every qualified one when
    # none did.
    #
    # Measuring it against all qualified clusters either way reads wrong in the
    # ordinary case and reads wrong in the safe direction, which is worse than it
    # sounds. A tee marker sitting beside the ball for the whole clip is agreed on
    # by exactly as many frames as the ball, so it ties on score and drives the
    # margin to zero -- reporting the identification as a coin toss when the ball
    # was in fact singled out unambiguously by being the only thing that left.
    # A margin that says "ambiguous" on every clip with a second white object in
    # it is a margin nobody reads, and then it is not there for the case that
    # matters: two objects that *both* leave, which one view genuinely cannot
    # resolve.
    pool = [cluster for cluster, _ in departing] if departing else qualified
    rivals = [cluster for cluster in pool if cluster is not chosen]
    runner_up = max(rivals, key=lambda item: item.score) if rivals else None
    return (
        _to_established(chosen, searched),
        _to_established(runner_up, searched) if runner_up is not None else None,
        departure_index,
    )


def _distance(a: _Cluster, b: _Cluster) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _to_established(cluster: _Cluster, searched: int) -> _Established:
    """A cluster settled into a position, taken as a median rather than a mean.

    The mean the cluster maintained while gathering is right for deciding what
    joins it and wrong for the answer: one candidate that landed on a shadow's
    edge pulls a mean by its whole offset and a median not at all. The position
    every drift check is measured against should be the one most observations
    actually agreed on.
    """
    x = float(np.median(cluster.xs))
    y = float(np.median(cluster.ys))
    offsets = [math.hypot(px - x, py - y) for px, py in zip(cluster.xs, cluster.ys, strict=True)]
    return _Established(
        x=x,
        y=y,
        radius=float(np.median(cluster.radii)),
        frames=len(cluster.frames),
        searched=searched,
        spread=float(np.median(offsets)) if offsets else 0.0,
        median_contrast=float(np.median(cluster.contrasts)),
        score=cluster.score,
    )


def _established_model(
    chosen: _Established,
    runner_up: _Established | None,
    torso_length: float,
    *,
    departed: bool,
) -> EstablishedBall:
    scale = torso_length if torso_length > 0 else 1.0
    margin = 1.0
    separation: float | None = None
    if runner_up is not None and chosen.score > 0.0:
        margin = float(np.clip((chosen.score - runner_up.score) / chosen.score, 0.0, 1.0))
        separation = math.hypot(chosen.x - runner_up.x, chosen.y - runner_up.y) / scale

    return EstablishedBall(
        x=chosen.x,
        y=chosen.y,
        radius=chosen.radius,
        radius_torso=chosen.radius / scale,
        frames=chosen.frames,
        searched_frames=max(1, chosen.searched),
        spread_torso=chosen.spread / scale,
        median_contrast=float(np.clip(chosen.median_contrast, 0.0, 1.0)),
        margin=margin,
        runner_up_distance_torso=separation,
        departed=departed,
        methodology=ESTABLISHMENT_METHODOLOGY,
    )


# --- per-frame assignment -------------------------------------------------


@dataclass(frozen=True)
class _Chosen:
    """One accepted frame's observation, before the report is assembled."""

    candidate: BallCandidateView
    contrast: float
    margin: float
    stillness: float
    runner_up_distance: float | None
    candidates: int


def _nearest(entry: BallEvidence, reference: tuple[float, float]) -> list[BallCandidateView]:
    """This frame's candidates, nearest to `reference` first."""
    return sorted(
        entry.candidates,
        key=lambda candidate: math.hypot(candidate.x - reference[0], candidate.y - reference[1]),
    )


def _follow(reference: tuple[float, float], candidate: BallCandidateView) -> tuple[float, float]:
    """Move the reference a little of the way towards where the ball was just seen.

    An exponential average rather than a jump to the candidate: a jump makes the
    reference a random walk driven by the detector's own centroid noise, and over
    hundreds of frames a random walk can step off the ball onto whatever is beside
    it. A slow follow tracks a drifting camera and averages the noise away.
    """
    weight = _DRIFT_FOLLOW_WEIGHT
    return (
        reference[0] + weight * (candidate.x - reference[0]),
        reference[1] + weight * (candidate.y - reference[1]),
    )


def _assign(
    evidence: list[BallEvidence],
    established: _Established,
    config: BallConfig,
    drift: float,
) -> dict[int, _Chosen | BallRefusal]:
    """Decide, for every frame, whether the ball is still where it was.

    The choice is made on **position**: the candidate nearest to where the ball
    was last seen, which is the one thing known for certain about a teed ball.
    Scoring runs afterwards and can only reject what position chose -- it never
    picks a different candidate, because a stronger blob elsewhere in the region
    is by construction not the object this clip identified.

    ## The reference follows the ball; it is not the clip's own median

    "A teed ball does not move" is a statement about the **world**, and this
    layer works in the **image**. In the image a stationary ball moves whenever
    the camera does, and consumer footage moves: handheld, digitally stabilised,
    or a repost with a slow pan across it.

    Measured on `data/face-on/rory_face_on.mp4`, where the camera drifts gently
    through the swing. Against a fixed reference the ball's offset grows
    monotonically from 11 px at frame 327 to 14 px at frame 360 while its contrast
    stays healthy at 0.52-0.62 throughout -- so `stillness` decays to 0.22, the
    product falls under the confidence bound, and the run ends **58 frames before
    the ball actually goes**. The reported instant would have been that early,
    with nothing in the numbers looking wrong: contrast good, margin 1.00, and a
    plausible confidence.

    So the reference follows. `established` picks *which* object; from there the
    per-frame question is how far the ball moved since it was last seen, which is
    Phase 10's `continuity` argument in a different medium -- agreement with a
    tracked neighbour rather than with a global prior. `BallQuality.drift_torso`
    reports how far the whole thing travelled, so a camera that moved is a
    measurement rather than a silent early answer.
    """
    decided: dict[int, _Chosen | BallRefusal] = {}
    reference = (established.x, established.y)

    for entry in evidence:
        if entry.refusal is not None:
            decided[entry.frame_index] = entry.refusal
            continue
        if not entry.candidates:
            decided[entry.frame_index] = BallRefusal.NO_CANDIDATE
            continue

        ranked = _nearest(entry, reference)
        best = ranked[0]
        offset = math.hypot(best.x - reference[0], best.y - reference[1])
        if offset > drift:
            decided[entry.frame_index] = BallRefusal.MOVED
            continue

        # The margin's competitor is the best other candidate in the *same place*.
        # See `BallConfidence.margin`: a candidate elsewhere in the region is not
        # a competitor, because the choice was never between them.
        rivals = [
            candidate
            for candidate in ranked[1:]
            if math.hypot(candidate.x - reference[0], candidate.y - reference[1]) <= drift
        ]
        rival = max(rivals, key=lambda candidate: candidate.contrast) if rivals else None
        margin = 1.0
        separation: float | None = None
        if rival is not None and best.contrast > 0.0:
            margin = float(np.clip((best.contrast - rival.contrast) / best.contrast, 0.0, 1.0))
            separation = math.hypot(best.x - rival.x, best.y - rival.y)

        if margin < config.min_margin:
            decided[entry.frame_index] = BallRefusal.AMBIGUOUS
            continue

        stillness = float(np.clip(1.0 - offset / drift, 0.0, 1.0))
        contrast = float(np.clip(best.contrast, 0.0, 1.0))
        if contrast * margin * stillness < config.min_confidence:
            decided[entry.frame_index] = BallRefusal.LOW_CONFIDENCE
            continue

        decided[entry.frame_index] = _Chosen(
            candidate=best,
            contrast=contrast,
            margin=margin,
            stillness=stillness,
            runner_up_distance=separation,
            candidates=len(entry.candidates),
        )
        reference = _follow(reference, best)

    return decided


# --- departure ------------------------------------------------------------


def _presence(evidence: list[BallEvidence], cluster: _Cluster, drift: float) -> list[bool]:
    """Whether the cluster's position held a candidate at each frame.

    Presence only, with none of the scoring the per-frame assignment applies.
    This runs for every qualifying cluster in order to decide which one to spend
    the assignment on, so it answers the cheap half of the question.

    The reference follows the candidate exactly as `_assign`'s does, and for the
    same reason: a fixed one turns a drifting camera into a departure. Keeping the
    two consistent matters more than it looks -- this pass decides which cluster is
    the ball, and if it saw a departure the assignment does not, the clip would
    identify on one thing and measure another.
    """
    reference = (cluster.x, cluster.y)
    seen: list[bool] = []
    for entry in evidence:
        ranked = _nearest(entry, reference) if entry.candidates else []
        if ranked and math.hypot(ranked[0].x - reference[0], ranked[0].y - reference[1]) <= drift:
            seen.append(True)
            reference = _follow(reference, ranked[0])
        else:
            seen.append(False)
    return seen


def _runs(mask: list[bool]) -> list[tuple[int, int]]:
    """Maximal half-open [start, stop) runs of True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _last_track_run(mask: list[bool], config: BallConfig) -> tuple[int, int] | None:
    """The last run of presence long enough to be a track.

    The **last** rather than the longest: a ball occluded partway through address
    and seen again afterwards produces two runs, and the departure belongs to the
    later one. Long enough to be a track excludes the one-frame false positive in
    the follow-through that would otherwise move the answer by a hundred frames.
    """
    qualifying = [run for run in _runs(mask) if run[1] - run[0] >= config.min_establishment_frames]
    return qualifying[-1] if qualifying else None


def _departure_index(
    evidence: list[BallEvidence], cluster: _Cluster, config: BallConfig, drift: float
) -> int | None:
    """The first absent frame after the last track run, or None if it never emptied."""
    mask = _presence(evidence, cluster, drift)
    run = _last_track_run(mask, config)
    if run is None or run[1] >= len(mask):
        return None
    return run[1]


def _departure(
    evidence: list[BallEvidence],
    chosen: dict[int, _Chosen | BallRefusal],
    config: BallConfig,
    phases: SwingPhases | None,
    warnings: list[str],
    established: _Established,
    drift: float,
) -> BallDeparture | None:
    """Locate the instant the ball stopped being visible, and score it.

    Built from the per-frame assignment rather than from raw presence, so the
    frame reported is the one the report's own frames agree with. A reader
    scrubbing to `last_seen_frame` sees a `BallFrame` carrying an observation, and
    to `first_absent_frame` one carrying a refusal.
    """
    mask = [isinstance(chosen.get(entry.frame_index), _Chosen) for entry in evidence]
    run = _last_track_run(mask, config)
    if run is None:
        return None
    if run[1] >= len(evidence):
        # The ball was still there when the clip ended. Correct, and not a
        # failure: nothing was struck inside this recording.
        return None

    last_seen, first_absent = run[1] - 1, run[1]
    last_entry, absent_entry = evidence[last_seen], evidence[first_absent]
    interval = absent_entry.timestamp_s - last_entry.timestamp_s
    if not math.isfinite(interval) or interval <= 0.0:
        warnings.append(
            f"The frames bracketing the ball's departure carry timestamps "
            f"{last_entry.timestamp_s:.4f} s and {absent_entry.timestamp_s:.4f} s, which do not "
            "advance. No instant is reported from them: the bracket that makes this "
            "measurement worth anything would be zero or negative wide."
        )
        return None

    phase = phases.phase_at(first_absent) if phases is not None and phases.detected else None
    if not _gated(first_absent, phases, warnings):
        return None

    confidence = _departure_confidence(evidence, chosen, run, config, established, drift)
    kinematic = phases.event(SwingEvent.IMPACT) if phases is not None and phases.detected else None
    return BallDeparture(
        frame_index=first_absent,
        timestamp_s=float(absent_entry.timestamp_s),
        last_seen_frame=last_seen,
        last_seen_s=float(last_entry.timestamp_s),
        first_absent_frame=first_absent,
        first_absent_s=float(absent_entry.timestamp_s),
        interval_s=float(interval),
        phase=phase,
        confidence=confidence,
        methodology=DEPARTURE_METHODOLOGY,
        kinematic_frame=None if kinematic is None else kinematic.frame_index,
        delta_s=None
        if kinematic is None
        else float(absent_entry.timestamp_s - kinematic.timestamp_s),
    )


def _within_strike_window(first_absent: int, phases: SwingPhases | None) -> bool:
    """Whether a departure at this frame could be a strike at all.

    **Structural, not a golf norm.** The statement is only that a ball cannot be
    struck before the club has started down or after the hands have come to rest,
    which bounds the departure between the top of the backswing and the finish.
    Anything outside that is a ball that rolled off a tee, blew away, was picked
    up, or was hidden by the player's own body on the way past -- all of which
    happen, and none of which is an impact.

    Nothing tighter is asserted, because tighter would mean claiming how long a
    downswing takes, and that is a golf norm needing the labelled set Phase 12
    builds. How far the departure sits from the kinematic estimate is measured
    instead, one layer up, where it is a *comparison* rather than a rule.

    True when there is no swing to check against, because the alternative is to
    discard an observation for want of a corroboration it never needed. The
    report says it went unchecked.
    """
    if phases is None or not phases.detected:
        return True
    top = phases.event(SwingEvent.TOP)
    finish = phases.event(SwingEvent.FINISH)
    if top is None or finish is None:
        return True
    return top.frame_index < first_absent <= finish.frame_index


def _gated(first_absent: int, phases: SwingPhases | None, warnings: list[str]) -> bool:
    """`_within_strike_window`, with the reason written down when it refuses.

    Applied a second time, to the departure derived from the per-frame assignment
    rather than from the cheap presence pass the identification used. The two can
    differ by a frame or two -- scoring rejects frames presence accepts -- and the
    instant that reaches a reader is this one, so this is the one that has to be
    inside the window.
    """
    if phases is None or not phases.detected:
        warnings.append(
            "No swing was detected in this clip, so the ball's departure could not be checked "
            "against where a strike can happen. A ball that rolls off a tee, or is picked up, "
            "departs exactly like one that is hit, and only the swing around it tells them "
            "apart. The instant is reported ungated."
        )
        return True

    top = phases.event(SwingEvent.TOP)
    finish = phases.event(SwingEvent.FINISH)
    if top is None or finish is None:
        return True

    if first_absent <= top.frame_index:
        warnings.append(
            f"The ball stopped being visible at frame {first_absent}, at or before the top of "
            f"the backswing (frame {top.frame_index}). A ball cannot be struck before the club "
            "starts down, so no impact is reported from it -- the ball was moved, rolled, or "
            "obscured by something crossing the tee."
        )
        return False
    if first_absent > finish.frame_index:
        warnings.append(
            f"The ball stopped being visible at frame {first_absent}, after the hands came to "
            f"rest (frame {finish.frame_index}). That is not an impact, so none is reported "
            "from it."
        )
        return False
    return True


def _departure_confidence(
    evidence: list[BallEvidence],
    chosen: dict[int, _Chosen | BallRefusal],
    run: tuple[int, int],
    config: BallConfig,
    established: _Established,
    drift: float,
) -> DepartureConfidence:
    """Score the instant from what happened either side of it.

    Reads nothing the per-frame confidence reads. See the module docstring on why
    the two are kept apart.
    """
    last_seen, first_absent = run[1] - 1, run[1]

    # Establishment: how much of the lead-in carried a ball at all.
    before = evidence[: last_seen + 1]
    seen = sum(1 for entry in before if isinstance(chosen.get(entry.frame_index), _Chosen))
    establishment = seen / len(before) if before else 0.0

    # Abruptness: the final sightings against the ball's own typical strength. A
    # ball being progressively covered dims first; a struck one does not.
    observations = [value.contrast for value in chosen.values() if isinstance(value, _Chosen)]
    typical = float(np.median(observations)) if observations else 0.0
    tail = [
        value.contrast
        for index in range(max(run[0], last_seen - config.abruptness_frames + 1), last_seen + 1)
        if isinstance(value := chosen.get(index), _Chosen)
    ]
    abruptness = float(np.clip(np.mean(tail) / typical, 0.0, 1.0)) if tail and typical > 0 else 0.0

    # Permanence: how much of the window the check asks for the clip both provided
    # and kept empty.
    #
    # Two decisions here, and both were made the other way first.
    #
    # The denominator comes from the clip's own frame interval and not from the
    # frames that happen to exist. A clip ending two frames after the departure
    # must score low; dividing by what is there would score it 1.0, which is the
    # check reporting a pass on a test it never ran.
    #
    # "Empty" means **the image offered no candidate there**, not "the tracker
    # accepted none". Those differ in exactly the case this factor exists to
    # catch. A ball being progressively covered gets dimmer until it falls below
    # the confidence bound, which ends the run a few frames early -- and then every
    # frame after the departure is one the tracker rejected, so an acceptance-based
    # permanence reads 1.0 and the abruptness factor, measuring only the healthy
    # frames before, reads high too. Measured on the fixture: a ball faded over
    # four frames located impact two frames early with both factors comfortable.
    # The dim ball is still in the picture and still produces a candidate, so
    # reading the candidates rather than the verdicts is what notices.
    needed = _expected_window_frames(evidence, config.permanence_window_s)
    empty = sum(
        1
        for entry in evidence[first_absent : first_absent + needed]
        if not any(
            math.hypot(candidate.x - established.x, candidate.y - established.y) <= drift
            for candidate in entry.candidates
        )
    )
    permanence = float(np.clip(empty / needed, 0.0, 1.0))

    return DepartureConfidence(
        overall=float(np.clip(establishment * abruptness * permanence, 0.0, 1.0)),
        establishment=float(np.clip(establishment, 0.0, 1.0)),
        abruptness=abruptness,
        permanence=permanence,
    )


def _expected_window_frames(evidence: list[BallEvidence], window_s: float) -> int:
    """How many frames the permanence window asks for, whether the clip has them."""
    times = np.array([entry.timestamp_s for entry in evidence], dtype=np.float64)
    if times.size < 2:
        return 1
    intervals = np.diff(times)
    finite = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if not finite.size:
        return 1
    return max(1, round(window_s / float(np.median(finite))))


# --- assembly -------------------------------------------------------------


def _refused(entry: BallEvidence, reason: BallRefusal) -> BallFrame:
    return BallFrame(
        frame_index=entry.frame_index,
        timestamp_s=entry.timestamp_s,
        detected=False,
        refusal=reason,
    )


def _assemble(
    evidence: list[BallEvidence],
    chosen: dict[int, _Chosen | BallRefusal],
    departure: BallDeparture | None,
    phases: SwingPhases | None,
) -> list[BallFrame]:
    """One `BallFrame` per video frame, observed or refused.

    Frames at or after a located departure carry `GONE` rather than whatever the
    detector said about them. That is not a relabelling of a failure: the ball is
    correctly absent there, and reporting `NO_CANDIDATE` would put the clip's
    largest block of refusals under a reason that means "the search failed".
    """
    frames: list[BallFrame] = []
    for entry in evidence:
        decision = chosen.get(entry.frame_index)
        phase = (
            phases.phase_at(entry.frame_index) if phases is not None and phases.detected else None
        )

        if departure is not None and entry.frame_index >= departure.first_absent_frame:
            frames.append(
                BallFrame(
                    frame_index=entry.frame_index,
                    timestamp_s=entry.timestamp_s,
                    detected=False,
                    refusal=BallRefusal.GONE,
                    phase=phase,
                )
            )
            continue

        if not isinstance(decision, _Chosen):
            frames.append(
                BallFrame(
                    frame_index=entry.frame_index,
                    timestamp_s=entry.timestamp_s,
                    detected=False,
                    refusal=decision
                    if isinstance(decision, BallRefusal)
                    else BallRefusal.NO_CANDIDATE,
                    phase=phase,
                )
            )
            continue

        frames.append(
            BallFrame(
                frame_index=entry.frame_index,
                timestamp_s=entry.timestamp_s,
                detected=True,
                ball=BallObservation(
                    x=decision.candidate.x,
                    y=decision.candidate.y,
                    radius=decision.candidate.radius,
                    radius_torso=decision.candidate.radius_torso,
                    confidence=BallConfidence(
                        overall=float(
                            np.clip(
                                decision.contrast * decision.margin * decision.stillness, 0.0, 1.0
                            )
                        ),
                        contrast=decision.contrast,
                        margin=decision.margin,
                        stillness=decision.stillness,
                    ),
                    runner_up_distance_torso=decision.runner_up_distance,
                    candidates=decision.candidates,
                ),
                phase=phase,
            )
        )
    return frames


def _count_refusals(frames: list[BallFrame]) -> dict[BallRefusal, int]:
    counts: dict[BallRefusal, int] = {}
    for frame in frames:
        if frame.refusal is not None:
            counts[frame.refusal] = counts.get(frame.refusal, 0) + 1
    return counts


def _pre_departure_coverage(frames: list[BallFrame], departure: BallDeparture | None) -> float:
    """Fraction of the frames before the departure that carried a ball.

    The number to read. A clip-wide rate counts every frame of the follow-through
    against a ball that is correctly no longer there, which would make a perfect
    result on a long clip look like a poor one.
    """
    limit = departure.first_absent_frame if departure is not None else len(frames)
    span = frames[:limit]
    if not span:
        return 0.0
    return sum(1 for frame in span if frame.detected) / len(span)


def _quality(
    frames: list[BallFrame],
    chosen: dict[int, _Chosen | BallRefusal],
    established: _Established,
    torso_length: float,
) -> BallQuality | None:
    observed = [frame.ball for frame in frames if frame.ball is not None]
    if not observed:
        return None

    # The one number here that is about the camera rather than about the track,
    # and the reason `BallCandidateView` carries a pixel radius at all: whether a
    # ball has any shape left to be round is a fact about the sensor grid, and no
    # normalised frame can express it.
    radii_px = [
        value.candidate.radius_px for value in chosen.values() if isinstance(value, _Chosen)
    ]

    # How far the ball's *picture* moved between its first and last sighting. A
    # teed ball does not move, so this measures the camera. See
    # `BallQuality.drift_torso`.
    positions = [(frame.ball.x, frame.ball.y) for frame in frames if frame.ball is not None]
    drift = (
        math.hypot(positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1])
        if len(positions) > 1
        else 0.0
    )

    scale = torso_length if torso_length > 0 else 1.0
    return BallQuality(
        median_contrast=float(np.median([ball.confidence.contrast for ball in observed])),
        median_margin=float(np.median([ball.confidence.margin for ball in observed])),
        median_stillness=float(np.median([ball.confidence.stillness for ball in observed])),
        median_confidence=float(np.median([ball.confidence.overall for ball in observed])),
        radius_px=float(np.median(radii_px)) if radii_px else None,
        spread_torso=established.spread / scale,
        drift_torso=drift / scale,
        methodology=METHODOLOGY,
    )


def _warnings(
    established: _Established,
    departure: BallDeparture | None,
    config: BallConfig,
    drift: float,
) -> list[str]:
    """What a reader should know before believing any of the numbers above."""
    notes: list[str] = []

    if departure is None:
        # Deliberately not "the ball never stopped being visible", which is only
        # one of the ways to get here and would contradict the gate's own warning
        # on a clip where something departed at the wrong moment.
        notes.append(
            "A ball was identified and nothing stopped being visible at a moment a ball "
            "can be struck at, so no impact is reported from it. A practice swing, a clip "
            "that ends before contact, a swing that missed, and a ball moved before the "
            "downswing all produce exactly this; where one of them can be told apart, a "
            "warning above says which."
        )
        return notes

    confidence = departure.confidence
    if confidence.permanence < 1.0:
        notes.append(
            f"The ball's absence was verified over {confidence.permanence:.0%} of the "
            f"{config.permanence_window_s:g} s this check asks for. Three things do that and "
            "they need different fixes: something crossed the tee and the ball came back, so "
            "this is not an impact; the clip ended too soon to tell, so recording two more "
            "seconds of follow-through settles it; or the ball was still faintly in the "
            "picture after the instant reported, which means it was being covered rather than "
            "struck and the instant runs early by however long the covering took."
        )

    # Measured on the fixture, fading the ball over N frames before removing it:
    #
    #   fade  0     2     4     6     8     12
    #   abr   1.00  0.91  0.85  0.80  0.73  0.66
    #   err   0     -1    -2    -3    -4    -6   frames early
    #
    # A clean departure scores exactly 1.00, and the instant runs early by half
    # the fade. The bound sits just under the two-frame row, which is about where
    # an error stops being a rounding of the bracket and starts being a bias.
    if confidence.abruptness < _ABRUPTNESS_WARNING:
        notes.append(
            f"The ball was seen at {confidence.abruptness:.0%} of its usual strength in the "
            "frames just before it went, rather than at full strength. A struck ball is as "
            "visible in its last frame as in its first; a dimming one is being covered by "
            "something, most often the club head crossing the line of sight on a "
            "down-the-line view. The instant may run a frame or two early."
        )

    if confidence.establishment < 0.5:
        notes.append(
            f"The ball was found in {confidence.establishment:.0%} of the frames before it "
            "went. A departure is only as good as the presence it interrupts, and below about "
            "half the clip has found something that flickers rather than something that sat "
            "there."
        )

    if established.spread > drift / 2.0:
        notes.append(
            f"The established position scattered by {established.spread:.3f} frame widths, "
            "which is a large fraction of the drift bound. A teed ball scatters by the "
            "detector's own noise and nothing else, so this suggests the clip settled on "
            "something that moves."
        )

    return notes
