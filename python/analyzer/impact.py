"""Reconciling the impact estimates the phases produce into one reported instant.

Sits above `phases`, `club` and `ball` and belongs to none of them, which is why
it is a module here rather than inside any of the three. Each of those layers
keeps its own estimate unchanged -- `SwingPhases` still reports the hand-speed
peak, `ClubTrackingReport` still reports the club-head minimum -- and this is the
single place that says which to believe for a given clip and by how much the
others disagreed.

The argument is in `contracts/impact.py`. In short: the four estimates are not
averaged, because three of them are biased and one is not, and averaging an
unbiased observation with a biased proxy moves the answer away from the truth
while producing a provenance nobody can name. They are ranked, and the
disagreements are kept as measurements rather than discarded as noise.

## What this module is allowed to do to Phase 4

Nothing. `detect_phases` returns what it returned before this phase existed, and
a caller that does not ask for a fusion sees no change. That is deliberate and it
is the same decision `contracts/club.py` recorded when it refused to let the
club-head estimate overwrite the kinematic one: an engine in which a number
changes depending on which other analyses happened to run is an engine whose
outputs cannot be compared across clips.
"""

from __future__ import annotations

import numpy as np

from analyzer.contracts.ball import BallTrackingReport
from analyzer.contracts.club import ClubTrackingReport
from analyzer.contracts.impact import FusedImpact, ImpactCandidate, ImpactSource
from analyzer.contracts.phases import SwingEvent, SwingPhases

# Precedence. An observation beats a measurement of a quantity that coincides
# with the event; that beats a proxy; and among proxies, the one with no known
# bias beats the one with a known one.
#
# HAND_LOW above HAND_SPEED is the one step here that is a **measurement rather
# than an argument**, and it is the question `docs/ROADMAP.md` left open after the
# tour-pro footage arrived: on the one clip in this project with an observable
# impact, the lowest point of the hand arc landed within 4 frames of it and stayed
# there across every assumed slow-motion factor, while peak hand speed landed 17
# to 40 frames away depending on the factor. `scripts/benchmark_ball.py --sweep
# agreement` measures the same comparison against a rendered ball, and this order
# follows what it reports.
#
# Phase 4's own primary estimate is unchanged by this and stays the hand-speed
# peak; see the module docstring on why nothing here reaches back down.
_PRECEDENCE = (
    ImpactSource.BALL_DEPARTURE,
    ImpactSource.CLUB_HEAD,
    ImpactSource.HAND_LOW,
    ImpactSource.HAND_SPEED,
)

# Fraction of the downswing's own duration beyond which two estimates are not
# disagreeing about the same event. Stated against the clip's own downswing
# rather than in seconds, because a disagreement of 40 ms is nothing on a 24 fps
# recording and most of the event on a 240 fps one.
_DISAGREEMENT_FRACTION = 0.5


def fuse_impact(
    phases: SwingPhases | None = None,
    *,
    ball: BallTrackingReport | None = None,
    club: ClubTrackingReport | None = None,
    smoothing_window_s: float | None = None,
    frame_interval_s: float | None = None,
) -> FusedImpact | None:
    """Reconcile every impact estimate available for one clip into one instant.

    Returns None when nothing located an impact at all, which is the correct
    outcome for a clip with no swing in it and for a practice swing -- neither of
    which has an impact to report, and neither of which should be given one at
    frame zero.
    """
    candidates = _candidates(phases, ball, club, smoothing_window_s, frame_interval_s)
    if not candidates:
        return None

    order = {source: rank for rank, source in enumerate(_PRECEDENCE)}
    chosen = min(candidates, key=lambda entry: order[entry.source])

    resolved = [
        candidate.model_copy(
            update={
                "delta_s": candidate.timestamp_s - chosen.timestamp_s,
                "delta_frames": candidate.frame_index - chosen.frame_index,
            }
        )
        for candidate in candidates
    ]
    deltas = [abs(entry.delta_s) for entry in resolved if entry.delta_s is not None]

    return FusedImpact(
        frame_index=chosen.frame_index,
        timestamp_s=chosen.timestamp_s,
        source=chosen.source,
        observed=chosen.source is ImpactSource.BALL_DEPARTURE,
        uncertainty_s=chosen.uncertainty_s,
        uncertainty_is_bracket=chosen.source is ImpactSource.BALL_DEPARTURE,
        confidence=chosen.confidence,
        methodology=chosen.methodology,
        candidates=sorted(resolved, key=lambda entry: order[entry.source]),
        agreement_s=max(deltas) if len(deltas) > 1 else None,
        warnings=_warnings(resolved, chosen, phases),
    )


def _candidates(
    phases: SwingPhases | None,
    ball: BallTrackingReport | None,
    club: ClubTrackingReport | None,
    smoothing_window_s: float | None,
    frame_interval_s: float | None,
) -> list[ImpactCandidate]:
    """One entry per source that produced an answer for this clip."""
    found: list[ImpactCandidate] = []

    if ball is not None and ball.departure is not None:
        departure = ball.departure
        found.append(
            ImpactCandidate(
                source=ImpactSource.BALL_DEPARTURE,
                frame_index=departure.frame_index,
                timestamp_s=departure.timestamp_s,
                # The one error bar in this list that is a bracket. The ball was
                # present at one end of it and absent at the other, so impact is
                # inside; see `FusedImpact.uncertainty_is_bracket`.
                uncertainty_s=departure.interval_s,
                confidence=departure.confidence.overall,
                methodology=departure.methodology,
            )
        )

    if club is not None and club.impact is not None:
        found.append(
            ImpactCandidate(
                source=ImpactSource.CLUB_HEAD,
                frame_index=club.impact.frame_index,
                timestamp_s=club.impact.timestamp_s,
                # The interval between the frames the arc was sampled at. A floor
                # rather than a bound: the minimum of a sampled arc can sit
                # anywhere between the samples either side of it, and Phase 10
                # requires those to be close but does not report how close.
                uncertainty_s=frame_interval_s,
                # `ClubImpactEstimate` carries no confidence of its own, and the
                # track's median shaft confidence is not it -- that scores the
                # frames, not the minimum read off them. Absent rather than
                # approximated.
                confidence=None,
                methodology=club.impact.methodology,
            )
        )

    if phases is not None and phases.detected:
        impact = phases.event(SwingEvent.IMPACT)
        if impact is not None:
            if impact.corroboration_frame is not None and impact.corroboration_delta_s is not None:
                found.append(
                    ImpactCandidate(
                        source=ImpactSource.HAND_LOW,
                        frame_index=impact.corroboration_frame,
                        timestamp_s=impact.timestamp_s + impact.corroboration_delta_s,
                        uncertainty_s=smoothing_window_s,
                        confidence=None,
                        methodology=(
                            "Lowest point the hands reached after the top, from the same "
                            "filtered trajectory Phase 4 reads. A proxy: the hands are at the "
                            "bottom of their arc at impact, but nothing here sees the ball."
                        ),
                    )
                )
            found.append(
                ImpactCandidate(
                    source=ImpactSource.HAND_SPEED,
                    frame_index=impact.frame_index,
                    timestamp_s=impact.timestamp_s,
                    # The width of the smoothing window, which is the scale on
                    # which a peak located in a smoothed signal could have moved.
                    # Not a bracket: nothing guarantees the true peak is inside it.
                    uncertainty_s=smoothing_window_s,
                    confidence=impact.confidence.overall,
                    methodology=impact.methodology,
                )
            )

    return found


def _warnings(
    candidates: list[ImpactCandidate], chosen: ImpactCandidate, phases: SwingPhases | None
) -> list[str]:
    """What a reader should know before believing the reported instant."""
    notes: list[str] = []

    if chosen.source is not ImpactSource.BALL_DEPARTURE:
        notes.append(
            "No ball was observed leaving in this clip, so the reported impact is inferred "
            "from the player's motion rather than seen. Phase 4's own methodology says the "
            "hands peak before the club reaches the ball; nothing here can say by how much "
            "on this clip, because the measurement that would is the one that is missing."
        )

    downswing = _downswing_s(phases)
    if downswing is not None and downswing > 0.0:
        limit = _DISAGREEMENT_FRACTION * downswing
        far = [
            (entry, entry.delta_s)
            for entry in candidates
            if entry.delta_s is not None and abs(entry.delta_s) > limit
        ]
        for entry, delta_s in far:
            notes.append(
                f"The {entry.source.value} estimate sits {delta_s * 1000:+.0f} ms from the "
                f"reported instant, which is more than half the {downswing * 1000:.0f} ms "
                "downswing. Two estimates that far apart are not disagreeing about one event: "
                "one of them is measuring something else, and which one is not decidable from "
                "this clip alone."
            )

    hand_speed = next(
        (entry for entry in candidates if entry.source is ImpactSource.HAND_SPEED), None
    )
    if (
        chosen.source is ImpactSource.BALL_DEPARTURE
        and hand_speed is not None
        and hand_speed.delta_s is not None
    ):
        # Not a warning about this clip so much as the measurement this phase
        # exists to make possible. Stated plainly, because it is the number that
        # turns Phase 4's documented caveat into a quantity.
        direction = "early" if hand_speed.delta_s < 0 else "late"
        notes.append(
            f"Measured against the observed departure, peak hand speed runs "
            f"{abs(hand_speed.delta_s) * 1000:.0f} ms {direction} on this clip "
            f"({abs(hand_speed.delta_frames or 0)} frames). That is one clip and not a "
            "correction; a correction needs the labelled set Phase 12 builds."
        )

    return notes


def _downswing_s(phases: SwingPhases | None) -> float | None:
    """How long the downswing lasted, as the scale a disagreement is judged against."""
    if phases is None or not phases.detected:
        return None
    top = phases.event(SwingEvent.TOP)
    impact = phases.event(SwingEvent.IMPACT)
    if top is None or impact is None:
        return None
    duration = impact.timestamp_s - top.timestamp_s
    return float(duration) if np.isfinite(duration) and duration > 0 else None
