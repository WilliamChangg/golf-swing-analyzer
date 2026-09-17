"""Aligning two clips: the entry point, and the honesty that goes with it.

Everything below this module produces one piece of evidence. This one decides
which pieces exist for a given pair, picks the map, checks it against whatever
did not produce it, and refuses when the result is too poor to report.

## What it will not tell you

**Nothing here can determine that the two clips show the same swing.** It aligns
swing-shaped signals; handed two different swings by the same player it will find
the alignment that fits them best and report it. The one thing it cannot do is
make their phase durations agree -- two swings of different tempo cannot be
brought into correspondence by any offset and any clock rate -- so a mismatch
survives as residual. That residual is the only evidence available on the
question, which is why it is reported as a number rather than folded into a
verdict, and why a pair with no spare degrees of freedom scores zero on
agreement rather than scoring well on nothing.

## Which estimator supplies what, and why it is not the obvious split

Both are computed whenever both can be, and each supplies the quantity it has
the evidence for.

**The offset comes from the correlation.** This was not the original design and
the benchmark overturned it. Four swing events look like four clocks and are
not: perturb the landmarks at the noise level Phase 3 measured from real
footage, and the top stays within 8 ms while the takeaway wanders 542 ms, because
one is an extremum and the other a threshold crossing on a signal that is barely
moving there. A correlation over the same pair averages several hundred samples
and recovers the offset to under a millisecond, against 8-31 ms for the anchored
fit. Choosing the better-supported estimator of one quantity is not averaging two
estimates -- which this layer refuses to do -- and the numbers are in
`scripts/benchmark_sync.py`.

**The rate and the residual come from the events.** A correlation yields one lag
and can produce neither. Where a rate *is* significant the events fit stands
whole, because substituting a single lag into a map whose offset varies with time
would introduce the very bias `SyncQuality.method_disagreement_ms` exists to
report.

A person's picks override both, because a person looking at two frames is not
making an estimate the engine should be arbitrating -- they are supplying the
answer the engine is meant to be accounting for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.sync import (
    AnchorResidual,
    ClipRef,
    CorrelationReport,
    OverlapReport,
    SyncAnchor,
    SyncConfidence,
    SyncConfig,
    SyncMethod,
    SyncModel,
    SyncQuality,
    TimeMap,
)
from analyzer.phases.signals import SwingSignals
from analyzer.sync.anchors import ManualPick, event_anchors, manual_anchors, reject_outliers
from analyzer.sync.correlate import SpeedTrack, cross_correlate
from analyzer.sync.timemap import fit_time_map, quantisation_floor_s, residuals_against

# A correlation peak this close to its best rival is not a decision. Two speed
# humps per swing means a rival always exists; what matters is the gap.
_AMBIGUOUS_PEAK_MARGIN = 0.10

# Beyond this the event fit and the correlation are not describing the same
# alignment, and saying so is more useful than reporting the one that won. Half
# a frame at 10 fps, and about a fifth of a downswing: below it the two methods
# are quibbling, above it they disagree about which feature they are on.
_METHOD_DISAGREEMENT_WARN_S = 0.050

_SAME_SWING_CAVEAT = (
    "Synchronisation aligns two swing-shaped signals; it cannot tell that both cameras "
    "filmed the same swing. The residual is the only evidence on that question, because "
    "two swings of different tempo cannot be aligned by any offset and any clock rate."
)


@dataclass(frozen=True)
class SyncInput:
    """One clip, as the sync layer needs to see it.

    Paths and pose files are resolved by the caller. Taking signals rather than a
    path keeps this package free of storage concerns and lets every test drive it
    with synthetic trajectories, which is the only way the known-offset
    measurements in `scripts/benchmark_sync.py` are possible at all.
    """

    path: Path
    signals: SwingSignals
    phases: SwingPhases
    notes: tuple[str, ...] = ()
    """What the filter said about this clip. Carried through because the most
    common reason a clip contributes nothing here is Phase 3's frame-rate floor,
    and its report already names the shortest window this clip's rate supports --
    which is the fix, and would otherwise be two layers away from the failure."""

    def clip_ref(self) -> ClipRef:
        times = self.signals.t
        intervals = np.diff(times) if times.size > 1 else np.array([], dtype=np.float64)
        return ClipRef(
            path=str(self.path),
            name=self.path.name,
            frames=int(times.size),
            start_s=float(times[0]) if times.size else 0.0,
            duration_s=float(times[-1] - times[0]) if times.size > 1 else 0.0,
            median_interval_s=float(np.median(intervals)) if intervals.size else float("nan"),
            slow_motion_factor=self.signals.slow_motion_factor,
        )

    def speed_track(self) -> SpeedTrack:
        return SpeedTrack(t=self.signals.t, speed=self.signals.speed)


def _overlap(reference: ClipRef, target: ClipRef, time_map: TimeMap) -> OverlapReport:
    """What the two clips have in common, stated in reference time."""
    reference_end = reference.start_s + reference.duration_s
    target_start = time_map.to_reference(target.start_s)
    target_end = time_map.to_reference(target.start_s + target.duration_s)

    start = max(reference.start_s, target_start)
    end = min(reference_end, target_end)
    duration = max(0.0, end - start)

    target_span = abs(time_map.to_target(end) - time_map.to_target(start)) if duration else 0.0
    return OverlapReport(
        start_s=start,
        end_s=max(start, end),
        duration_s=duration,
        reference_fraction=(
            float(np.clip(duration / reference.duration_s, 0.0, 1.0))
            if reference.duration_s > 0
            else 0.0
        ),
        target_fraction=(
            float(np.clip(target_span / target.duration_s, 0.0, 1.0))
            if target.duration_s > 0
            else 0.0
        ),
    )


def _agreement(discrepancy_s: float | None, floor_s: float) -> float:
    """How well independent evidence agrees, scaled against the quantisation floor.

    1.0 when the evidence agrees as closely as the frame rates permit, halving
    for every doubling beyond that. Scaled against the floor rather than against
    a fixed number of milliseconds because 5 ms of disagreement is excellent on a
    pair of 30 fps clips and poor on a pair at 240.
    """
    if discrepancy_s is None:
        return 0.0
    if floor_s <= 0.0:
        return 1.0 if discrepancy_s <= 0.0 else 0.0
    return float(np.clip(floor_s / max(discrepancy_s, floor_s), 0.0, 1.0))


def _peak_agreement(correlation: CorrelationReport) -> float:
    """Agreement for a map with nothing but the correlation behind it.

    The margin by which the peak beat its nearest rival, on the correlation
    scale, and nothing else. A peak of 0.98 that beat a rival of 0.94 is a
    coin toss between two alignments a whole phase apart and scores 0.04; the
    same peak against a rival of 0.3 scores 0.68.

    Deliberately *not* normalised into the room above the rival, which would
    score any perfect peak at 1.0 however close behind the rival sat. A
    correlation-only map has one parameter and nothing independent checking it,
    so it should not be able to reach the score an anchored fit earns by
    producing a residual and surviving it.
    """
    peak = float(np.clip(correlation.peak_correlation, 0.0, 1.0))
    rival = correlation.rival_correlation
    if rival is None:
        return peak
    return float(np.clip(correlation.peak_correlation - rival, 0.0, 1.0))


def _stability(time_map: TimeMap, overlap: OverlapReport) -> float:
    """How much of the map's accuracy survives to the worst point of the overlap.

    The uncertainty is largest at whichever end of the overlap sits further from
    the pivot, because that is where a fitted rate's lever arm is longest. The
    ratio against the pivot's own uncertainty isolates the extrapolation cost
    from everything else: an offset-only map has no lever arm and scores 1.0
    whether its anchors agreed well or badly, leaving that question entirely to
    `agreement` rather than charging for it twice.
    """
    best = time_map.offset_uncertainty_s
    if best is None or best <= 0.0:
        return 1.0
    worst = max(
        (
            value
            for value in (
                time_map.uncertainty_at(overlap.start_s),
                time_map.uncertainty_at(overlap.end_s),
            )
            if value is not None
        ),
        default=best,
    )
    return float(np.clip(best / max(worst, best), 0.0, 1.0))


def _refused(
    reference: ClipRef,
    target: ClipRef,
    config: SyncConfig,
    reason: str,
    *,
    correlation: CorrelationReport | None = None,
    warnings: list[str] | None = None,
) -> SyncModel:
    return SyncModel(
        aligned=False,
        method=None,
        reference=reference,
        target=target,
        correlation=correlation,
        config=config,
        refusal=reason,
        warnings=warnings or [],
    )


def align(
    reference: SyncInput,
    target: SyncInput,
    config: SyncConfig | None = None,
    *,
    picks: list[ManualPick] | None = None,
) -> SyncModel:
    """Relate two clips' clocks, and say how well the relation is known."""
    resolved = config or SyncConfig()
    reference_clip, target_clip = reference.clip_ref(), target.clip_ref()

    # Carried up before anything else is attempted. The commonest reason a clip
    # contributes nothing here is that its frame rate could not support the
    # smoothing window, and the filter's own note names the window that would --
    # which is the fix, and is otherwise two layers away from the symptom.
    warnings: list[str] = [
        f"{role} clip ({clip.path.name}): {note}"
        for role, clip in (("Reference", reference), ("Target", target))
        for note in clip.notes
    ]

    if reference_clip.frames < 2 or target_clip.frames < 2:
        return _refused(
            reference_clip,
            target_clip,
            resolved,
            "A clip of fewer than two frames has no clock to align; nothing can be measured "
            "about how its timing relates to another recording.",
        )

    floor = quantisation_floor_s(reference_clip.median_interval_s, target_clip.median_interval_s)

    # Computed whatever method is chosen: where it does not produce the map it is
    # the only thing available to check it.
    correlation = cross_correlate(reference.speed_track(), target.speed_track(), resolved)

    anchors, method, notes = _gather_anchors(reference, target, resolved, picks)
    warnings.extend(notes)

    if not anchors and method is not SyncMethod.CORRELATION:
        if correlation is None:
            return _refused(
                reference_clip,
                target_clip,
                resolved,
                "Neither clip yielded anchors to align on, and "
                + _no_signal_reason(reference, target, resolved),
                warnings=warnings,
            )
        method = SyncMethod.CORRELATION

    if method is SyncMethod.CORRELATION:
        if correlation is None:
            return _refused(
                reference_clip,
                target_clip,
                resolved,
                "Cross-correlation was asked for, but no lag gave the two clips "
                f"{resolved.min_overlap_s:g} s of common hand-speed signal to score.",
                warnings=warnings,
            )
        return _from_correlation(
            reference_clip, target_clip, correlation, resolved, floor, warnings
        )

    return _from_anchors(
        reference_clip,
        target_clip,
        anchors,
        method or SyncMethod.EVENTS,
        correlation,
        resolved,
        floor,
        warnings,
    )


def _no_signal_reason(reference: SyncInput, target: SyncInput, config: SyncConfig) -> str:
    """Say which clip left nothing to correlate, and what usually causes it.

    Naming the clip matters because the two failures need different actions. A
    clip with no usable hand trajectory at all is a filtering or tracking
    problem in that one recording; two clips that each have one and still cannot
    be overlapped are a capture problem in the pair.
    """
    empty = [
        clip.path.name
        for clip in (reference, target)
        if not bool(np.any(np.isfinite(clip.signals.speed)))
    ]
    if empty:
        return (
            f"{' and '.join(empty)} produced no usable hand trajectory at all. The commonest "
            "cause is the smoothing window: at a low frame rate it holds too few samples to "
            "fit, and the filter's own report names the shortest window that clip's rate "
            "supports. Both clips share one window on purpose, so the coarser one sets it."
        )
    return (
        f"no lag gives them {config.min_overlap_s:g} s of common hand-speed signal. Either "
        "the clips do not overlap in time, or the hands are tracked in different parts of each."
    )


def _gather_anchors(
    reference: SyncInput,
    target: SyncInput,
    config: SyncConfig,
    picks: list[ManualPick] | None,
) -> tuple[list[SyncAnchor], SyncMethod | None, list[str]]:
    """Decide what the map will be fitted to, and report what was discarded."""
    if picks:
        return manual_anchors(picks, reference.signals.t, target.signals.t), SyncMethod.MANUAL, []

    if config.method is SyncMethod.MANUAL:
        raise ValueError(
            "method='manual' needs anchors; none were supplied. Pick the same instant in "
            "each clip and pass its frame number from both."
        )
    if config.method is SyncMethod.CORRELATION:
        return [], SyncMethod.CORRELATION, []

    notes: list[str] = []
    for role, clip in (("reference", reference), ("target", target)):
        if not clip.phases.detected:
            notes.append(
                f"No swing was detected in the {role} clip, so it contributed no events to "
                "align on. Correlating the hand-speed signals is all that is left."
            )

    found = event_anchors(reference.phases, target.phases)
    if not found:
        if config.method is SyncMethod.EVENTS:
            raise ValueError(
                "method='events' needs the same swing event located in both clips; no event "
                "was found in both. Run `analyzer phases` on each clip to see which were."
            )
        return [], None, notes

    kept, rejected = reject_outliers(found, config)
    notes.extend(rejected)
    return kept, SyncMethod.EVENTS, notes


def _from_correlation(
    reference: ClipRef,
    target: ClipRef,
    correlation: CorrelationReport,
    config: SyncConfig,
    floor_s: float,
    warnings: list[str],
) -> SyncModel:
    """Build the map from the correlation peak alone.

    The offset is the peak lag. The rate is not estimated and cannot be: a single
    lag is one number, and a clock rate needs two instants far apart.

    The offset uncertainty is reported as the quantisation floor, which is a
    **bound rather than a measurement**. Sub-frame lag estimation genuinely beats
    the frame interval on a smooth signal, so the truth is probably better than
    this; how much better depends on the noise and the slope of the correlation
    peak, and neither is measured here. Claiming the better number without
    measuring it is the thing this project does not do.
    """
    pivot = reference.start_s + reference.duration_s / 2.0
    time_map = TimeMap(
        offset_s=correlation.peak_offset_s,
        rate=1.0,
        rate_estimated=False,
        pivot_s=pivot,
        offset_uncertainty_s=floor_s,
        rate_uncertainty=None,
        # A correlation is supported wherever the two signals overlapped, which
        # is the whole reference clip clipped to the overlap -- not a pair of
        # instants, so it does not extrapolate the way an anchored fit does.
        support_start_s=reference.start_s,
        support_end_s=reference.start_s + reference.duration_s,
    )
    overlap = _overlap(reference, target, time_map)

    agreement = _peak_agreement(correlation)
    observation = (
        float(np.clip(correlation.overlap_s / overlap.duration_s, 0.0, 1.0))
        if overlap.duration_s > 0
        else 0.0
    )
    confidence = SyncConfidence(
        overall=agreement * observation,
        agreement=agreement,
        anchors=observation,
        # A constant offset is exactly as good at the edges of the overlap as at
        # its middle, so there is nothing for this factor to take away.
        stability=1.0,
    )

    warnings.append(
        "This alignment rests on the correlation peak alone. With no second instant there "
        "is no clock rate and no residual, so what stands behind it is how far the peak sat "
        "above its nearest rival."
    )
    warnings.extend(_correlation_warnings(correlation))
    warnings.append(_SAME_SWING_CAVEAT)

    return SyncModel(
        aligned=True,
        method=SyncMethod.CORRELATION,
        reference=reference,
        target=target,
        time_map=time_map,
        anchors=[],
        residuals=[],
        quality=SyncQuality(
            residual_rms_ms=None,
            residual_max_ms=None,
            degrees_of_freedom=0,
            quantisation_floor_ms=floor_s * 1000.0,
            method_disagreement_ms=None,
        ),
        confidence=confidence,
        correlation=correlation,
        overlap=overlap,
        config=config,
        warnings=warnings,
    )


def _with_correlated_offset(
    fitted: TimeMap,
    correlation: CorrelationReport,
    anchors: list[SyncAnchor],
    floor_s: float,
) -> tuple[TimeMap, list[AnchorResidual]]:
    """Re-centre an offset-only map on the correlation peak, and re-score the anchors.

    Everything else about the map is kept: the pivot, the anchored support, and
    the rate held at 1.0. Only the one number the correlation is better at
    estimating changes.

    The uncertainty is reported as the quantisation floor, which the benchmark
    says is **conservative** -- a correlation over several hundred samples
    recovers a 120 fps pair's offset to well under a millisecond, against a
    3.4 ms floor. Quoting the measured figure instead would mean claiming a
    per-pair uncertainty this function has not computed; the floor is a bound it
    can defend, and the gap between them is documented rather than banked.

    The residuals are recomputed against the published map, because a residual
    against a map nobody is given is not a residual. They therefore now carry
    both the anchors' disagreement with each other and their collective
    disagreement with the correlation -- and `SyncQuality.method_disagreement_ms`
    is the second of those on its own, so a reader can separate them.
    """
    moved = fitted.model_copy(
        update={
            "offset_s": correlation.peak_offset_s,
            "offset_uncertainty_s": floor_s,
            "rate_uncertainty": None,
        }
    )
    return moved, residuals_against(anchors, moved)


def _correlation_warnings(correlation: CorrelationReport) -> list[str]:
    rival = correlation.rival_correlation
    if rival is None or correlation.peak_correlation - rival >= _AMBIGUOUS_PEAK_MARGIN:
        return []
    return [
        f"The best correlation ({correlation.peak_correlation:.2f} at "
        f"{correlation.peak_offset_s:+.3f} s) barely beat a rival alignment "
        f"({rival:.2f} at {correlation.rival_offset_s:+.3f} s). A swing has two speed humps, "
        "so the rival is usually the lag that lays one clip's backswing over the other's "
        "downswing -- which is a whole phase wrong. Check the alignment by eye."
    ]


def _from_anchors(
    reference: ClipRef,
    target: ClipRef,
    anchors: list[SyncAnchor],
    method: SyncMethod,
    correlation: CorrelationReport | None,
    config: SyncConfig,
    floor_s: float,
    warnings: list[str],
) -> SyncModel:
    """Fit the map to identified instants and account for what checks it."""
    time_map, residuals, notes = fit_time_map(anchors, floor_s=floor_s, config=config)
    warnings.extend(notes)

    fitted_offset_s = time_map.offset_s
    if (
        config.prefer_correlated_offset
        and correlation is not None
        and not time_map.rate_estimated
        # Only where the caller left the choice open. Asking for `events`
        # explicitly and being handed a correlation offset would make the option
        # do nothing, and it is the option the benchmark uses to measure what
        # anchoring on events alone is actually worth.
        and config.method is None
        and method is not SyncMethod.MANUAL
    ):
        time_map, residuals = _with_correlated_offset(time_map, correlation, anchors, floor_s)
        method = SyncMethod.COMBINED
        warnings.append(
            f"The offset is the cross-correlation's ({correlation.peak_offset_s * 1000:+.1f} ms), "
            f"not the anchored fit's ({fitted_offset_s * 1000:+.1f} ms). The correlation "
            f"averages {correlation.samples} samples where the anchors are "
            f"{len(anchors)} instants, two of which -- the takeaway and the finish -- are "
            "threshold crossings on a signal that is barely moving there. The anchors still "
            "supply the residual below, which is what checks the result."
        )

    overlap = _overlap(reference, target, time_map)

    dof = len(anchors) - (2 if time_map.rate_estimated else 1)
    residual_rms: float | None = None
    residual_max: float | None = None
    if dof > 0:
        squared = [(entry.residual_ms / 1000.0) ** 2 for entry in residuals]
        residual_rms = float(np.sqrt(sum(squared) / len(squared)))
        residual_max = max(abs(entry.residual_ms) for entry in residuals) / 1000.0

    # Always the gap between the two *estimators*, never between an estimator
    # and the published map: once the correlation has supplied the offset, the
    # published map agrees with it exactly, and comparing them would report zero
    # disagreement wherever they disagreed most.
    disagreement: float | None = None
    if correlation is not None:
        disagreement = abs(fitted_offset_s - correlation.peak_offset_s)

    # Agreement is evidence about the estimator that actually produced the
    # offset, which is not the same quantity for the two kinds of map.
    #
    # For a COMBINED map the correlation supplied the offset, so what backs it is
    # how decisively that peak beat its nearest rival. The residual cannot do the
    # job here, and the benchmark is why: under realistic landmark noise the four
    # events scatter 34 ms about the map on a pair that is *correctly* aligned to
    # under a millisecond, so scoring that scatter against the 3.4 ms frame
    # quantisation reports 0.1 for a perfect alignment. Worse, it does not
    # discriminate -- two clips of genuinely different swings scatter 44 ms,
    # which overlaps. The correlation margin separates the same cases cleanly:
    # 0.36 for one swing, 0.09 at an 8% tempo difference, 0.01 at 15%.
    #
    # For an EVENTS or MANUAL map the anchors supplied the offset, so the
    # residual is the direct measure of how well they determined it, and the
    # correlation is the independent check to be taken at its worst alongside.
    # That check drops out once a rate is fitted: a correlation yields one lag
    # for the whole clip and a rate-bearing map has a different offset at every
    # instant, so a 9 ms gap would be punished on maps whose rate is real.
    if method is SyncMethod.COMBINED and correlation is not None:
        agreement = _peak_agreement(correlation)
    else:
        checks = [residual_rms]
        if not time_map.rate_estimated:
            checks.append(disagreement)
        agreement = min(
            (_agreement(value, floor_s) for value in checks if value is not None), default=0.0
        )

    anchor_confidence = float(np.mean([anchor.confidence for anchor in anchors]))
    stability = _stability(time_map, overlap)

    quality = SyncQuality(
        residual_rms_ms=None if residual_rms is None else residual_rms * 1000.0,
        residual_max_ms=None if residual_max is None else residual_max * 1000.0,
        degrees_of_freedom=dof,
        quantisation_floor_ms=floor_s * 1000.0,
        method_disagreement_ms=None if disagreement is None else disagreement * 1000.0,
    )

    if dof <= 0:
        warnings.append(
            f"{len(anchors)} anchor(s) and {2 if time_map.rate_estimated else 1} fitted "
            "parameter(s) leaves no spare degrees of freedom, so the fit passes exactly "
            "through every anchor and its residual is zero by construction rather than by "
            "agreement. No residual is reported for that reason."
            + ("" if correlation is not None else " Nothing else checks this map.")
        )
    if correlation is not None:
        warnings.extend(_correlation_warnings(correlation))
        if disagreement is not None and disagreement > _METHOD_DISAGREEMENT_WARN_S:
            warnings.append(
                f"The fitted offset ({time_map.offset_s:+.3f} s) and the independent "
                f"cross-correlation ({correlation.peak_offset_s:+.3f} s) disagree by "
                f"{disagreement * 1000:.0f} ms. They are not averaged: one of them is "
                "aligned on the wrong feature, and which is not decidable from here."
            )
    warnings.append(_SAME_SWING_CAVEAT)

    if residual_rms is not None and residual_rms * 1000.0 > config.max_residual_ms:
        return _refused(
            reference,
            target,
            config,
            f"The anchors cannot be reconciled: they scatter {residual_rms * 1000:.0f} ms "
            f"about the best map, against a {config.max_residual_ms:g} ms bound and a "
            f"{floor_s * 1000:.1f} ms floor set by the two frame rates. Either an event is "
            "detected in the wrong place, a slow-motion factor is wrong, or the two clips "
            "are not the same swing.",
            correlation=correlation,
            warnings=warnings,
        )

    if overlap.duration_s <= 0.0:
        return _refused(
            reference,
            target,
            config,
            "The map places the two clips end to end with no overlap at all, so there is no "
            "instant both cameras recorded. Check that the clips are of the same swing.",
            correlation=correlation,
            warnings=warnings,
        )

    return SyncModel(
        aligned=True,
        method=method,
        reference=reference,
        target=target,
        time_map=time_map,
        anchors=anchors,
        residuals=residuals,
        quality=quality,
        confidence=SyncConfidence(
            overall=agreement * anchor_confidence * stability,
            agreement=agreement,
            anchors=anchor_confidence,
            stability=stability,
        ),
        correlation=correlation,
        overlap=overlap,
        config=config,
        warnings=warnings,
    )


__all__ = [
    "AnchorResidual",
    "ManualPick",
    "SyncInput",
    "align",
]
