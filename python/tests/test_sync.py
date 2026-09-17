"""Two-camera synchronisation tests.

Judged against **known offsets**, which is the one thing this layer can be
tested against honestly: no simultaneous two-camera recording exists in this
project, so there is no real pair with a measured ground truth. What there is
instead is one synthetic swing sampled twice, by two cameras with different
clocks and different frame rates, where the offset between them is an input.

`tests/synthetic.py` builds those, and the convention is worth stating once:
`world_start_s = -offset` gives a target camera that started `offset` seconds
before the reference, so its own clock reads `offset` ahead at every instant
both saw. The aligner should recover exactly that number.

Two things the tests deliberately do *not* assume. That the recovered offset is
exact -- an instant is located to about a frame, so the bound is the
quantisation floor the pair's frame rates impose, which the engine computes and
reports. And that a high confidence means a correct answer: several tests below
check that a *wrong* alignment scores near zero, which is the more useful half
of the claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.contracts.phases import SwingEvent
from analyzer.contracts.sync import (
    AnchorSource,
    SyncConfig,
    SyncMethod,
    TimeMap,
)
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases, swing_signals
from analyzer.sync import (
    AnchorError,
    ManualPick,
    SpeedTrack,
    SyncInput,
    align,
    cross_correlate,
    event_anchors,
    manual_anchors,
    quantisation_floor_s,
)
from analyzer.sync.correlate import _correlate, _lags
from analyzer.sync.timemap import TimeMapError, fit_time_map
from tests.synthetic import DURATION_S, FPS, swing_sequence

# A 30 fps clip cannot support the 0.10 s default: the window holds three
# samples and a degree-4 fit needs five. Every mismatched-rate test below shares
# this wider window across *both* clips, which is the engine's rule rather than
# a convenience -- smoothing two clips differently shifts the features the
# alignment keys on.
WIDE = FilterConfig(smoothing=SmoothingConfig(window_s=0.25))


def camera(
    offset_s: float = 0.0,
    fps: float = FPS,
    *,
    name: str = "camera.mov",
    duration_s: float = DURATION_S,
    time_scale: float = 1.0,
    config: FilterConfig | None = None,
    **kwargs: object,
) -> SyncInput:
    """One camera's view of the shared synthetic swing.

    `offset_s` is how far ahead of the reference this camera's clock reads, and
    is therefore the ground truth `align` is asked to recover.
    """
    sequence = swing_sequence(
        duration_s=duration_s,
        fps=fps,
        world_start_s=-offset_s,
        time_scale=time_scale,
        **kwargs,
    )
    filtered = filter_sequence(sequence, config or FilterConfig())
    return SyncInput(
        path=Path(name),
        signals=swing_signals(filtered),
        phases=detect_phases(filtered),
        notes=tuple(filtered.report.warnings),
    )


@pytest.fixture(scope="module")
def reference() -> SyncInput:
    return camera(0.0, name="reference.mov")


@pytest.fixture(scope="module")
def late() -> SyncInput:
    """A camera that started 1.0 s into the swing, so its clock runs a second behind.

    Past the takeaway, which means its own "takeaway" is where the clip begins
    rather than where the hands moved -- the anchor the outlier check has to
    catch.
    """
    return camera(-1.0, name="late.mov", duration_s=1.6)


class TestCorrelationConvention:
    """The index arithmetic that turns a lag into an offset, pinned.

    One reindexing line converts SciPy's correlation convention into this
    module's, and a sign error there would misalign every pair by twice the
    offset while leaving every correlation value looking perfect.
    """

    def test_lag_index_matches_the_documented_shift(self) -> None:
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

        out = _correlate(a, b)
        lags = _lags(a.size)

        # a[0] lines up with b[2], so the peak is at lag +2.
        assert lags[int(np.argmax(out))] == 2

    def test_a_positive_lag_means_the_target_ran_ahead(self) -> None:
        target = camera(0.35, name="target.mov")
        report = cross_correlate(
            SpeedTrack(camera(0.0).signals.t, camera(0.0).signals.speed),
            SpeedTrack(target.signals.t, target.signals.speed),
            SyncConfig(),
        )
        assert report is not None
        assert report.peak_offset_s == pytest.approx(0.35, abs=1 / FPS)


class TestKnownOffsets:
    """The headline claim: a known offset comes back, to within the frame quantisation.

    The sweep stays inside +-0.5 s because the fixture clip is 2.6 s long and the
    swing occupies 0.5 s to 2.1 s of it. Shift the second camera further than
    that and one end of the swing falls outside its recording, at which point the
    pair no longer contains a common swing to align -- a real situation, and the
    subject of `test_a_clipped_swing_is_reported_as_unreliable` below rather than
    of this claim.
    """

    @pytest.mark.parametrize("truth", [-0.4, -0.25, 0.0, 0.25, 0.4, 0.5])
    def test_offset_is_recovered(self, reference: SyncInput, truth: float) -> None:
        result = align(reference, camera(truth, name="target.mov"))

        assert result.aligned
        assert result.time_map is not None
        floor = quantisation_floor_s(1 / FPS, 1 / FPS)
        assert result.time_map.offset_s == pytest.approx(truth, abs=max(floor, 1 / FPS))

    @pytest.mark.parametrize("truth", [-0.75, 0.75])
    def test_a_clipped_swing_is_reported_as_unreliable(
        self, reference: SyncInput, truth: float
    ) -> None:
        """Shifted this far, one camera misses an end of the swing.

        It then locates an event on its own clip boundary rather than where the
        hands moved, and the anchors stop agreeing with each other. The offset
        may well survive -- the correlation does not care that an event was
        mislabelled -- so what has to hold is that the *reported* trust
        collapses, because nothing here can tell this case from a genuinely
        misaligned one.
        """
        result = align(reference, camera(truth, name="clipped.mov"))

        assert result.confidence is not None
        assert result.confidence.overall < 0.1
        assert result.quality is not None
        assert result.quality.residual_rms_ms is not None
        assert result.quality.residual_rms_ms > result.quality.quantisation_floor_ms * 5

    def test_a_clean_pair_lands_within_the_frame_quantisation(self, reference: SyncInput) -> None:
        """What a clean pair earns is an exact offset, not a confidence of 1.0.

        The headline stays modest even here, and the reason is a property of the
        fixture rather than of the pair: `agreement` on a combined map is how far
        the correlation peak sat above its nearest rival, and the synthetic
        swing's backswing and downswing are near mirror images by construction,
        so a lag one phase away still correlates at ~0.8. Real footage is less
        self-similar. Asserting a high score here would be asserting that this
        fixture is representative, which it is not.
        """
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.time_map is not None
        assert result.time_map.offset_s == pytest.approx(0.4, abs=1 / FPS)
        assert result.quality is not None
        assert result.quality.residual_rms_ms is not None
        assert result.quality.residual_rms_ms < result.quality.quantisation_floor_ms

    def test_the_offset_comes_from_the_correlation_and_the_anchors_check_it(
        self, reference: SyncInput
    ) -> None:
        """The default splits the work: correlation for the offset, events for the rest."""
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.method is SyncMethod.COMBINED
        assert result.correlation is not None
        assert result.time_map is not None
        assert result.time_map.offset_s == pytest.approx(result.correlation.peak_offset_s)
        assert result.anchors, "the anchors still have to be there to supply the residual"
        assert any("cross-correlation's" in note for note in result.warnings)

    def test_forcing_events_keeps_the_anchored_offset(self, reference: SyncInput) -> None:
        """Asking for a method has to produce that method, or the option does nothing."""
        target = camera(0.4, name="target.mov")
        forced = align(reference, target, SyncConfig(method=SyncMethod.EVENTS))

        assert forced.method is SyncMethod.EVENTS
        assert forced.correlation is not None
        assert forced.time_map is not None
        assert forced.time_map.offset_s != pytest.approx(forced.correlation.peak_offset_s)

    def test_the_two_estimators_are_both_computed_and_agree(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.quality is not None
        assert result.quality.method_disagreement_ms is not None
        assert abs(result.quality.method_disagreement_ms) < 1000 / FPS

    def test_every_event_found_in_both_clips_becomes_an_anchor(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))

        assert {anchor.event for anchor in result.anchors} == set(SwingEvent)
        assert all(anchor.source is AnchorSource.DETECTED for anchor in result.anchors)

    def test_the_same_clip_twice_aligns_at_zero(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.0, name="copy.mov"))

        assert result.time_map is not None
        assert result.time_map.offset_s == pytest.approx(0.0, abs=1e-9)


class TestMismatchedFrameRates:
    """A 30 fps camera beside a 120 fps one, which is the ordinary consumer pair."""

    def test_the_offset_survives_a_four_to_one_rate_difference(self) -> None:
        result = align(
            camera(0.0, 120.0, name="fast.mov", config=WIDE),
            camera(0.4, 30.0, name="slow.mov", config=WIDE),
        )

        assert result.aligned
        assert result.time_map is not None
        assert result.quality is not None
        assert result.time_map.offset_s == pytest.approx(
            0.4, abs=result.quality.quantisation_floor_ms / 1000 * 2
        )

    def test_upgrading_one_camera_is_worth_at_most_root_two(self) -> None:
        """A number a reader can act on before buying a camera.

        The floor adds the two clips' quantisations in quadrature, so replacing
        one 30 fps camera with an infinitely fast one removes one of two equal
        terms and no more: a gain of sqrt(2), approached and never beaten.
        Replacing both is worth the full frame-rate ratio.
        """
        both_slow = quantisation_floor_s(1 / 30, 1 / 30)
        one_fast = quantisation_floor_s(1 / 30, 1 / 240)
        both_fast = quantisation_floor_s(1 / 240, 1 / 240)

        assert both_slow * 1000 == pytest.approx(13.6, abs=0.1)
        assert one_fast * 1000 == pytest.approx(9.7, abs=0.1)
        assert both_fast * 1000 == pytest.approx(1.7, abs=0.1)

        # The asymptote: a perfect second camera still leaves the 30 fps clip's
        # own contribution, which is the slow pair's floor over sqrt(2).
        assert one_fast > both_slow / np.sqrt(2)
        assert one_fast < both_slow

    def test_the_default_window_refuses_a_thirty_fps_clip_and_says_why(self) -> None:
        """Phase 3's frame-rate floor arrives here as a refusal naming the fix."""
        result = align(camera(0.0, 120.0, name="fast.mov"), camera(0.4, 30.0, name="slow.mov"))

        assert not result.aligned
        assert result.refusal is not None
        assert "smoothing window" in result.refusal
        assert "slow.mov" in result.refusal
        assert any("0.1 s window" in note for note in result.warnings)


class TestPartialOverlap:
    """A camera that started rolling after the swing began."""

    def test_the_offset_is_still_recovered(self, reference: SyncInput, late: SyncInput) -> None:
        result = align(reference, late)

        assert result.aligned
        assert result.time_map is not None
        assert result.time_map.offset_s == pytest.approx(-1.0, abs=1 / FPS)

    def test_the_spurious_anchor_is_dropped_and_named(
        self, reference: SyncInput, late: SyncInput
    ) -> None:
        result = align(reference, late)

        assert result.anchor(SwingEvent.TAKEAWAY) is None
        assert any("takeaway" in note and "dropped" in note for note in result.warnings)

    def test_the_overlap_is_reported_in_reference_time(
        self, reference: SyncInput, late: SyncInput
    ) -> None:
        result = align(reference, late)

        assert result.overlap is not None
        assert result.overlap.duration_s == pytest.approx(1.6, abs=0.05)
        # The whole of the shorter clip is inside the overlap; only part of the
        # longer one is, and the two fractions must say so differently.
        assert result.overlap.target_fraction > 0.95
        assert result.overlap.reference_fraction < 0.7

    def test_outliers_are_not_dropped_below_two_anchors(self) -> None:
        """With no majority to believe, nothing is discarded and the residual says so."""
        anchors = event_anchors(camera(0.0).phases, camera(0.4).phases)
        scattered = [
            anchor.model_copy(update={"target_s": anchor.target_s + shift})
            for anchor, shift in zip(anchors, (0.0, 1.0, 2.0, 3.0), strict=True)
        ]

        from analyzer.sync.anchors import reject_outliers

        kept, notes = reject_outliers(scattered, SyncConfig())
        assert len(kept) == len(scattered)
        assert any("no majority" in note for note in notes)


class TestRateEstimation:
    """The second parameter, and the three gates it has to pass."""

    def test_a_clean_pair_does_not_fit_a_rate(self, reference: SyncInput) -> None:
        """Two clocks that agree should be reported as agreeing, not as 1.0000003."""
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.time_map is not None
        assert not result.time_map.rate_estimated
        assert result.time_map.rate == 1.0
        assert result.time_map.rate_uncertainty is None

    def test_an_insignificant_rate_is_dropped_and_said_so(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))

        assert any("standard errors" in note for note in result.warnings)

    def test_a_wrong_slow_motion_factor_is_measured_as_a_rate(self, reference: SyncInput) -> None:
        """What a second camera makes measurable and one camera cannot.

        The target clip's supplied factor is 10% out, which shows up as a clock
        rate of 1/1.10 -- and Phase 6 recorded that nothing in a single clip can
        recover its own factor. The *ratio* of two is a different question.
        """
        result = align(reference, camera(0.0, name="wrong.mov", time_scale=1 / 1.10))

        assert result.time_map is not None
        assert result.time_map.rate_estimated
        assert result.time_map.rate == pytest.approx(1 / 1.10, rel=0.01)
        assert result.time_map.rate_uncertainty is not None
        assert any("slow-motion" in note for note in result.warnings)

    def test_a_short_baseline_refuses_the_rate(self) -> None:
        anchors = event_anchors(camera(0.0).phases, camera(0.4).phases)
        fitted, _, notes = fit_time_map(
            anchors, floor_s=0.004, config=SyncConfig(min_rate_span_s=10.0)
        )

        assert not fitted.rate_estimated
        assert any("span" in note for note in notes)

    def test_an_absurd_rate_is_refused_outright(self) -> None:
        anchors = event_anchors(camera(0.0).phases, camera(0.4).phases)
        stretched = [
            anchor.model_copy(update={"target_s": anchor.target_s * 4.0}) for anchor in anchors
        ]
        fitted, _, notes = fit_time_map(stretched, floor_s=0.004, config=SyncConfig())

        assert not fitted.rate_estimated
        assert any("refused" in note for note in notes)

    def test_one_anchor_determines_an_offset_and_nothing_else(self) -> None:
        anchors = event_anchors(camera(0.0).phases, camera(0.4).phases)[:1]
        fitted, residuals, notes = fit_time_map(anchors, floor_s=0.004, config=SyncConfig())

        assert not fitted.rate_estimated
        assert fitted.offset_s == pytest.approx(anchors[0].target_s - anchors[0].reference_s)
        assert len(residuals) == 1
        assert any("nothing else" in note for note in notes)

    def test_no_anchors_is_an_error_rather_than_an_identity_map(self) -> None:
        with pytest.raises(TimeMapError):
            fit_time_map([], floor_s=0.004, config=SyncConfig())


class TestTimeMapAlgebra:
    """The map itself, independent of anything that produces one."""

    @staticmethod
    def _map(rate: float = 1.0, **kwargs: object) -> TimeMap:
        defaults: dict[str, object] = {
            "offset_s": 0.5,
            "rate": rate,
            "rate_estimated": rate != 1.0,
            "pivot_s": 2.0,
            "support_start_s": 1.0,
            "support_end_s": 3.0,
        }
        return TimeMap(**{**defaults, **kwargs})  # type: ignore[arg-type]

    def test_the_offset_is_exact_at_the_pivot(self) -> None:
        mapping = self._map(rate=1.2)
        assert mapping.to_target(2.0) == pytest.approx(2.5)

    def test_a_unit_rate_is_a_constant_offset_everywhere(self) -> None:
        mapping = self._map()
        for t in (-10.0, 0.0, 2.0, 100.0):
            assert mapping.to_target(t) == pytest.approx(t + 0.5)

    @pytest.mark.parametrize("rate", [0.8, 1.0, 1.25])
    def test_the_two_directions_invert_each_other(self, rate: float) -> None:
        mapping = self._map(rate=rate)
        for t in (0.0, 1.5, 2.0, 7.25):
            assert mapping.to_reference(mapping.to_target(t)) == pytest.approx(t)

    def test_uncertainty_is_flat_without_a_rate(self) -> None:
        mapping = self._map(offset_uncertainty_s=0.01)
        assert mapping.uncertainty_at(2.0) == pytest.approx(0.01)
        assert mapping.uncertainty_at(50.0) == pytest.approx(0.01)

    def test_uncertainty_grows_away_from_the_pivot_with_a_rate(self) -> None:
        mapping = self._map(rate=1.1, offset_uncertainty_s=0.01, rate_uncertainty=0.02)

        at_pivot = mapping.uncertainty_at(2.0)
        nearby = mapping.uncertainty_at(3.0)
        far = mapping.uncertainty_at(12.0)

        assert at_pivot is not None and nearby is not None and far is not None
        assert at_pivot < nearby < far
        assert at_pivot == pytest.approx(0.01)
        # Quadrature, which is only valid because the pivot is the centroid.
        assert nearby == pytest.approx(np.hypot(0.01, 0.02))

    def test_unknown_uncertainty_stays_unknown(self) -> None:
        assert self._map().uncertainty_at(5.0) is None


class TestQuantisationFloor:
    def test_it_is_the_quadrature_sum_of_two_uniform_errors(self) -> None:
        assert quantisation_floor_s(1 / 30, 1 / 30) == pytest.approx(
            np.sqrt(2 * (1 / 30) ** 2 / 12)
        )

    def test_two_thirty_fps_cameras_are_bounded_near_ten_milliseconds(self) -> None:
        assert quantisation_floor_s(1 / 30, 1 / 30) * 1000 == pytest.approx(13.6, abs=0.1)

    def test_the_reported_uncertainty_never_beats_the_floor_per_anchor(self) -> None:
        """Four anchors agreeing to the microsecond have got lucky with rounding."""
        anchors = event_anchors(camera(0.0).phases, camera(0.4).phases)
        floor = 0.010
        fitted, _, _ = fit_time_map(anchors, floor_s=floor, config=SyncConfig())

        assert fitted.offset_uncertainty_s is not None
        # Averaging four anchors beats the per-anchor floor by sqrt(4), and no more.
        assert fitted.offset_uncertainty_s == pytest.approx(floor / 2, rel=0.01)


def pick(
    reference: SyncInput, target: SyncInput, event: SwingEvent, label: str | None = None
) -> ManualPick:
    """The pick a person would make, taken from where each clip put the event.

    Derived rather than hard-coded: a literal frame number is a fact about the
    fixture's frame rate and offset, and silently becomes a different instant the
    moment either changes.
    """
    left, right = reference.phases.event(event), target.phases.event(event)
    assert left is not None and right is not None
    return ManualPick(
        label=label or event.value,
        reference_frame=left.frame_index,
        target_frame=right.frame_index,
    )


class TestManualAnchors:
    """A person's picks are the answer, not an estimate to be arbitrated."""

    def test_frames_are_converted_through_the_clips_own_timestamps(
        self, reference: SyncInput
    ) -> None:
        target = camera(0.4, name="target.mov")
        chosen = pick(reference, target, SwingEvent.IMPACT)

        anchors = manual_anchors([chosen], reference.signals.t, target.signals.t)

        assert anchors[0].reference_s == pytest.approx(reference.signals.t[chosen.reference_frame])
        assert anchors[0].target_s == pytest.approx(target.signals.t[chosen.target_frame])
        assert anchors[0].source is AnchorSource.MANUAL
        assert anchors[0].event is None

    def test_picks_override_the_automatic_methods(self, reference: SyncInput) -> None:
        target = camera(0.4, name="target.mov")
        result = align(
            reference,
            target,
            picks=[
                pick(reference, target, SwingEvent.TOP),
                pick(reference, target, SwingEvent.IMPACT),
            ],
        )

        assert result.method is SyncMethod.MANUAL
        assert all(anchor.source is AnchorSource.MANUAL for anchor in result.anchors)
        assert result.time_map is not None
        assert result.time_map.offset_s == pytest.approx(0.4, abs=0.02)

    def test_a_frame_outside_the_clip_is_an_error_naming_the_clip(
        self, reference: SyncInput
    ) -> None:
        target = camera(0.4, name="target.mov")
        with pytest.raises(AnchorError, match="target clip"):
            manual_anchors(
                [ManualPick(label="impact", reference_frame=0, target_frame=99_999)],
                reference.signals.t,
                target.signals.t,
            )

    def test_a_single_pick_is_reported_as_unchecked(self, reference: SyncInput) -> None:
        """One anchor fixes an offset; nothing at all corroborates it except the correlation."""
        target = camera(0.4, name="target.mov")
        result = align(
            reference,
            target,
            picks=[pick(reference, target, SwingEvent.IMPACT)],
        )

        assert result.quality is not None
        assert result.quality.residual_rms_ms is None
        assert result.quality.degrees_of_freedom == 0
        assert any("zero by construction" in note for note in result.warnings)


class TestDegreesOfFreedom:
    """A residual with nothing left over is not evidence, and is not reported as one."""

    def test_an_exactly_determined_fit_reports_no_residual(self, reference: SyncInput) -> None:
        target = camera(0.4, name="target.mov")
        result = align(
            reference,
            target,
            picks=[pick(reference, target, SwingEvent.IMPACT)],
        )

        assert result.quality is not None
        assert result.quality.residual_rms_ms is None
        assert result.quality.residual_max_ms is None

    def test_four_anchors_against_one_parameter_leave_three(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.quality is not None
        assert result.quality.degrees_of_freedom == 3
        assert result.quality.residual_rms_ms is not None

    def test_residuals_are_reported_per_anchor_and_signed(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))

        assert [entry.label for entry in result.residuals] == [
            anchor.label for anchor in result.anchors
        ]
        for entry in result.residuals:
            assert entry.residual_ms == pytest.approx(
                (entry.observed_target_s - entry.predicted_target_s) * 1000
            )


class TestDifferentSwings:
    """The question this layer cannot answer, and reports rather than hides."""

    def test_two_different_swings_score_near_zero(self, reference: SyncInput) -> None:
        # 12% slower: no offset and no clock rate can reconcile two tempos.
        result = align(reference, camera(0.4, name="other.mov", time_scale=1.12))

        assert result.confidence is not None
        assert result.confidence.overall < 0.1
        assert result.quality is not None
        assert result.quality.residual_rms_ms is not None
        assert result.quality.residual_rms_ms > result.quality.quantisation_floor_ms * 5

    def test_the_caveat_is_always_stated(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.4, name="target.mov"))
        assert any("cannot tell that both cameras" in note for note in result.warnings)

    def test_an_irreconcilable_pair_is_refused_outright(self, reference: SyncInput) -> None:
        result = align(
            reference,
            camera(0.4, name="other.mov", time_scale=1.12),
            SyncConfig(max_residual_ms=20.0),
        )

        assert not result.aligned
        assert result.time_map is None, "a refusal must not leave a map to be read anyway"
        assert result.refusal is not None
        assert "not the same swing" in result.refusal


class TestMethodSelection:
    def test_correlation_can_be_forced_and_produces_no_rate(self, reference: SyncInput) -> None:
        result = align(
            reference, camera(0.4, name="target.mov"), SyncConfig(method=SyncMethod.CORRELATION)
        )

        assert result.method is SyncMethod.CORRELATION
        assert result.anchors == []
        assert result.time_map is not None
        assert not result.time_map.rate_estimated
        assert result.time_map.offset_s == pytest.approx(0.4, abs=1 / FPS)

    def test_correlation_alone_reports_that_nothing_checked_it(self, reference: SyncInput) -> None:
        """The two arrive at the same offset; only one of them can be checked.

        A combined map and a correlation-only map share their offset by
        construction, so the honest difference between them is not the number --
        it is that one has anchors to produce a residual and the other has
        nothing at all.
        """
        target = camera(0.4, name="target.mov")
        alone = align(reference, target, SyncConfig(method=SyncMethod.CORRELATION))
        combined = align(reference, target)

        assert alone.time_map is not None and combined.time_map is not None
        assert alone.time_map.offset_s == pytest.approx(combined.time_map.offset_s)

        assert alone.quality is not None and combined.quality is not None
        assert alone.quality.residual_rms_ms is None
        assert combined.quality.residual_rms_ms is not None
        assert any("rests on the correlation peak alone" in note for note in alone.warnings)

    def test_events_can_be_forced_and_fails_loudly_without_them(self) -> None:
        still = camera(0.0, name="still.mov", duration_s=0.4)
        with pytest.raises(ValueError, match="events"):
            align(camera(0.0), still, SyncConfig(method=SyncMethod.EVENTS))

    def test_manual_without_picks_is_an_error(self, reference: SyncInput) -> None:
        with pytest.raises(ValueError, match="needs anchors"):
            align(reference, camera(0.4), SyncConfig(method=SyncMethod.MANUAL))

    def test_the_rival_peak_is_reported(self, reference: SyncInput) -> None:
        """A swing has two speed humps, so a rival alignment always exists."""
        result = align(reference, camera(0.4, name="target.mov"))

        assert result.correlation is not None
        assert result.correlation.rival_correlation is not None
        assert result.correlation.rival_offset_s is not None
        assert abs(result.correlation.rival_offset_s - result.correlation.peak_offset_s) >= 0.2


class TestRefusals:
    def test_a_one_frame_clip_has_no_clock(self, reference: SyncInput) -> None:
        result = align(reference, camera(0.0, name="frame.mov", duration_s=1 / FPS))

        assert not result.aligned
        assert result.refusal is not None
        assert "two frames" in result.refusal

    def test_a_refusal_sets_a_reason_exactly_when_it_refuses(self, reference: SyncInput) -> None:
        refused = align(reference, camera(0.0, name="frame.mov", duration_s=1 / FPS))
        accepted = align(reference, camera(0.4, name="target.mov"))

        assert (refused.refusal is not None) == (not refused.aligned)
        assert (accepted.refusal is not None) == (not accepted.aligned)


class TestCorrelationMechanics:
    def test_a_lag_with_too_little_overlap_is_not_scored(self, reference: SyncInput) -> None:
        """Three coincident samples correlate perfectly and mean nothing."""
        report = cross_correlate(
            SpeedTrack(reference.signals.t, reference.signals.speed),
            SpeedTrack(reference.signals.t, reference.signals.speed),
            SyncConfig(min_overlap_s=0.5),
        )
        assert report is not None
        assert report.overlap_s >= 0.5

    def test_the_sub_grid_refinement_stays_within_half_a_step(self, reference: SyncInput) -> None:
        target = camera(0.4 + 1 / (3 * FPS), name="target.mov")
        report = cross_correlate(
            SpeedTrack(reference.signals.t, reference.signals.speed),
            SpeedTrack(target.signals.t, target.signals.speed),
            SyncConfig(),
        )
        assert report is not None
        assert abs(report.sub_grid_shift_s) <= report.grid_interval_s / 2 + 1e-12

    def test_a_signal_with_no_usable_samples_correlates_with_nothing(self) -> None:
        empty = np.full(10, np.nan)
        assert (
            cross_correlate(
                SpeedTrack(np.arange(10.0), empty),
                SpeedTrack(np.arange(10.0), empty),
                SyncConfig(),
            )
            is None
        )

    def test_gaps_are_not_interpolated_across(self) -> None:
        """A grid point far from any observation must not acquire a value."""
        times = np.concatenate([np.arange(0.0, 1.0, 0.01), np.arange(3.0, 4.0, 0.01)])
        speeds = np.sin(times * 5.0)
        track = SpeedTrack(times, speeds)

        from analyzer.sync.correlate import _resample

        grid = np.arange(0.0, 4.0, 0.01)
        _, valid = _resample(track, grid)

        # The two-second hole in the middle carries no support at all.
        hole = (grid > 1.1) & (grid < 2.9)
        assert not np.any(valid[hole])
        assert np.any(valid[grid < 1.0])
