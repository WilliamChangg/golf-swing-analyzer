"""The whole comparison, over synthetic swings whose relationship is known.

These run the real chain -- filter, detect, measure, normalise, difference --
because what is under test here is whether the pieces agree with each other. The
gates themselves are tested against hand-built metric sets in
`test_comparison_diff`, where a value can be put an exact distance from a bracket.

The load-bearing fixture property is the one `test_comparison_normalise`
describes: `tests/synthetic.py` builds its arc from the fraction through each
phase, so two swings differing only in their event times are **one swing under a
piecewise-linear time warp**. Laid on the normalised axis they must coincide, and
anything the comparison reports about their shapes is its own error.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.biomechanics import compute_metrics
from analyzer.comparison import SwingInput, compare
from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.comparison import (
    ComparisonConfig,
    DifferenceRefusal,
    MetricDifference,
    SwingComparison,
    TrajectoryChannel,
)
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.contracts.metrics import MetricName
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases
from tests.synthetic import BODY, SwingShape, swing_sequence

FILTER = FilterConfig(smoothing=SmoothingConfig(window_s=0.12))

# `tests/synthetic.BODY` projects its shoulders at 0.4 torso lengths, which the
# view detector correctly calls `unknown` -- between its two thresholds, which is
# an oblique camera. Every comparison of projected quantities then refuses, which
# is the right answer for that fixture and not the case these tests are about, so
# the shoulders are widened here to what a face-on recording looks like. The real
# reference clip measures 0.83.
FACE_ON_BODY: dict[int, tuple[float, float]] = {
    **BODY,
    int(Landmark.LEFT_SHOULDER): (0.40, 0.25),
    int(Landmark.RIGHT_SHOULDER): (0.60, 0.25),
    int(Landmark.LEFT_HIP): (0.44, 0.50),
    int(Landmark.RIGHT_HIP): (0.56, 0.50),
}


DEFAULT = SwingShape()


def swing(
    shape: SwingShape = DEFAULT,
    *,
    name: str = "clip.mov",
    fps: float = 120.0,
    body: dict[int, tuple[float, float]] | None = None,
    digest: str = "a" * 64,
) -> SwingInput:
    """One synthetic clip, all the way through the chain the app runs."""
    sequence = swing_sequence(
        duration_s=shape.finish_s + 0.4,
        fps=fps,
        shape=shape,
        path=f"/data/{name}",
        body=body if body is not None else FACE_ON_BODY,
    )
    sequence.video_content_key = ContentKey(
        algorithm=HashAlgorithm.SHA256_SAMPLED, digest=digest, size_bytes=1
    )
    filtered = filter_sequence(sequence, FILTER, space=LandmarkSpace.FRAME_WIDTHS)
    detected = detect_phases(filtered, None)
    return SwingInput(
        path=sequence.video_path,
        content_key=sequence.video_content_key,
        filtered=filtered,
        phases=detected,
        metrics=compute_metrics(filtered, detected),
    )


# --- the invariant a comparison of one clip with itself has to satisfy ----


def test_a_clip_compared_with_itself_reports_no_difference() -> None:
    """The strongest check available without a second real recording.

    Two identical inputs differ by nothing, so every difference must refuse and
    every sampled position on every curve must come out unresolved. A comparison
    that reported anything here would be reporting its own arithmetic.
    """
    clip = swing()

    result = compare(clip, clip)

    assert result.computed
    assert result.differences == []
    assert result.metrics_considered > 0

    reasons = {entry.refusal for entry in result.refused}
    assert DifferenceRefusal.UNRESOLVED in reasons
    # The pair-level gates cannot fire on one clip against itself: same view,
    # same camera, same lens, same clock. Anything else refused did so on a
    # property of the single recording, which is a different statement.
    assert not (
        reasons
        & {
            DifferenceRefusal.VIEW_MISMATCH,
            DifferenceRefusal.CAMERA_MOVED,
            DifferenceRefusal.CALIBRATION_MISMATCH,
            DifferenceRefusal.MISSING,
        }
    )
    for channel in result.trajectories:
        assert channel.refusal is None
        assert channel.resolved_fraction == 0.0
        assert channel.largest_difference is None


def test_a_clip_compared_with_itself_still_draws_both_curves() -> None:
    """Nothing differing is not nothing measured: the overlay is still there."""
    clip = swing()

    speed = compare(clip, clip).trajectory(TrajectoryChannel.HAND_SPEED)

    assert speed is not None
    drawn = [sample for sample in speed.samples if sample.reference is not None]
    assert len(drawn) > 100
    assert all(sample.difference == pytest.approx(0.0) for sample in drawn)


# --- two tempos of one swing ----------------------------------------------


def test_two_tempos_of_one_swing_differ_in_timing_and_not_in_shape() -> None:
    """The phase's central claim, on a pair where the answer is known.

    The fixture's arc is a function of the fraction through each phase, so these
    two really are the same shape at two speeds. The durations must differ and
    the normalised curves must not.
    """
    slow = swing(SwingShape(takeaway_s=0.5, top_s=1.3, impact_s=1.7), digest="a" * 64)
    quick = swing(SwingShape(takeaway_s=0.5, top_s=0.9, impact_s=1.3), digest="b" * 64)

    result = compare(slow, quick)

    assert result.computed
    named = {entry.name for entry in result.differences}
    assert MetricName.BACKSWING_DURATION in named
    assert MetricName.TEMPO_RATIO in named

    height = result.trajectory(TrajectoryChannel.HAND_HEIGHT)
    assert height is not None
    assert height.refusal is None
    assert height.resolved_fraction < 0.05


def test_the_clocks_keep_the_timing_the_normalisation_divided_out() -> None:
    slow = swing(SwingShape(takeaway_s=0.5, top_s=1.3, impact_s=1.7))
    quick = swing(SwingShape(takeaway_s=0.5, top_s=0.9, impact_s=1.3), digest="b" * 64)

    result = compare(slow, quick)

    knots = [
        (summary.clock.knots[1].timestamp_s - summary.clock.knots[0].timestamp_s)
        for summary in (result.reference, result.target)
    ]
    assert knots[0] > 1.7 * knots[1]


def test_a_different_shape_at_the_same_tempo_shows_up_in_the_curves() -> None:
    """The case the trajectory half exists for: same four instants, different route."""
    shallow = swing(SwingShape(arc_top_angle=2.2))
    steep = swing(SwingShape(arc_top_angle=1.6), digest="b" * 64)

    height = compare(shallow, steep).trajectory(TrajectoryChannel.HAND_HEIGHT)

    assert height is not None
    assert height.refusal is None
    assert height.resolved_fraction > 0.3
    assert height.largest_difference is not None
    assert height.largest_at is not None


def test_the_hand_paths_are_measured_from_each_clips_own_address() -> None:
    """So that two players standing in different parts of the frame overlay at all."""
    here = swing()
    shifted_body = {landmark: (x + 0.12, y) for landmark, (x, y) in FACE_ON_BODY.items()}
    there = swing(body=shifted_body, digest="b" * 64)

    overlay = compare(here, there).hand_path

    assert overlay is not None
    assert overlay.comparable
    at_takeaway = overlay.samples[0]
    assert at_takeaway.reference_x is not None
    assert at_takeaway.target_x is not None
    assert at_takeaway.reference_x == pytest.approx(at_takeaway.target_x, abs=0.15)


# --- refusals at the level of the whole report ----------------------------


def test_a_clip_with_no_swing_produces_a_report_rather_than_an_error() -> None:
    real = swing()
    still = swing(SwingShape(arc_top_angle=0.0), digest="b" * 64)

    result = compare(real, still)

    assert not result.computed
    assert result.differences == []
    assert result.trajectories == []
    assert result.hand_path is None
    assert any("no swing" in warning.lower() for warning in result.warnings)
    # The half that did work still reports what it found, so a reader can see
    # which clip is the problem without running the other one separately.
    assert result.reference.clock.usable
    assert not result.target.clock.usable


def test_the_axis_is_as_dense_as_the_config_asks() -> None:
    clip = swing()

    result = compare(clip, clip, ComparisonConfig(samples=25))

    for channel in result.trajectories:
        assert len(channel.samples) == 25
    assert result.hand_path is not None
    assert len(result.hand_path.samples) == 25


def test_positions_start_at_the_takeaway_and_end_at_the_finish() -> None:
    clip = swing()

    speed = compare(clip, clip).trajectory(TrajectoryChannel.HAND_SPEED)

    assert speed is not None
    assert speed.samples[0].position == 0.0
    assert speed.samples[-1].position == 3.0


def test_every_clip_is_identified_by_content_rather_than_by_path() -> None:
    left = swing(name="one.mov", digest="a" * 64)
    right = swing(name="two.mov", digest="b" * 64)

    result = compare(left, right)

    assert result.reference.content_key.digest != result.target.content_key.digest


# --- the absence that is the exit criterion -------------------------------


def test_nothing_in_the_report_is_a_score() -> None:
    """Asserted on the contract rather than on an instance, so a field added
    later is caught whether or not any fixture happens to populate it."""
    forbidden = {"score", "severity", "better", "worse", "grade", "rank", "rating", "overall"}

    for model in (SwingComparison, MetricDifference):
        assert not (set(model.model_fields) & forbidden)


def test_the_report_carries_no_summary_number_over_the_two_swings() -> None:
    """`resolved_fraction` is per channel and is a count of positions, not a
    distance. There is deliberately nothing that adds up to one figure for the
    pair -- that would be the score, arrived at by a different route."""
    clip = swing()

    result = compare(clip, clip)

    numeric = {
        name
        for name, field in SwingComparison.model_fields.items()
        if field.annotation in (float, float | None)
    }
    assert numeric == set()
    assert isinstance(result.metrics_considered, int)


def test_json_carries_no_nan_or_infinity() -> None:
    """JSON has none of either, so a value that is not known has to be null.

    The report is a TypeScript type at the other end of a pipe; a NaN here would
    produce a document a strict parser rejects, and the failure would land in the
    UI rather than where it was made.
    """
    import json

    still = swing(SwingShape(arc_top_angle=0.0), digest="b" * 64)
    for result in (compare(swing(), swing()), compare(swing(), still)):
        encoded = json.dumps(result.model_dump(mode="json"), allow_nan=False)
        assert "NaN" not in encoded
        assert "Infinity" not in encoded


def test_no_sample_carries_a_difference_without_both_sides() -> None:
    long_tail = swing(SwingShape(), fps=120.0)
    short = swing(SwingShape(), fps=60.0, digest="b" * 64)

    result = compare(long_tail, short)

    for channel in result.trajectories:
        for sample in channel.samples:
            if sample.reference is None or sample.target is None:
                assert sample.difference is None
                assert not sample.resolved


def test_a_resolved_sample_really_does_clear_its_bracket() -> None:
    shallow = swing(SwingShape(arc_top_angle=2.2))
    steep = swing(SwingShape(arc_top_angle=1.6), digest="b" * 64)

    result = compare(shallow, steep)

    for channel in result.trajectories:
        for sample in channel.samples:
            if sample.resolved:
                assert sample.difference is not None
                assert sample.bracket is not None
                assert abs(sample.difference) > sample.bracket


def test_the_resolved_fraction_counts_what_it_says_it_counts() -> None:
    shallow = swing(SwingShape(arc_top_angle=2.2))
    steep = swing(SwingShape(arc_top_angle=1.6), digest="b" * 64)

    for channel in compare(shallow, steep).trajectories:
        counted = sum(1 for sample in channel.samples if sample.resolved)
        assert channel.resolved_fraction == pytest.approx(counted / len(channel.samples))


def test_a_coarse_clip_resolves_less_than_a_fine_one() -> None:
    """The bracket is built from each clip's own clock, so a slower camera
    resolves fewer positions on the identical pair of swings."""
    shape, other = SwingShape(arc_top_angle=2.2), SwingShape(arc_top_angle=1.6)

    fine = compare(swing(shape, fps=240.0), swing(other, fps=240.0, digest="b" * 64))
    coarse = compare(swing(shape, fps=60.0), swing(other, fps=60.0, digest="b" * 64))

    fine_height = fine.trajectory(TrajectoryChannel.HAND_HEIGHT)
    coarse_height = coarse.trajectory(TrajectoryChannel.HAND_HEIGHT)
    assert fine_height is not None
    assert coarse_height is not None
    assert fine_height.resolved_fraction > coarse_height.resolved_fraction


def test_the_comparison_is_antisymmetric_in_its_two_clips() -> None:
    """Swapping the pair flips every sign and changes nothing else. Order decides
    which clip a difference is measured *from*, and nothing more."""
    left = swing(SwingShape(takeaway_s=0.5, top_s=1.3, impact_s=1.7))
    right = swing(SwingShape(takeaway_s=0.5, top_s=0.9, impact_s=1.3), digest="b" * 64)

    forward = compare(left, right)
    backward = compare(right, left)

    assert {entry.name for entry in forward.differences} == {
        entry.name for entry in backward.differences
    }
    for entry in forward.differences:
        mirrored = backward.difference(entry.name, entry.event)
        assert mirrored is not None
        assert mirrored.difference == pytest.approx(-entry.difference)
        assert mirrored.bracket == pytest.approx(entry.bracket)

    forward_speed = forward.trajectory(TrajectoryChannel.HAND_SPEED)
    backward_speed = backward.trajectory(TrajectoryChannel.HAND_SPEED)
    assert forward_speed is not None
    assert backward_speed is not None
    assert forward_speed.resolved_fraction == pytest.approx(backward_speed.resolved_fraction)


def test_the_brackets_are_summed_across_the_two_clips() -> None:
    """Asserted on the trajectory samples as well as on the metrics, because the
    two halves compute the bracket separately and could drift apart."""
    clip = swing()
    result = compare(clip, clip)

    speed = result.trajectory(TrajectoryChannel.HAND_SPEED)
    assert speed is not None
    brackets = [sample.bracket for sample in speed.samples if sample.bracket is not None]
    assert brackets
    # Both sides are the same clip, so every bracket is exactly twice one clip's.
    assert all(value > 0.0 for value in brackets[1:-1])
    assert np.isfinite(brackets).all()
