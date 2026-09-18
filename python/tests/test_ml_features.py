"""What the model is fed: the grid, the normalisation, and the gaps.

Every test here is about a property that would be invisible in a training score.
A feature that leaks the future, or that means something different at a different
camera distance, produces a model that converges, evaluates cleanly and fails on
the next clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.contracts.ml import FeatureSpec
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.ml.dataset import filter_config_for
from analyzer.ml.features import (
    FEATURE_SPEC,
    VALID_CHANNEL,
    FeatureError,
    clip_features,
    grid_frame_to_index,
)
from tests import synthetic


def filtered_swing(fps: float = 120.0, **kwargs: object):  # type: ignore[no-untyped-def]
    sequence = synthetic.swing_sequence(fps=fps, **kwargs)
    return filter_sequence(sequence, filter_config_for(fps), space=LandmarkSpace.FRAME_WIDTHS)


def scaled_swing(scale: float, fps: float = 120.0):  # type: ignore[no-untyped-def]
    """The same swing filmed from further away: everything shrinks about the centre."""
    clock = np.arange(0.0, synthetic.DURATION_S, 1.0 / fps)
    x, y = synthetic.hand_path(clock)
    body = {
        landmark: (0.5 + scale * (px - 0.5), 0.5 + scale * (py - 0.5))
        for landmark, (px, py) in synthetic.BODY.items()
    }
    sequence = synthetic.pose_sequence(
        0.5 + scale * (x - 0.5), 0.5 + scale * (y - 0.5), clock, body=body
    )
    return filter_sequence(sequence, filter_config_for(fps), space=LandmarkSpace.FRAME_WIDTHS)


def test_the_array_has_one_column_per_named_channel() -> None:
    features = clip_features(filtered_swing())
    assert features.values.shape[1] == len(FEATURE_SPEC.names)
    assert features.values.shape[0] == len(features.t)


def test_every_value_is_finite() -> None:
    """A NaN reaching the optimiser destroys every weight in one step."""
    features = clip_features(filtered_swing())
    assert np.all(np.isfinite(features.values))


def test_the_grid_is_the_same_rate_whatever_the_clip_was_shot_at() -> None:
    """A dilation stack spans samples, so the samples have to mean fixed time."""
    at_60 = clip_features(filtered_swing(fps=60.0))
    at_120 = clip_features(filtered_swing(fps=120.0))

    spacing_60 = float(np.median(np.diff(at_60.t)))
    spacing_120 = float(np.median(np.diff(at_120.t)))
    assert spacing_60 == pytest.approx(1 / FEATURE_SPEC.sample_rate_hz)
    assert spacing_120 == pytest.approx(1 / FEATURE_SPEC.sample_rate_hz)
    assert abs(len(at_60) - len(at_120)) <= 1


def test_moving_the_camera_further_away_does_not_change_the_features() -> None:
    """Every channel is divided by the subject's torso, so framing cancels."""
    near = clip_features(scaled_swing(1.0))
    far = clip_features(scaled_swing(0.6))

    assert far.torso_length < near.torso_length
    overlap = min(len(near), len(far))
    difference = np.abs(near.values[:overlap] - far.values[:overlap])
    assert float(np.nanmax(difference)) < 0.02


def test_no_channel_is_scaled_by_a_statistic_of_the_clip() -> None:
    """A clip-wide peak carries the end of a clip into its first frame.

    Truncating the swing removes its fastest frames. If any channel were
    normalised by the clip's own peak, the surviving samples would change value;
    with a body-relative normalisation they do not.
    """
    full = clip_features(filtered_swing())
    cut = clip_features(filtered_swing(duration_s=synthetic.TOP_S))

    overlap = len(cut) - 2  # the last samples lose support at the clip's new end
    difference = np.abs(full.values[:overlap] - cut.values[:overlap])
    assert float(np.nanmax(difference)) < 0.02


def test_a_gap_in_tracking_invalidates_only_the_samples_inside_it() -> None:
    """Phase 3 refuses to interpolate across a gap, and resampling must not undo it."""
    detected = np.ones(int(synthetic.DURATION_S * 120), dtype=bool)
    detected[150:156] = False
    features = clip_features(filtered_swing(detected=detected))

    assert not np.all(features.valid)
    lost = ~features.valid
    assert np.all(features.values[lost, :VALID_CHANNEL] == 0.0)
    assert np.all(features.values[lost, VALID_CHANNEL] == 0.0)
    assert np.all(features.values[features.valid, VALID_CHANNEL] == 1.0)
    # The gap is six frames of a 120 fps clip; it must not swallow the clip.
    assert features.valid_fraction > 0.8


def test_a_grid_sample_maps_back_to_a_frame_of_the_source_clip() -> None:
    """A prediction has to come back in units somebody can step a video to."""
    features = clip_features(filtered_swing(fps=60.0))
    for frame in (0, 40, 120, features.frames - 1):
        index = grid_frame_to_index(features, frame)
        assert abs(int(features.source_frames[index]) - frame) <= 1


def test_source_frames_are_monotonic() -> None:
    features = clip_features(filtered_swing())
    assert np.all(np.diff(features.source_frames) >= 0)


def test_the_window_the_clip_was_smoothed_with_is_carried() -> None:
    """The grid equalises the rate and not the smoothing; the difference is recorded."""
    fast = clip_features(filtered_swing(fps=120.0))
    slow = clip_features(filtered_swing(fps=30.0))
    assert fast.window_s == pytest.approx(0.1)
    assert slow.window_s > fast.window_s


def test_a_clip_with_no_torso_is_refused_rather_than_measured_in_frame_widths() -> None:
    sequence = synthetic.swing_sequence()
    filtered = filter_sequence(
        sequence,
        space=LandmarkSpace.FRAME_WIDTHS,
        landmarks=(
            Landmark.LEFT_WRIST,
            Landmark.RIGHT_WRIST,
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
        ),
    )
    with pytest.raises(FeatureError, match="torso"):
        clip_features(filtered)


def test_a_clip_too_short_to_have_a_trajectory_is_refused() -> None:
    """And the message names the clip's length rather than its landmarks."""
    with pytest.raises(FeatureError, match=r"fewer than two samples|no trajectory"):
        clip_features(filtered_swing(duration_s=0.015, fps=120.0))


def test_a_foreign_feature_spec_is_refused_rather_than_approximated() -> None:
    """An old spec is a request for code that no longer exists."""
    other = FeatureSpec(
        names=("hand_height", "valid"),
        sample_rate_hz=30.0,
        normalisation="something else",
        interpolation="something else",
    )
    with pytest.raises(FeatureError, match="versioned by content"):
        clip_features(filtered_swing(), other)


def test_the_spec_digest_changes_with_any_part_of_the_definition() -> None:
    base = FEATURE_SPEC.digest()
    assert FEATURE_SPEC.model_copy(update={"sample_rate_hz": 30.0}).digest() != base
    assert FEATURE_SPEC.model_copy(update={"normalisation": "by peak speed"}).digest() != base
    assert FEATURE_SPEC.model_copy(update={"names": ("a", "b")}).digest() != base


def test_the_resample_floor_is_half_a_sample() -> None:
    assert FEATURE_SPEC.resample_floor_s == pytest.approx(0.5 / FEATURE_SPEC.sample_rate_hz)
    assert FEATURE_SPEC.resample_floor_s * 1000 == pytest.approx(8.333, abs=0.01)


def test_hand_height_rises_through_the_backswing() -> None:
    """The sign convention is upward-positive, and a swing is the test of it."""
    features = clip_features(filtered_swing())
    height = features.values[:, FEATURE_SPEC.names.index("hand_height")]
    address = height[grid_frame_to_index_by_time(features, synthetic.TAKEAWAY_S)]
    top = height[grid_frame_to_index_by_time(features, synthetic.TOP_S)]
    assert top > address


def grid_frame_to_index_by_time(features, seconds: float) -> int:  # type: ignore[no-untyped-def]
    return int(np.argmin(np.abs(features.t - seconds)))
