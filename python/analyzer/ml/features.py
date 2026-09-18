"""What the model is fed, and why each channel is the shape it is.

Ten channels per sample, on a fixed grid in real seconds. Three decisions did
most of the work here, and each of them is a place where the obvious thing would
have produced a model that scored well and meant nothing.

## Everything is divided by the subject's torso, not by the clip

A channel normalised by the clip's own peak hand speed carries a fact about the
end of the clip into its first frame. Offline that is merely odd; the moment
anyone runs this while a camera is recording it is impossible, and -- worse --
the offline score never shows the difference, so the failure appears only when
the model is put somewhere it matters. Torso length is a property of the body, so
every channel here means the same thing at frame zero as at the finish, and the
same thing for a tall player as a short one.

## The origin is the hips, and Phase 4 deliberately made the opposite choice

`phases/signals.py` refuses hip-local coordinates and measures hand height in the
frame, because its rule is about the highest point the hands reach and a moving
origin makes "highest" mean something else. A learned detector needs the opposite
thing: where the hands are **relative to the body**, with the camera removed.
Measured in the frame, a handheld clip that drifts upward through the downswing
presents rising hands the model has no way to distinguish from a real one -- and
Phase 11 found exactly that drift, 11 px of it, on real footage. The two layers
want different quantities, and each says which it wants.

## Every clip is resampled onto one grid in real seconds

A dilated convolution stack spans a fixed number of *samples*. Without a common
rate, a stack with a 127-sample receptive field sees 2.1 s of a 60 fps clip and
0.53 s of a 240 fps one, so the same weights would be looking at a whole swing in
one clip and at part of a downswing in another. The grid is in **real** seconds,
with the slow-motion factor already divided out by the layer below, so an 8x
slow-motion clip and a real-time one present the same swing at the same rate.

`SAMPLE_RATE_HZ` is 60. Above the 30 fps most footage arrives at, so an ordinary
clip is upsampled -- which invents nothing, the values between two samples of an
already-smoothed trajectory are the fit's own -- and below the 120-240 fps of the
clips worth capturing, which is a genuine loss and is what `resample_floor_s`
reports. It costs half a sample, 8.3 ms, on the placement of every event, and
that is larger than the 4.2 ms bracket a ball departure gives at 240 fps. A
learned detector is not a replacement for seeing the ball.

## Gaps are not interpolated across

Phase 3 refuses to interpolate a landmark through frames where it was not seen,
and that refusal has to survive the resampling or it was never a refusal. A grid
sample is valid only if **both** source samples bracketing it are valid, so a
single lost frame invalidates the grid samples inside it and nothing wider. The
model is told where it is blind through the `valid` channel rather than being
handed a plausible number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.ml import FEATURE_SCHEMA_VERSION, FeatureSpec
from analyzer.contracts.pose import Landmark
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.phases.signals import SignalError, swing_signals

SAMPLE_RATE_HZ = 60.0

FEATURE_SPEC = FeatureSpec(
    version=FEATURE_SCHEMA_VERSION,
    names=(
        "hand_height",
        "hand_lateral",
        "hand_speed",
        "hand_vx",
        "hand_vy",
        "hand_accel",
        "shoulder_tilt",
        "hip_tilt",
        "torsion",
        "valid",
    ),
    sample_rate_hz=SAMPLE_RATE_HZ,
    normalisation=(
        "Positions are relative to the hip midpoint and divided by the clip's "
        "median torso length; velocities and accelerations are in torso lengths "
        "per real second; angles are degrees divided by 90. Nothing is scaled by "
        "a statistic of the clip itself."
    ),
    interpolation=(
        "Linear onto a uniform grid in real seconds. A grid sample is valid only "
        "where both bracketing source samples are valid, so no value is "
        "interpolated across a frame in which the landmark was not seen."
    ),
)

# Channel index of the validity flag, used by the dataset and the model to know
# where the subject was not tracked. Derived from the spec rather than written
# twice, so adding a channel cannot leave it pointing at the wrong column.
VALID_CHANNEL = FEATURE_SPEC.names.index("valid")


class FeatureError(RuntimeError):
    """A clip cannot be turned into features, and guessing would be worse."""


@dataclass(frozen=True)
class ClipFeatures:
    """One clip on the canonical grid.

    `source_frames` is what makes a prediction sayable in the units a person can
    check: the model works on the grid, and every grid sample records the frame
    of the original clip nearest to it, so a predicted boundary comes back as a
    frame number somebody can step to. The mapping is nearest-neighbour and
    therefore costs up to half a grid sample, which `FeatureSpec.resample_floor_s`
    reports and every evaluation adds into its noise floor.
    """

    spec: FeatureSpec
    t: NDArray[np.float64]
    """Grid times in real seconds, starting at the clip's first sample."""
    values: NDArray[np.float32]
    """(samples, channels), finite everywhere. Invalid samples are zero and the
    `valid` channel says so."""
    valid: NDArray[np.bool_]
    source_frames: NDArray[np.int64]
    torso_length: float
    frames: int
    """Frames in the source clip, which is what label frame indices refer to."""
    window_s: float
    """The smoothing window these channels were fitted with, in real seconds.

    Carried because the grid equalises the sample rate and **does not** equalise
    the smoothing. A 30 fps clip cannot support the 0.10 s window the filter
    defaults to -- there are three samples in it and the degree-4 fit needs five --
    so it has to be widened, and a wider window flattens the velocity peak. Two
    clips of the same swing at different frame rates therefore arrive at the model
    with differently-shaped speed channels. That is a real confound, it is a
    property of the footage rather than of this layer, and it is recorded here so
    a dataset can report it instead of averaging over it silently."""

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def valid_fraction(self) -> float:
        if self.valid.size == 0:
            return 0.0
        return float(np.count_nonzero(self.valid) / self.valid.size)


def _midpoint(
    filtered: FilteredSequence, left: Landmark, right: Landmark
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Midpoint of a pair of landmarks, and where both were tracked."""
    first = filtered.landmarks.get(left)
    second = filtered.landmarks.get(right)
    if first is None or second is None:
        raise FeatureError(
            f"Features need {left.name} and {right.name}, and this sequence was "
            "filtered with a landmark subset that excludes one of them."
        )
    return (first.position + second.position) / 2.0, first.valid & second.valid


def _source_channels(filtered: FilteredSequence) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The ten channels on the clip's own samples, plus where they are supported.

    Assembled from `swing_signals`, which is Phase 4's own reduction of a filtered
    sequence. Reusing it is not only economy: it means the learned detector and
    the rule-based one are looking at the same hand, chosen by the same rule, so a
    difference between them is a difference in what they do with the signal rather
    than in which wrist they happened to pick.
    """
    signals = swing_signals(filtered)
    hand = signals.hand
    count = len(signals)

    torso = signals.torso_length
    if not np.isfinite(torso) or torso <= 0.0:
        raise FeatureError(
            "The subject's torso was never tracked, so there is no body scale to "
            "normalise by. Every channel would be in frame widths, which means "
            "something different at every camera distance."
        )

    hips, hips_valid = _midpoint(filtered, Landmark.LEFT_HIP, Landmark.RIGHT_HIP)
    body = hand.valid & hips_valid

    offset = (hand.position - hips) / torso
    velocity = hand.velocity / torso
    acceleration = np.full_like(velocity, np.nan)
    left_wrist = filtered.landmarks.get(Landmark.LEFT_WRIST)
    right_wrist = filtered.landmarks.get(Landmark.RIGHT_WRIST)
    if left_wrist is not None and right_wrist is not None:
        acceleration = (left_wrist.acceleration + right_wrist.acceleration) / (2.0 * torso)

    channels = np.full((count, len(FEATURE_SPEC.names)), np.nan, dtype=np.float64)
    channels[:, 0] = offset[:, 1]
    channels[:, 1] = offset[:, 0]
    channels[:, 2] = np.linalg.norm(velocity[:, :2], axis=1)
    channels[:, 3] = velocity[:, 0]
    channels[:, 4] = velocity[:, 1]
    channels[:, 5] = np.linalg.norm(acceleration[:, :2], axis=1)
    channels[:, 6] = signals.shoulder_angle_deg / 90.0
    channels[:, 7] = signals.hip_angle_deg / 90.0
    channels[:, 8] = (signals.shoulder_angle_deg - signals.hip_angle_deg) / 90.0
    channels[:, VALID_CHANNEL] = 1.0

    # The angle channels are independent of the hand: a clip can lose a wrist and
    # keep its shoulders. They are still gated on `body` below, because a sample
    # the model is told is invalid must not carry values it might learn from.
    valid = body & np.all(np.isfinite(channels[:, :VALID_CHANNEL]), axis=1)
    return channels, valid


def _grid(t: NDArray[np.float64], rate_hz: float) -> NDArray[np.float64]:
    """Uniform sample times covering the clip, in real seconds."""
    if t.size < 2:
        raise FeatureError(
            f"A clip of {t.size} frame(s) has no trajectory to resample. Features "
            "describe how a body moves through time."
        )
    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span <= 0.0:
        raise FeatureError("The clip's timestamps do not advance, so it has no time base.")
    count = int(np.floor(span * rate_hz)) + 1
    if count < 2:
        raise FeatureError(
            f"A clip spanning {span:.3f} s produces fewer than two samples at "
            f"{rate_hz:g} Hz. It is shorter than the grid can describe."
        )
    return t[0] + np.arange(count, dtype=np.float64) / rate_hz


def clip_features(filtered: FilteredSequence, spec: FeatureSpec = FEATURE_SPEC) -> ClipFeatures:
    """Turn a filtered pose sequence into the model's input.

    Raises `FeatureError` rather than returning a degraded array whenever the
    clip cannot support the channels: no torso, no time base, too short. A
    dataset that quietly contained rows of zeros for such clips would train a
    model to associate "nothing was tracked" with whatever those clips were
    labelled, which is the sort of correlation that survives a held-out split.
    """
    if spec.names != FEATURE_SPEC.names or spec.sample_rate_hz != FEATURE_SPEC.sample_rate_hz:
        raise FeatureError(
            f"This build computes feature spec {FEATURE_SPEC.digest()[:12]}; it was asked "
            f"for {spec.digest()[:12]}. Features are versioned by content, so an old "
            "spec is a request for code that no longer exists rather than for an option."
        )

    # The grid first, because it is the cheaper check and the better message: a
    # clip too short to hold a trajectory has no torso either, and "the subject's
    # torso was never tracked" would send a reader looking at their landmarks
    # instead of at the two-frame clip they passed in.
    source_t = filtered.t
    grid = _grid(source_t, spec.sample_rate_hz)

    try:
        channels, valid = _source_channels(filtered)
    except SignalError as exc:
        raise FeatureError(str(exc)) from exc

    # Index of the source sample at or before each grid time. `side="right"`
    # minus one puts a grid time exactly on a source sample into the interval
    # starting at that sample, which is what makes the last grid point land on
    # the final source sample rather than past it.
    lower = np.clip(np.searchsorted(source_t, grid, side="right") - 1, 0, source_t.size - 2)
    upper = lower + 1

    # No interpolation across a gap: both ends of the bracket must be real.
    grid_valid = valid[lower] & valid[upper]

    span = source_t[upper] - source_t[lower]
    weight = np.where(span > 0.0, (grid - source_t[lower]) / np.where(span > 0.0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)[:, None]

    filled = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
    values = filled[lower] * (1.0 - weight) + filled[upper] * weight
    values[~grid_valid] = 0.0
    values[:, VALID_CHANNEL] = grid_valid.astype(np.float64)

    # Nearest source frame, for reporting a prediction as a frame of the clip.
    nearest = np.where(weight[:, 0] < 0.5, lower, upper)

    return ClipFeatures(
        spec=spec,
        t=grid,
        values=values.astype(np.float32),
        valid=grid_valid,
        source_frames=nearest.astype(np.int64),
        torso_length=float(swing_signals(filtered).torso_length),
        frames=int(source_t.size),
        window_s=float(filtered.report.config.smoothing.window_s),
    )


def grid_frame_to_index(features: ClipFeatures, frame_index: int) -> int:
    """The grid sample nearest a given frame of the source clip.

    The inverse of `source_frames`, used to put a label's frame numbers onto the
    grid the model is trained on. Both directions are nearest-neighbour and both
    cost up to half a sample; doing it twice does not cost twice, because a label
    placed on the grid and read back lands on the frame it started from whenever
    the clip's own rate is at or below the grid's.
    """
    if len(features) == 0:
        raise FeatureError("An empty feature grid has no sample to map onto.")
    return int(np.argmin(np.abs(features.source_frames - int(frame_index))))
