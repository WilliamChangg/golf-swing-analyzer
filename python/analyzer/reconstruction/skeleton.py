"""Bone lengths: the check the reprojection error cannot make.

A reconstruction's residual measures how far the fitted point lands from the two
detections, and it is blind to displacement *along* the epipolar line -- that
component slides the point up or down its own ray, where a perfect fit is always
available at the wrong depth. Something else has to see it, and something does.

**A bone does not change length during a swing.** Reconstruct each frame
independently and measure the distance between two adjacent joints, and every
frame should agree. When one of them slides in depth, the distance changes.
Nothing about this needs ground truth, an anatomical table, or a labelled set:
the *variation* of a reconstructed segment across a clip is error, whatever its
absolute value turns out to be.

## Two checks, catching two different failures

**Variation** catches error that changes frame to frame, which is what noise and
occlusion produce. It is measured as the interquartile range over the median
rather than as a standard deviation over a mean, because a handful of badly
reconstructed frames should not be allowed to describe a clip that has a usable
middle -- and a clip whose worst frames are bad is exactly the ordinary case,
since a swing passes through motion blur at its fastest point.

**Left-right symmetry** compares two chains reconstructed from entirely
different landmarks with different occlusion histories. A person's two upper arms
really are the same length to within a per cent or two, so a disagreement is
evidence about the reconstruction rather than about the body -- and because it is
a comparison between *sides*, it says which side is the worse one, which a
per-segment variation does not.

It was built expecting to be the stronger of the two, on the reasoning that a
consistent depth bias would give a stably wrong length that variation could not
see. **Measured, that reasoning is wrong**, because a swing rotates the body: a
displacement constant in the camera's frame is not constant relative to the bone,
so the length moves and variation notices first. The numbers are in
`SymmetryCheck`. It is kept as corroboration and as localisation, which is what
it is, rather than as the instrument it was meant to be.

## What a length here is, and is not

`median_length_m` is the first statement this engine makes about a body in
metres, and it is not an anatomical measurement. MediaPipe's landmarks are the
model's estimate of where a joint appears, not where a joint centre is, so the
distance between two of them is not the length of the bone between them -- it is
systematically short at the shoulder, where the landmark sits on the acromion
rather than in the glenohumeral joint, and it wanders at the hip for the same
reason. Quoted so that a person can hold a tape measure against themselves and
notice a factor of two, which is what a mis-measured board square produces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.pose import POSE_CONNECTIONS, Landmark
from analyzer.contracts.reconstruction import (
    BoneConsistency,
    ReconstructionConfig,
    SymmetryCheck,
)

# Segments whose two ends both carry a side, so a left one has a right twin.
# Derived from `POSE_CONNECTIONS` rather than listed, so a connection added there
# is checked here without anyone remembering to add it twice.
_SIDED = ("LEFT_", "RIGHT_")


def _segment_name(pair: tuple[Landmark, Landmark]) -> str:
    return f"{pair[0].name.lower()}-{pair[1].name.lower()}"


def _mirror(pair: tuple[Landmark, Landmark]) -> tuple[Landmark, Landmark] | None:
    """The same segment on the other side of the body, or None if it has no side."""
    names = [end.name for end in pair]
    if not all(any(name.startswith(prefix) for prefix in _SIDED) for name in names):
        return None
    if not all(name.startswith("LEFT_") for name in names):
        return None
    try:
        return (
            Landmark[names[0].replace("LEFT_", "RIGHT_", 1)],
            Landmark[names[1].replace("LEFT_", "RIGHT_", 1)],
        )
    except KeyError:  # pragma: no cover - every LEFT_ landmark has a RIGHT_ twin
        return None


def _unsided_name(pair: tuple[Landmark, Landmark]) -> str:
    return "-".join(
        end.name.lower().replace("left_", "", 1).replace("right_", "", 1) for end in pair
    )


def segment_lengths(
    points: NDArray[np.float64],
    valid: NDArray[np.bool_],
    landmarks: tuple[Landmark, ...],
    pair: tuple[Landmark, Landmark],
) -> NDArray[np.float64]:
    """One segment's reconstructed length per frame, NaN where an end is missing.

    NaN rather than a skipped entry so the result stays aligned with the frame
    index, which is what lets a caller line a suspicious length up against the
    video frame that produced it.
    """
    index = {landmark: column for column, landmark in enumerate(landmarks)}
    first, second = index.get(pair[0]), index.get(pair[1])
    if first is None or second is None:
        return np.full(points.shape[0], np.nan, dtype=np.float64)

    both = valid[:, first] & valid[:, second]
    lengths = np.full(points.shape[0], np.nan, dtype=np.float64)
    lengths[both] = np.linalg.norm(points[both, first] - points[both, second], axis=1)
    return lengths


def bone_consistency(
    points: NDArray[np.float64],
    valid: NDArray[np.bool_],
    landmarks: tuple[Landmark, ...],
    config: ReconstructionConfig,
) -> list[BoneConsistency]:
    """Every skeletal segment's length across the clip, and how much it moved.

    Face landmarks are absent because `POSE_CONNECTIONS` omits them: they are
    centimetres apart, so their reconstructed length is almost all error, and
    including them would put a dozen meaningless rows above the ones that matter.
    """
    results: list[BoneConsistency] = []

    for pair in POSE_CONNECTIONS:
        lengths = segment_lengths(points, valid, landmarks, pair)
        measured = lengths[np.isfinite(lengths)]
        if measured.size == 0:
            continue

        median = float(np.median(measured))
        spread = float(np.subtract(*np.percentile(measured, [75, 25])))
        variation = spread / median if median > 0.0 else float("inf")

        results.append(
            BoneConsistency(
                name=_segment_name(pair),
                frames=int(measured.size),
                median_length_m=median,
                variation=variation,
                spread_m=spread,
                stable=variation <= config.max_bone_variation,
            )
        )

    return results


def symmetry(bones: list[BoneConsistency], config: ReconstructionConfig) -> list[SymmetryCheck]:
    """Pair each left segment with its right twin and compare their medians.

    Only segments that survived on both sides appear. A missing twin is not a
    failure worth reporting here -- it already shows as that landmark's coverage,
    and repeating it as an absent symmetry row would make one fact look like two.
    """
    by_name = {entry.name: entry for entry in bones}
    checks: list[SymmetryCheck] = []

    for pair in POSE_CONNECTIONS:
        twin = _mirror(pair)
        if twin is None:
            continue
        left = by_name.get(_segment_name(pair))
        right = by_name.get(_segment_name(twin))
        if left is None or right is None:
            continue

        mean = 0.5 * (left.median_length_m + right.median_length_m)
        disagreement = (
            abs(left.median_length_m - right.median_length_m) / mean if mean > 0.0 else float("inf")
        )
        checks.append(
            SymmetryCheck(
                segment=_unsided_name(pair),
                left_m=left.median_length_m,
                right_m=right.median_length_m,
                disagreement=disagreement,
                agrees=disagreement <= config.max_asymmetry,
            )
        )

    return checks
