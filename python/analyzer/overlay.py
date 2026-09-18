"""Turn filtered trajectories back into something that can be drawn on a frame.

One function, and most of its weight is in a decision the type system cannot
carry: **the overlay is drawn from the filtered landmarks, converted back to
image coordinates, and not from the estimator's stored output.**

The reason is that an overlay in this project is an instrument, not decoration.
Phase 4 built a frame-by-frame inspector because a detector reporting "impact:
frame 46" cannot be agreed or disagreed with from that sentence; Phase 5 checked
every metric it published by drawing it on the frame it came from, with
`scripts/overlay_metrics.py`. Both of those work only if the thing on screen is
the thing the number was computed from. Draw the raw landmarks instead and every
disagreement between the picture and the panel becomes unattributable -- the
reader cannot tell whether the metric is wrong, the filter moved the point, or
the overlay is lying, and a check that cannot fail informatively is not a check.

The conversion back is exact: `frame_widths_to_image` is the algebraic inverse
of the conversion that produced the measurement frame, so nothing is lost on the
return trip and no second opinion about the aspect ratio is formed.

What is *not* exact is the relationship between these points and the pixels
underneath them, whenever a calibration was applied. Undistorting moves a
landmark by up to the tens of pixels Phase 8 measured near the frame edge, and
the video element is showing the frame as recorded -- so a corrected skeleton
genuinely does not sit on the person. That is a true statement about the
footage rather than a rendering bug, and `PoseOverlay.undistorted` carries it so
the UI can say so instead of the reader discovering it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from analyzer.contracts.overlay import (
    OverlayFrame,
    OverlayPoint,
    OverlayShaft,
    OverlayState,
    PoseOverlay,
)
from analyzer.contracts.pose import POSE_CONNECTIONS, Landmark
from analyzer.coordinates import frame_widths_to_image

if TYPE_CHECKING:
    from analyzer.contracts.cache import ContentKey
    from analyzer.contracts.club import ClubTrackingReport
    from analyzer.contracts.pose import FrameGeometry
    from analyzer.filtering.landmarks import FilteredSequence

# The largest range one call will serialise. A canvas asks for a window, not a
# clip: at 33 landmarks a frame this is already about a million numbers, and the
# request that would exceed it is a caller that meant to page and did not.
MAX_OVERLAY_FRAMES = 2000


class OverlayError(ValueError):
    """An overlay was asked for that cannot be produced as asked."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def _visibility(reported: np.float64) -> float:
    """The estimator's reported visibility, as a number the contract can carry.

    NaN becomes zero. The store writes NaN for a frame the estimator returned
    nothing on, and a frame with no detection is a frame where the landmark was
    not seen -- so zero is the reading rather than a substitution, and it always
    arrives next to a `state` of `filled` or `blocked` that says the same thing
    a second way. The alternative, a nullable visibility, would put a null check
    in every consumer to express something the state already expresses.
    """
    value = float(reported)
    return 0.0 if not np.isfinite(value) else float(np.clip(value, 0.0, 1.0))


def _state(valid: bool, observed: bool) -> OverlayState:
    """Which of the three things is known about this landmark on this frame.

    `valid` is whether a position is supported at all; `observed` is whether
    this frame contributed an observation to it. The pair distinguishes a point
    the estimator found from one the filter carried across a gap, and both from
    a point nothing supports -- which is the distinction the whole enum exists
    for, since only the last is a fact about the recording.
    """
    if not valid:
        return OverlayState.BLOCKED
    return OverlayState.OBSERVED if observed else OverlayState.FILLED


def build_overlay(
    filtered: FilteredSequence,
    *,
    video_path: str,
    content_key: ContentKey,
    start_frame: int,
    end_frame: int,
    landmarks: tuple[Landmark, ...] | None = None,
    club: ClubTrackingReport | None = None,
    club_requested: bool = False,
) -> PoseOverlay:
    """Filtered landmarks over `[start_frame, end_frame)`, in drawing coordinates.

    `end_frame` is exclusive, as every frame range in this project is, and both
    ends are clamped to the clip rather than refused: a canvas asking for a
    window around the last frame is asking a reasonable question, and making it
    compute the clamp itself would put the clip length in two places.

    `club_requested` is separate from `club` being present because the two
    answer different questions, and collapsing them would lose the one that
    matters. A caller that asked for the club and got a track where the detector
    refused every frame is in a different position from one that never asked:
    the first has evidence that the club could not be seen, the second has no
    evidence at all. `ClubTrackingReport` is built around that distinction
    frame by frame, and the overlay would throw it away at the clip level.
    """
    total = int(filtered.t.size)
    start = max(0, min(start_frame, total))
    end = max(start, min(end_frame, total))

    if end - start > MAX_OVERLAY_FRAMES:
        raise OverlayError(
            f"An overlay of {end - start} frames was asked for; the limit is {MAX_OVERLAY_FRAMES}.",
            remediation=(
                f"Ask for at most {MAX_OVERLAY_FRAMES} frames at a time. A canvas draws one "
                "frame; the range exists to avoid a request per frame, not to move the clip."
            ),
        )

    wanted = tuple(landmarks) if landmarks is not None else tuple(Landmark)
    missing = [name for name in wanted if name not in filtered.landmarks]
    if missing:
        raise OverlayError(
            "This sequence was filtered without "
            f"{', '.join(entry.name.lower() for entry in missing)}, so there is nothing "
            "to draw for them.",
            remediation="Filter the whole landmark set, which is what `filter_sequence` does.",
        )

    geometry = filtered.geometry
    # Converted once for the whole requested range rather than per frame: this is
    # an array operation over (frames, 3) and doing it inside the loop would run
    # the aspect correction thirty-three times per frame for no benefit.
    drawable = {
        name: frame_widths_to_image(filtered[name].position[start:end], geometry) for name in wanted
    }

    shafts = _shafts_by_frame(club, geometry) if club is not None else {}

    frames: list[OverlayFrame] = []
    for offset in range(end - start):
        index = start + offset
        points: list[OverlayPoint] = []
        for name in wanted:
            series = filtered[name]
            position = drawable[name][offset]
            # Finiteness is folded into `valid` rather than checked separately,
            # so that the contract's invariant -- x and y are null exactly when
            # the state is blocked -- holds by construction. The filter's mask is
            # the authority and a NaN should never survive it; if one does, it
            # renders as a point nothing knows the position of, which is both
            # true and visible, rather than as a NaN on the wire that no strict
            # JSON parser would accept.
            drawn = bool(series.valid[index]) and bool(np.isfinite(position[:2]).all())
            state = _state(drawn, bool(series.observed[index]))
            blocked = state is OverlayState.BLOCKED
            points.append(
                OverlayPoint(
                    landmark=name,
                    x=None if blocked else float(position[0]),
                    y=None if blocked else float(position[1]),
                    state=state,
                    visibility=_visibility(series.visibility[index]),
                )
            )

        frames.append(
            OverlayFrame(
                frame_index=index,
                timestamp_s=float(filtered.t[index]),
                points=points,
                shaft=shafts.get(index),
            )
        )

    requested = set(wanted)
    return PoseOverlay(
        video_path=video_path,
        content_key=content_key,
        geometry=geometry,
        start_frame=start,
        end_frame=end,
        frames=frames,
        landmarks=list(wanted),
        # Filtered to what was asked for, so a caller drawing a subset never
        # receives an edge with an endpoint it has no position for.
        connections=[pair for pair in POSE_CONNECTIONS if set(pair) <= requested],
        slow_motion_factor=filtered.slow_motion_factor,
        undistorted=filtered.intrinsics is not None,
        club_tracked=club_requested or club is not None,
    )


def _shafts_by_frame(club: ClubTrackingReport, geometry: FrameGeometry) -> dict[int, OverlayShaft]:
    """The tracked shaft on each frame it was found, in drawing coordinates.

    A dict rather than a list because the report covers every frame and only
    some carry an observation, and because the overlay's own range need not be
    the track's. Frames the detector refused are simply absent, which renders as
    no club rather than as a club of zero length.
    """
    found: dict[int, OverlayShaft] = {}
    for entry in club.frames:
        shaft = entry.shaft
        if not entry.detected or shaft is None:
            continue
        ends = frame_widths_to_image(
            np.array(
                [[shaft.grip_x, shaft.grip_y, 0.0], [shaft.tip_x, shaft.tip_y, 0.0]],
                dtype=np.float64,
            ),
            geometry,
        )
        found[entry.frame_index] = OverlayShaft(
            grip_x=float(ends[0][0]),
            grip_y=float(ends[0][1]),
            tip_x=float(ends[1][0]),
            tip_y=float(ends[1][1]),
            reaches_head=shaft.reaches_head,
            confidence=shaft.confidence.overall,
        )
    return found
