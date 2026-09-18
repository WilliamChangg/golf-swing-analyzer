"""Typed contracts for what is drawn on top of a video frame.

Phase 14 puts a picture behind the frame indices every panel since Phase 4 has
been reporting. This module is the other half of that: what to draw on the
picture, in the coordinates it is drawn in.

**The landmarks here are the filtered ones, not the estimator's raw output, and
that is the whole reason this contract exists rather than the UI reading a pose
file.** The overlay is not decoration. It exists so that a number can be checked
against the frame it came from -- which is the argument `scripts/overlay_metrics.py`
already makes for the command line, and the argument Phase 4 made for building a
frame-by-frame inspector at all. A metric is computed from the filtered
trajectory. An overlay drawn from the raw landmarks would therefore sit in a
slightly different place from the thing being checked, and every disagreement
between the number and the picture would be unattributable: the reader cannot
tell whether the metric is wrong or the overlay is.

So the positions below come through `filter_sequence`, are undistorted when a
calibration is available, and are converted back to IMAGE coordinates for
drawing by the exact inverse of the conversion that took them out. The cost is
that this is not what the estimator saw, and `OverlayState` is what keeps that
visible: a point the filter could not support is `blocked` and is drawn as
absent rather than as a guess.

Coordinates are **IMAGE** -- x divided by the displayed width, y by the displayed
height, y increasing downward, both in [0, 1]. That is the anisotropic frame,
which is the wrong frame to measure anything in and the right one to draw in: it
maps to a canvas by multiplying by its size, with no aspect correction for a
caller to forget. Nothing here is a measurement and nothing above may treat it
as one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.pose import FrameGeometry, Landmark

# Bump on any breaking change to the models in this module.
OVERLAY_SCHEMA_VERSION = 1


class OverlayState(StrEnum):
    """How much the system actually knows about where a landmark is.

    Three values rather than a nullable position, because the two ways of not
    knowing are different and only one of them is a problem with the recording.

    OBSERVED  - the estimator reported this landmark on this frame and the
                confidence gate kept it. The position is supported by an
                observation of this frame.
    FILLED    - the position is supported, but not by an observation of this
                frame: either the gap policy bridged a short absence, or the
                local fit was carried by neighbouring frames. Drawn, and drawn
                differently, because a reader checking a metric against a frame
                should be able to see that this particular frame contributed
                nothing to it.
    BLOCKED   - no position is supported here at all, so `x` and `y` are null.
                This is the state Phase 4 found on the down-the-line clip, where
                motion blur smeared the wrists for 1.92 s exactly when they were
                moving fastest. An overlay that interpolated across it would
                hide the one thing worth seeing.
    """

    OBSERVED = "observed"
    FILLED = "filled"
    BLOCKED = "blocked"


class OverlayPoint(BaseModel):
    """One landmark on one frame, ready to be drawn.

    `x` and `y` are null exactly when `state` is `blocked`. They are not NaN:
    JSON has no NaN, and a serialiser that emitted one would produce a document
    no strict parser accepts -- so the absence is carried in the type where a
    TypeScript consumer has to handle it.
    """

    landmark: Landmark = Field(description="The model's own landmark index; see `Landmark`.")
    x: float | None = Field(description="Fraction of the displayed width. Null when blocked.")
    y: float | None = Field(
        description="Fraction of the displayed height, measured downward. Null when blocked."
    )
    state: OverlayState
    visibility: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "What the estimator reported for this landmark on this frame, carried "
            "through unchanged -- including on frames the gate rejected. It is the "
            "estimator's own opinion of itself and this project has measured it "
            "being confidently wrong: see the Post-Phase 6 note, where both "
            "shoulders scored 1.00 across frames whose implied turn varied by 27 "
            "degrees. Shown, never trusted."
        ),
    )


class OverlayShaft(BaseModel):
    """The club shaft on one frame, as a segment in drawing coordinates.

    The grip end is the hand anchor the search started from rather than
    something the detector found, exactly as `ShaftObservation` records -- so a
    reader watching the segment track the hands is watching a constraint that
    was applied, not a measurement that was obtained.

    `reaches_head` is the field to draw differently on. False means the evidence
    stopped before the end of the club, so the tip is where the image stopped
    drawing a line and **not** the club head. Drawing the two the same way would
    put a club head on screen in every frame where the club was blurred or
    pointing at the camera, which is precisely where a reader would most want to
    know that nothing was found.
    """

    grip_x: float
    grip_y: float
    tip_x: float
    tip_y: float
    reaches_head: bool = Field(
        description="Whether `tip` is the club head or merely the end of the supported evidence."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class OverlayFrame(BaseModel):
    """Everything drawn over one frame of the clip."""

    frame_index: int
    timestamp_s: float = Field(
        description=(
            "Real-clock time from the first frame, after any slow-motion factor "
            "has been divided out. On a slowed clip this is **not** where the "
            "frame sits in the video file, which is what the player seeks by -- "
            "see `SeekIndex`."
        )
    )
    points: list[OverlayPoint] = Field(
        description="One entry per landmark asked for, in ascending landmark order."
    )
    shaft: OverlayShaft | None = Field(
        default=None,
        description="The club, when it was tracked and found on this frame. Null otherwise.",
    )


class PoseOverlay(BaseModel):
    """Filtered landmarks over a range of frames, in the frame they are drawn in.

    A range rather than a clip, because the caller is a canvas: a 60 s clip at
    240 fps is 14,400 frames of 33 landmarks, and serialising all of it to draw
    one of them is half a million points to move so that thirty-three can be
    used. The range is also why `frames` carries its own indices rather than
    being positional -- a consumer holding two ranges should not have to
    remember where each one started.
    """

    schema_version: int = OVERLAY_SCHEMA_VERSION
    video_path: str
    content_key: ContentKey
    geometry: FrameGeometry = Field(
        description=(
            "The displayed pixel size the coordinates are fractions of. Carried "
            "so a canvas can size itself, and so a caller can tell that a clip is "
            "portrait without probing the file again."
        )
    )
    start_frame: int
    end_frame: int = Field(description="Exclusive, as every frame range in this project is.")
    frames: list[OverlayFrame]
    landmarks: list[Landmark] = Field(
        description="Which landmarks each frame carries, in the order they appear."
    )
    connections: list[tuple[Landmark, Landmark]] = Field(
        description=(
            "Skeleton edges to draw, from `POSE_CONNECTIONS`. Sent rather than "
            "left to the consumer so that the app cannot grow a second opinion "
            "about human anatomy; Phase 9's bone-length checks read the same "
            "tuple. Edges whose endpoints were not both requested are omitted."
        )
    )
    slow_motion_factor: float = Field(
        description="The factor the timestamps were divided by. 1.0 for an ordinary recording."
    )
    undistorted: bool = Field(
        description=(
            "Whether a lens correction was applied to these positions. When true "
            "they will not sit exactly on the raw pixels of the frame underneath, "
            "by up to the tens of pixels Phase 8 measured near the frame edge -- "
            "which is a real disagreement between the picture and the measurement, "
            "and belongs on screen rather than hidden."
        )
    )
    club_tracked: bool = Field(
        description=(
            "Whether club tracking was asked for. False means every `shaft` is "
            "null because nothing looked, which is not the same as nothing being "
            "found."
        )
    )
    warnings: list[str] = Field(default_factory=list)
