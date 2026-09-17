"""Typed contracts for pose estimation.

The single most important thing in this module is the distinction between the
two coordinate spaces a pose estimator produces, because conflating them
produces numbers that look metric and are not.

`IMAGE` landmarks are normalised to the frame: x and y in [0, 1] across the
displayed width and height, with z on roughly the same scale as x and measured
relative to the hips. They are the only space in which a landmark can be drawn
on a video frame.

`HIP_LOCAL` landmarks are what MediaPipe calls "world landmarks". They are in
approximate metres, centred on the midpoint of the hips, and oriented to the
body rather than to the room. **They are not calibrated world coordinates.**
They carry no information about where the camera was, how far away the subject
stood, or which way the target line ran, and no metric-scale claim may rest on
them. Real world coordinates arrive in Phase 9, from stereo triangulation
against a calibrated pair; this module deliberately does not use the word
"world" for anything, so that the two can never be confused in a call site.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey

# Bump on any breaking change to the models in this module. A persisted pose
# sequence carrying a different version is refused rather than reinterpreted.
#
# 2 - added `PoseSequence.geometry`. A file written before it cannot supply the
#     frame's aspect ratio, and without that every distance mixing x and y in
#     IMAGE space is anisotropically wrong (see `FrameGeometry`). Reading such a
#     file and defaulting the aspect to 1 would produce exactly the silent error
#     the field exists to remove, so those files are refused instead.
POSE_SCHEMA_VERSION = 2


class Landmark(IntEnum):
    """The 33 body landmarks produced by MediaPipe's pose models.

    The integer values are the model's own output indices, so this enum doubles
    as the mapping: `result.pose_landmarks[0][Landmark.LEFT_WRIST]` is the left
    wrist. Keeping the values rather than renumbering means there is no
    translation table to get wrong.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


LANDMARK_COUNT = len(Landmark)

# Skeleton edges, for overlay rendering and for bone-length consistency checks
# in Phase 9. Face landmarks are deliberately omitted: they add clutter to an
# overlay and no golf metric depends on them.
POSE_CONNECTIONS: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
)


class LandmarkSpace(StrEnum):
    """Which coordinate space a set of landmarks is expressed in.

    IMAGE      - normalised to the displayed frame, x and y in [0, 1]. The only
                 space that can be drawn on a video frame.
    HIP_LOCAL  - approximate metres, centred on the hip midpoint, oriented to
                 the body. MediaPipe calls these "world landmarks"; they are not
                 calibrated world coordinates and carry no camera geometry.
    """

    IMAGE = "image"
    HIP_LOCAL = "hip_local"


class FrameGeometry(BaseModel):
    """The displayed pixel dimensions IMAGE landmarks were normalised against.

    Carried with the landmarks because IMAGE space is **anisotropic** and
    nothing downstream can discover that on its own. x is divided by the frame
    width and y by the frame height, so on a 1080x1920 clip one pixel of
    vertical travel becomes 1/1920 while one pixel of horizontal travel becomes
    1/1080: the same displacement in pixels counts for 0.5625 as much going down
    as going across. Any Euclidean quantity that mixes the two -- a distance, a
    speed, an angle -- is wrong by an amount that depends only on the shape of
    the frame, and nothing about the result looks wrong.

    One number fixes it, and it is not recoverable from the landmarks: the
    aspect ratio. It is recorded here, at the point where it is still known,
    rather than re-derived later by probing a video file that may have moved.
    """

    width: int = Field(gt=0, description="Displayed frame width in pixels, after rotation.")
    height: int = Field(gt=0, description="Displayed frame height in pixels, after rotation.")

    @property
    def aspect_ratio(self) -> float:
        """Displayed height divided by displayed width.

        The factor a normalised y must be multiplied by to put it in the same
        units as a normalised x. Both are then in **frame widths**, which is an
        isotropic unit: a displacement of n pixels measures the same whichever
        way it points.
        """
        return self.height / self.width


class LandmarkPoint(BaseModel):
    """One landmark in one space.

    `visibility` and `presence` are the model's own two confidences and mean
    different things: presence is whether the landmark is in frame at all,
    visibility is whether it is unoccluded. Both are kept because the filtering
    layer gates on them separately.
    """

    x: float
    y: float
    z: float = Field(
        description=(
            "Depth. In IMAGE space, roughly the same scale as x and relative to the "
            "hips; from a single camera it is an estimate, not a measurement."
        )
    )
    visibility: float = Field(description="Model confidence that the landmark is unoccluded.")
    presence: float = Field(description="Model confidence that the landmark is in frame.")


class PoseFrame(BaseModel):
    """The pose found in one video frame, or the record that none was.

    A frame with no detection is kept rather than dropped. The gap is itself
    information -- the filtering layer needs to know a value is missing rather
    than interpolating over an absence it cannot see -- and dropping it would
    also break the correspondence between frame index and position in the
    sequence.
    """

    frame_index: int
    timestamp_s: float = Field(description="Elapsed time from the first frame of the clip.")
    detected: bool
    image: list[LandmarkPoint] = Field(
        default_factory=list,
        description="33 landmarks in IMAGE space, or empty when nothing was detected.",
    )
    hip_local: list[LandmarkPoint] = Field(
        default_factory=list,
        description="33 landmarks in HIP_LOCAL space, or empty when nothing was detected.",
    )


class PoseModelInfo(BaseModel):
    """Which model produced a sequence, pinned so results stay attributable."""

    name: str = Field(description="Manifest name, e.g. 'pose_landmarker_full'.")
    variant: str = Field(description="lite / full / heavy.")
    precision: str = Field(description="Weight precision as the manifest records it.")
    sha256: str = Field(description="Digest of the model file actually loaded.")
    delegate: str = Field(description="Where inference ran, as measured: 'cpu' or 'gpu'.")
    min_pose_detection_confidence: float
    min_pose_presence_confidence: float
    min_tracking_confidence: float


class PoseExtractionStats(BaseModel):
    """Measured facts about one extraction run.

    `detection_rate` is a count, not a quality score. A low rate means the model
    found nobody in most frames; it says nothing about whether the landmarks it
    did find are correct, and no accuracy claim may be built on it.
    """

    frames_processed: int
    frames_detected: int
    detection_rate: float
    elapsed_s: float
    ms_per_frame: float
    mean_visibility: float | None = Field(
        default=None,
        description="Mean visibility across detected landmarks. None when nothing was detected.",
    )


class PoseSequence(BaseModel):
    """Every frame's pose for one clip, plus what produced it."""

    schema_version: int = POSE_SCHEMA_VERSION
    video_path: str
    video_content_key: ContentKey
    geometry: FrameGeometry = Field(
        description="Displayed frame size the IMAGE landmarks are normalised against."
    )
    model: PoseModelInfo
    extracted_at: datetime
    stats: PoseExtractionStats
    frames: list[PoseFrame]


class PoseExtractionResult(BaseModel):
    """What the desktop app is told about a completed extraction.

    Deliberately excludes the landmarks themselves. A 240 fps clip carries tens
    of thousands of frames of 33 landmarks in two spaces, which belongs in a
    columnar file on disk rather than in a JSON-RPC response; the app is given
    the path to read and the numbers to display.
    """

    schema_version: int = POSE_SCHEMA_VERSION
    video_path: str
    output_path: str = Field(description="Parquet file holding the landmarks.")
    model: PoseModelInfo
    extracted_at: datetime
    stats: PoseExtractionStats
    warnings: list[str] = Field(default_factory=list)
