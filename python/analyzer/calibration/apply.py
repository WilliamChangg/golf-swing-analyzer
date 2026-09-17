"""Using a calibration: removing the lens, and turning a pixel into a direction.

Two operations, and the distance between them is the whole of what Phase 8 can
and cannot do.

**Undistortion** moves a detected point to where a pinhole camera would have put
it. It is a correction *within* the image plane -- pixels in, pixels out -- and
it makes every projected measurement above it more nearly true, because the
measurements assume straight lines project to straight lines and a real lens
makes that false. This is the part that pays off immediately: every angle,
distance and speed Phases 4 to 6 compute is taken from landmark positions that
the lens has displaced, by tens of pixels near the frame edge on an ordinary
phone.

**Bearing** turns a pixel into a unit vector in the camera's frame. That is the
most a single calibrated camera can produce, and it is not a position: the ray
says which direction the point lay in and nothing about how far along it sat.
Two rays from two calibrated cameras meet at a point, which is Phase 9; one ray
does not meet anything.

The module refuses rather than guesses in the two cases where a plausible answer
would be wrong:

* a calibration whose frame size does not match the footage, because rescaling
  intrinsics is right only if the sensor was scaled rather than cropped and
  nothing in a video file says which happened;
* a request for a metric 3D point, which is `CalibrationStatus.STEREO`'s job and
  is refused here by having no function that offers it.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.calibration import CameraIntrinsics
from analyzer.contracts.pose import FrameGeometry


class CalibrationMismatchError(ValueError):
    """A calibration was applied to footage it does not describe."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def require_applicable(intrinsics: CameraIntrinsics, geometry: FrameGeometry) -> None:
    """Refuse a calibration that was not measured on frames of this size.

    The most common way to get a wrong answer from a correct calibration, and
    the only part of the problem a file can be checked for. Zoom, lens choice
    and digital stabilisation are equally capable of invalidating a calibration
    and leave no trace in the frame size, which is why `CameraCalibration.notes`
    exists and why this check is necessary rather than sufficient.
    """
    if not intrinsics.applies_to(geometry.width, geometry.height):
        raise CalibrationMismatchError(
            f"The calibration was measured on {intrinsics.image_width}x"
            f"{intrinsics.image_height} frames and this clip is "
            f"{geometry.width}x{geometry.height}.",
            remediation=(
                "Calibrate the camera at the setting this footage was shot at. Intrinsics "
                "are in pixels, and rescaling them is only correct if the sensor was scaled "
                "rather than cropped -- which the file does not record."
            ),
        )


def undistort_pixels(
    points_px: NDArray[np.float64], intrinsics: CameraIntrinsics
) -> NDArray[np.float64]:
    """Move detected pixel positions to where a pinhole camera would put them.

    Takes and returns `(..., 2)` in pixels. NaN passes through as NaN: an
    undetected landmark must stay undetected, and OpenCV would otherwise turn it
    into a coordinate. That is the same rule the rest of the pipeline follows --
    one representation of absence, carried all the way to the layer that decides
    what to do about it.

    The output is expressed through the **same** camera matrix rather than
    through OpenCV's "optimal new camera matrix". Using a new matrix would
    change the focal length and principal point, so every downstream quantity
    would be in a slightly different frame than the one the calibration
    describes, and the difference is small enough to survive review.
    """
    array = np.asarray(points_px, dtype=np.float64)
    shape = array.shape
    flat = array.reshape(-1, 2)

    finite = np.isfinite(flat).all(axis=1)
    result = np.full_like(flat, np.nan)
    if not np.any(finite):
        return result.reshape(shape)

    matrix = intrinsics.matrix()
    corrected = cv2.undistortPoints(
        flat[finite].reshape(-1, 1, 2),
        matrix,
        intrinsics.distortion_vector(),
        P=matrix,
    )
    result[finite] = corrected.reshape(-1, 2)
    return result.reshape(shape)


def undistort_normalized(
    points: NDArray[np.float64], intrinsics: CameraIntrinsics, geometry: FrameGeometry
) -> NDArray[np.float64]:
    """Undistort landmarks that are normalised to the frame, in place of pixels.

    The form the pose layer holds: IMAGE coordinates on [0, 1] in each axis,
    divided by width and height respectively. Converting to pixels, undistorting
    and converting back keeps the caller in the frame it already uses, so
    undistortion can be inserted into the pipeline without moving the boundary
    between coordinate systems -- which is where sign and scale errors live.

    A `(..., 3)` array is accepted and its third channel passes through
    untouched. That channel is MediaPipe's depth estimate, which is not a
    measurement, is not in this camera's geometry, and must not be silently
    treated as one by a function that has just acquired real geometry.
    """
    require_applicable(intrinsics, geometry)

    array = np.asarray(points, dtype=np.float64).copy()
    scale = np.array([geometry.width, geometry.height], dtype=np.float64)

    pixels = array[..., :2] * scale
    array[..., :2] = undistort_pixels(pixels, intrinsics) / scale
    return array


def bearings(points_px: NDArray[np.float64], intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    """Unit direction vectors in the camera's frame, one per pixel.

    The most a single calibrated camera produces. `(..., 2)` in, `(..., 3)` out,
    each row a unit vector pointing from the optical centre towards whatever
    made that pixel. It is **not** a position and cannot be turned into one
    here: the distance along the ray is exactly the information the projection
    destroyed, and recovering it needs a second view of the same instant.

    Returned normalised rather than as the conventional z = 1 form because a
    unit vector is the honest shape for a direction -- the z = 1 form invites
    being read as a point one metre away, which is how a bearing becomes a
    fabricated 3D coordinate.
    """
    array = np.asarray(points_px, dtype=np.float64)
    flat = array.reshape(-1, 2)

    finite = np.isfinite(flat).all(axis=1)
    result = np.full((flat.shape[0], 3), np.nan)
    if not np.any(finite):
        return result.reshape(*array.shape[:-1], 3)

    normalised = cv2.undistortPoints(
        flat[finite].reshape(-1, 1, 2),
        intrinsics.matrix(),
        intrinsics.distortion_vector(),
    ).reshape(-1, 2)

    rays = np.concatenate((normalised, np.ones((normalised.shape[0], 1))), axis=1)
    result[finite] = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    return result.reshape(*array.shape[:-1], 3)


def distortion_displacement(intrinsics: CameraIntrinsics, samples: int = 24) -> tuple[float, float]:
    """How far the lens moves a pixel: at the frame edge, and at worst.

    Returns (edge, maximum) in pixels, over a grid across the frame. This is the
    number that says whether undistortion is worth doing on a given camera, and
    it is worth reporting rather than assuming: a phone's main camera displaces
    a corner pixel by tens of pixels, and a long lens by less than one. The
    same correction is therefore either a real improvement to every measurement
    above it or an expensive no-op, and only the calibration knows which.
    """
    width, height = intrinsics.image_width, intrinsics.image_height
    xs = np.linspace(0, width - 1, samples)
    ys = np.linspace(0, height - 1, samples)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    moved = np.linalg.norm(undistort_pixels(grid, intrinsics) - grid, axis=1)

    centre = np.array([width / 2.0, height / 2.0])
    radius = np.linalg.norm(grid - centre, axis=1)
    edge = moved[radius >= 0.9 * radius.max()]

    return float(edge.mean()) if edge.size else 0.0, float(moved.max())
