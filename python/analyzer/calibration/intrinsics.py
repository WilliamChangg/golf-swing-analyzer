"""Fitting one camera's interior geometry, and judging whether it was determined.

The fit itself is a call into OpenCV. Everything else in this module exists
because the fit's headline number -- the RMS reprojection error -- answers a
question nobody is asking.

It measures how well the model reproduces the corners it was fitted to. The
question that matters is whether those corners **determined** the model, and the
two come apart in the one case that matters: a capture where the board never
moved much. Such a set fits superbly and constrains almost nothing, so the
residual gets *better* as the calibration gets worse. A tool that reports it
alone is reporting the one number a degenerate capture improves.

Three things are therefore computed and reported separately:

    the fit        rms_reprojection_px, max_reprojection_px, per view
    the answer     fx_uncertainty and friends, from the fit's own covariance
    the cause      CoverageReport -- area, tilt, scale

and `usable` is decided on the second and third, never on the first alone.
`scripts/benchmark_calibration.py` is where the relationship between them is
measured rather than asserted.

## Which coefficients get fitted, and why that is a decision

OpenCV fits five distortion coefficients by default. A coefficient the data does
not determine does not settle harmlessly near zero: it takes whatever value best
cancels the residual over the region the board happened to occupy, and then
diverges outside it. For calibration footage shot at arm's length that region is
the middle of the frame, and "outside it" is the edge -- which is where the
distortion does all its work, and where a golf swing's hands and club head
spend most of the downswing.

So the model is named on `CalibrationConfig`, defaults to four coefficients, and
the fifth is available for a lens that needs it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.calibration.detect import BoardView
from analyzer.contracts.calibration import (
    BoardObservation,
    BoardSpec,
    CalibrationConfig,
    CalibrationQuality,
    CameraCalibration,
    CameraIntrinsics,
    CoverageReport,
    DetectionReport,
    DistortionModel,
)
from analyzer.contracts.camera import CameraRole

# Coarse grid the image is divided into when measuring how much of the frame the
# board visited. Deliberately coarse: this asks whether the lens was sampled
# across its area, not whether every pixel was touched, and a fine grid would
# report a thorough capture as sparse.
_COVERAGE_GRID = (12, 8)

# Corners beyond this fraction of the maximum image radius count as "edge".
# Distortion grows with radius and is negligible near the centre, so this is the
# share of the evidence that says anything about the distortion coefficients.
# 0.7 rather than 0.8, on the measurement: the half-diagonal is a long radius, so
# at 0.8 only the frame's actual corners qualify and a thorough capture scores
# 0.02. At 0.7 the same capture scores 0.16 and a centred one still scores zero,
# which is the separation this number exists to provide.
_EDGE_RADIUS_FRACTION = 0.7


class CalibrationError(ValueError):
    """A calibration could not be attempted."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def _flags(model: DistortionModel) -> int:
    """OpenCV flags that fit exactly the coefficients a model declares."""
    if model is DistortionModel.PINHOLE:
        return cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    if model is DistortionModel.RADIAL_TANGENTIAL_4:
        return cv2.CALIB_FIX_K3
    return 0


def _tilt_deg(rotation: NDArray[np.float64]) -> float:
    """How far off square to the camera a board view sat, in degrees.

    The board's normal is its own z axis; rotated into the camera frame, its
    component along the optical axis is the cosine of the tilt. Taken as an
    absolute value because the board has two faces and only one of them is
    printed on -- a normal pointing towards the camera and one pointing away
    describe the same view.
    """
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))


def coverage(
    views: list[BoardView],
    image_size: tuple[int, int],
    tilts_deg: list[float],
) -> CoverageReport:
    """What the views sampled: area, tilt spread and scale spread.

    The diagnostic half of a calibration, and the half that is *not* derivable
    from the residual. Each field detects a degeneracy that makes the residual
    look better rather than worse; the module docstring says which.
    """
    width, height = image_size
    columns, rows = _COVERAGE_GRID
    occupied: set[tuple[int, int]] = set()
    edge_corners = 0
    total_corners = 0

    centre = np.array([width / 2.0, height / 2.0])
    max_radius = float(np.hypot(width / 2.0, height / 2.0))

    for view in views:
        points = view.corners_px.reshape(-1, 2)
        total_corners += points.shape[0]
        for x, y in points:
            column = int(np.clip(x / width * columns, 0, columns - 1))
            row = int(np.clip(y / height * rows, 0, rows - 1))
            occupied.add((column, row))
        radii = np.linalg.norm(points - centre, axis=1)
        edge_corners += int(np.count_nonzero(radii >= _EDGE_RADIUS_FRACTION * max_radius))

    extents = [view.extent_px for view in views if view.extent_px > 0.0]
    scale_range = max(extents) / min(extents) if extents else 1.0
    tilt_range = (max(tilts_deg) - min(tilts_deg)) if tilts_deg else 0.0

    return CoverageReport(
        views=len(views),
        corners=total_corners,
        image_fraction=len(occupied) / float(columns * rows),
        edge_fraction=(edge_corners / total_corners) if total_corners else 0.0,
        tilt_range_deg=tilt_range,
        scale_range=max(scale_range, 1.0),
        methodology=(
            f"Area is the fraction of a {columns}x{rows} grid over the frame containing at "
            f"least one detected corner. Edge fraction is the share of corners beyond "
            f"{_EDGE_RADIUS_FRACTION:.0%} of the maximum image radius, which is where lens "
            "distortion is measurable at all. Tilt is the spread between the least and most "
            "oblique board view, taken from the board poses the fit itself recovered; scale "
            "is the largest apparent board diagonal over the smallest."
        ),
    )


def _residuals(
    views: list[BoardView],
    spec: BoardSpec,
    matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    rvecs: tuple[NDArray[np.float64], ...],
    tvecs: tuple[NDArray[np.float64], ...],
) -> tuple[float, float, list[float]]:
    """Reproject every corner and measure how far each one landed from its detection.

    OpenCV's extended calibration already returns a per-view RMS, and this
    recomputes it. Both are kept because they answer different questions: the
    per-view number says which view is bad, and the per-*corner* maximum this
    produces says whether a view is uniformly mediocre or has one misdetected
    corner in it. Those call for different responses -- reshoot, or drop a frame
    -- and an RMS cannot distinguish them.
    """
    squared_total = 0.0
    count = 0
    worst = 0.0
    per_view: list[float] = []

    for view, rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(view.object_corners(spec), rvec, tvec, matrix, distortion)
        observed = view.corners_px.reshape(-1, 2)
        errors = np.linalg.norm(projected.reshape(-1, 2) - observed, axis=1)
        squared_total += float(np.sum(errors**2))
        count += errors.size
        worst = max(worst, float(np.max(errors)))
        per_view.append(float(np.sqrt(np.mean(errors**2))))

    rms = float(np.sqrt(squared_total / count)) if count else float("nan")
    return rms, worst, per_view


def _gate(
    intrinsics: CameraIntrinsics,
    quality: CalibrationQuality,
    config: CalibrationConfig,
) -> tuple[bool, str | None, list[str]]:
    """Decide whether this calibration may be used, and say why not.

    Ordered so that the first refusal a reader sees is the one whose fix comes
    first. A gross failure (the residual bound) means the board or the
    dictionary is wrong and the capture was never going to work; the coverage
    bounds mean the capture was the wrong shape; the uncertainty bound is the
    summary of those, and is last because it is the symptom rather than the
    instruction.
    """
    warnings: list[str] = []

    if quality.rms_reprojection_px > config.max_rms_reprojection_px:
        return (
            False,
            (
                f"The fit leaves an RMS reprojection error of "
                f"{quality.rms_reprojection_px:.2f} px, above the "
                f"{config.max_rms_reprojection_px:.2f} px bound. An error this large is "
                "not a mediocre calibration but a wrong one: the usual causes are a board "
                "whose square size or square count does not match the printed sheet, or a "
                "sheet that is not mounted flat."
            ),
            warnings,
        )

    if quality.coverage.tilt_range_deg < config.min_tilt_range_deg:
        return (
            False,
            (
                f"The board was held within {quality.coverage.tilt_range_deg:.0f} degrees of "
                f"one orientation throughout, and {config.min_tilt_range_deg:.0f} degrees of "
                "spread is required. Held square to the camera, a board cannot separate "
                "focal length from distance -- a longer lens further away makes the same "
                "picture -- so the fit is free to choose badly while fitting well. Tilt the "
                "board substantially between views."
            ),
            warnings,
        )

    if quality.coverage.image_fraction < config.min_image_fraction:
        return (
            False,
            (
                f"The board visited {quality.coverage.image_fraction:.0%} of the frame, below "
                f"the {config.min_image_fraction:.0%} required. Lens distortion is measurable "
                "only where the board went, and it is strongest at the edges and corners -- "
                "so a calibration from a centred capture is extrapolating over exactly the "
                "region a swing is filmed in."
            ),
            warnings,
        )

    if intrinsics.focal_disagreement > config.max_focal_disagreement:
        return (
            False,
            (
                f"The horizontal and vertical focal lengths disagree by "
                f"{intrinsics.focal_disagreement:.1%}, above the "
                f"{config.max_focal_disagreement:.1%} bound. Consumer sensors have square "
                "pixels, so these two describe one physical quantity and should agree; a "
                "real difference is evidence of a poor fit rather than of an unusual camera."
            ),
            warnings,
        )

    if intrinsics.fx_uncertainty is not None:
        ratio = intrinsics.fx_uncertainty / intrinsics.fx
        if ratio > config.max_focal_uncertainty_ratio:
            return (
                False,
                (
                    f"The focal length is determined to {ratio:.1%} of itself "
                    f"({intrinsics.fx_uncertainty:.1f} px of {intrinsics.fx:.0f} px), outside "
                    f"the {config.max_focal_uncertainty_ratio:.1%} bound. This is the number "
                    "the reprojection error cannot tell you: the model fits these views, and "
                    "these views did not pin it down. More tilt and more of the frame covered "
                    "are what change it."
                ),
                warnings,
            )
    else:
        warnings.append(
            "The fit did not report parameter uncertainties, so how well these views "
            "determined the focal length is unknown. The reprojection error below says "
            "only that the model fits them."
        )

    if quality.coverage.edge_fraction < 0.05:
        warnings.append(
            f"Only {quality.coverage.edge_fraction:.1%} of corners lie near the frame edge, "
            "so the distortion coefficients rest on very little evidence. They are being "
            "applied out there regardless, which is where they do all their work."
        )

    if quality.coverage.scale_range < 1.5:
        warnings.append(
            f"Every view was taken at close to one distance (a spread of "
            f"{quality.coverage.scale_range:.2f}x in apparent board size). The calibration "
            "describes the lens at that working distance and is less well determined away "
            "from it."
        )

    if quality.degrees_of_freedom <= 0:
        warnings.append(
            "The fit has no spare degrees of freedom, so its residual is zero by "
            "construction and is not evidence of anything."
        )

    return True, None, warnings


def calibrate_intrinsics(
    views: list[BoardView],
    spec: BoardSpec,
    *,
    role: CameraRole,
    source: str,
    detection: DetectionReport | None = None,
    config: CalibrationConfig | None = None,
    notes: str = "",
) -> CameraCalibration:
    """Fit one camera's intrinsics from a set of board views.

    Raises `CalibrationError` when a fit cannot be attempted at all -- too few
    views, or views of two different frame sizes. It **returns** a calibration
    with `usable` false when the fit ran and its result should not be trusted,
    because that case carries information a caller needs: the numbers, and which
    bound they failed. An exception would throw both away.
    """
    resolved = config or CalibrationConfig()

    usable_views = [view for view in views if len(view) >= resolved.min_corners_per_view]
    if len(usable_views) < resolved.min_views:
        raise CalibrationError(
            f"{len(usable_views)} board view(s) with at least "
            f"{resolved.min_corners_per_view} corners; {resolved.min_views} are required.",
            remediation=(
                "Capture more footage of the board, moving it between shots so the views "
                "differ: across the frame, towards and away from the camera, and tilted."
            ),
        )

    sizes = {view.image_size for view in usable_views}
    if len(sizes) > 1:
        listed = ", ".join(f"{width}x{height}" for width, height in sorted(sizes))
        raise CalibrationError(
            f"The board views come from frames of different sizes ({listed}).",
            remediation=(
                "Calibrate each capture setting separately. Intrinsics are in pixels, so "
                "one set of numbers cannot describe two frame sizes."
            ),
        )

    image_size = usable_views[0].image_size
    object_corners = [
        view.object_corners(spec).astype(np.float32).reshape(-1, 1, 3) for view in usable_views
    ]
    image_corners = [view.corners_px.astype(np.float32).reshape(-1, 1, 2) for view in usable_views]

    fitted = cv2.calibrateCameraExtended(
        object_corners,
        image_corners,
        image_size,
        # OpenCV initialises both from the data unless CALIB_USE_INTRINSIC_GUESS
        # is set, which it is not; zeros are placeholders it overwrites.
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
        flags=_flags(resolved.distortion_model),
    )

    # Coerced on the way out of OpenCV rather than used as returned. Everything
    # below is float64 arithmetic against float64 contracts, and a silently
    # float32 camera matrix propagates as a rounding difference nothing reports.
    matrix = np.asarray(fitted[1], dtype=np.float64)
    distortion = np.asarray(fitted[2], dtype=np.float64)
    rvecs = tuple(np.asarray(entry, dtype=np.float64) for entry in fitted[3])
    tvecs = tuple(np.asarray(entry, dtype=np.float64) for entry in fitted[4])

    coefficients = resolved.distortion_model.coefficients
    deviations = np.asarray(fitted[5], dtype=np.float64).reshape(-1)

    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=[float(value) for value in distortion.reshape(-1)[:coefficients]],
        model=resolved.distortion_model,
        image_width=image_size[0],
        image_height=image_size[1],
        fx_uncertainty=float(deviations[0]) if deviations.size > 3 else None,
        fy_uncertainty=float(deviations[1]) if deviations.size > 3 else None,
        cx_uncertainty=float(deviations[2]) if deviations.size > 3 else None,
        cy_uncertainty=float(deviations[3]) if deviations.size > 3 else None,
    )

    rms, worst, per_view = _residuals(usable_views, spec, matrix, distortion, rvecs, tvecs)
    tilts = [_tilt_deg(np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)) for rvec in rvecs]

    observations = _rebuild_observations(usable_views, per_view, tilts, tvecs, detection)

    # Two coordinates per corner observed, against four intrinsics, the fitted
    # distortion coefficients, and six extrinsics per view.
    parameters = 4 + coefficients + 6 * len(usable_views)
    quality = CalibrationQuality(
        rms_reprojection_px=rms,
        max_reprojection_px=worst,
        per_view_rms_px=per_view,
        coverage=coverage(usable_views, image_size, tilts),
        degrees_of_freedom=2 * sum(len(view) for view in usable_views) - parameters,
    )

    usable, refusal, warnings = _gate(intrinsics, quality, resolved)

    return CameraCalibration(
        role=role,
        intrinsics=intrinsics,
        quality=quality,
        detection=detection
        or DetectionReport(
            frames_scanned=len(views),
            frames_with_board=len(views),
            views_used=len(usable_views),
            corners_total=sum(len(view) for view in usable_views),
            board=spec,
            observations=observations,
        ),
        calibrated_at=datetime.now(UTC),
        source=source,
        notes=notes,
        usable=usable,
        refusal=refusal,
        warnings=warnings,
    )


def _rebuild_observations(
    views: list[BoardView],
    per_view_rms: list[float],
    tilts: list[float],
    tvecs: tuple[NDArray[np.float64], ...],
    detection: DetectionReport | None,
) -> list[BoardObservation]:
    """Fold the fit's per-view results back onto the detection records.

    The detection report already lists every frame the board was found in,
    including the ones selection dropped. This adds what only the fit knows --
    each used view's residual, tilt and metric distance -- without losing the
    dropped ones, because 8.4 draws them all and the dropped views are how a
    person sees that their capture was repetitive.
    """
    measured = {
        view.frame: (rms, tilt, float(np.asarray(tvec).reshape(-1)[2]))
        for view, rms, tilt, tvec in zip(views, per_view_rms, tilts, tvecs, strict=True)
    }

    existing = detection.observations if detection is not None else []
    if not existing:
        existing = [
            BoardObservation(
                frame=view.frame,
                corners=len(view),
                centroid_x=view.centroid[0],
                centroid_y=view.centroid[1],
                used=True,
            )
            for view in views
        ]

    updated: list[BoardObservation] = []
    for entry in existing:
        found = measured.get(entry.frame)
        if found is None:
            updated.append(entry)
            continue
        rms, tilt, distance = found
        updated.append(
            entry.model_copy(
                update={
                    "reprojection_rms_px": rms,
                    "tilt_deg": tilt,
                    "distance_m": distance,
                    "used": True,
                }
            )
        )

    if detection is not None:
        detection.observations = updated
    return updated
