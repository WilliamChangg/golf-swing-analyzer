"""Reconstructing one swing from two calibrated views.

The entry point. Four things have to be true before a single point is produced,
and each of them is a refusal rather than a degraded answer:

1. **Both cameras are calibrated and their relative pose is measured.** That is
   `CalibrationStatus.STEREO`, and `CameraRig.status` is what decides it. One
   calibrated camera turns a pixel into a direction and nothing here can
   intersect one direction with itself.
2. **Each calibration describes the footage it is being applied to.** Frame size
   is the only part of that a file records, and a mismatch is refused rather than
   rescaled, for the reason `CameraIntrinsics.applies_to` gives.
3. **The two clocks are related.** Triangulation is only meaningful for two views
   of the same instant, so a project with no usable alignment is refused by name
   rather than reconstructed at zero offset -- which is both a plausible-looking
   answer and, as Phase 7's `SyncModel` documents, exactly the wrong default.
4. **The two clips overlap.** Reference instants that map outside the target
   clip's own timestamps produce nothing, because resampling there would be
   extrapolation.

What survives is triangulated per landmark per frame, gated on the geometry
rather than on the residual, and checked against the bones.

## The order of the gates is the argument

    visibility  ->  the estimator saw it in both views
    convergence ->  the two rays met at an angle that determines a point
    reprojection -> the two views agree about where it is
    uncertainty ->  the answer is worth quoting

Convergence comes **before** reprojection deliberately. A pair of nearly parallel
rays can be made to agree to a hundredth of a pixel by sliding the point a metre
in depth, so a point that fails the convergence test will usually pass the
reprojection test -- and refusing it for the residual it did not fail would tell
a reader the wrong thing about their capture. The fix for a shallow convergence
angle is to move a camera. The fix for a large residual is to re-synchronise or
re-calibrate. They must not be confused for each other.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.calibration import CalibrationStatus, CameraRig
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.contracts.reconstruction import (
    LandmarkReconstruction,
    PairingSummary,
    ReconstructionConfig,
    ReconstructionQuality,
    ReconstructionReport,
    RefusalReason,
)
from analyzer.contracts.sync import TimeMap
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.reconstruction import pairing as pairing_module
from analyzer.reconstruction import skeleton
from analyzer.reconstruction.triangulate import (
    StereoGeometry,
    convergence_angles,
    positional_uncertainty,
    reprojection_errors,
    triangulate,
    velocity_from_image,
)

# Landmark scatter measured from real footage in Phase 3
# (`scripts/benchmark_filter.py`), in frame widths. Used only when this clip's
# own filter residual cannot supply a figure -- which happens exactly when the
# smoothing window holds as many samples as the polynomial has coefficients, so
# the fit passes through every point and the residual is zero by construction.
# Zero is not a measurement of perfect landmarks, so it is not used as one.
_MEASURED_LANDMARK_SIGMA_FRAME_WIDTHS = 0.0014


class ReconstructionError(ValueError):
    """A reconstruction was asked for and cannot be produced."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class ReconstructedSequence:
    """One swing in metres, plus everything needed to decide whether to believe it.

    The arrays are `(frames, landmarks, ...)` on the **reference clip's** frames
    and timestamps, because the reference clock is what the time map is
    parameterised in and what every swing event was located in. A caller holding
    a frame number from Phase 4 can index straight in.

    `points` are `LandmarkSpace.CAMERA`: metres, centred on the reference camera.
    Not WORLD -- see `analyzer/contracts/reconstruction.py` for what a scene-fixed
    frame would additionally need and why guessing it is worse than naming it.
    """

    t: NDArray[np.float64]
    """Reference-clip timestamps, in real seconds."""
    points: NDArray[np.float64]
    """(frames, landmarks, 3) in metres, NaN wherever a point was refused."""
    velocity: NDArray[np.float64]
    """(frames, landmarks, 3) in metres per second, from the two views' fitted
    image velocities rather than from differencing `points`."""
    valid: NDArray[np.bool_]
    uncertainty_m: NDArray[np.float64]
    convergence_deg: NDArray[np.float64]
    reprojection_px: NDArray[np.float64]
    refusal: NDArray[np.str_]
    """`RefusalReason` values per point, empty where the point was produced.

    The report counts these per landmark, which is what a reader of numbers
    needs; a viewport draws the individual hole and has to be able to say which
    of the five it is. Kept as the strings the gates assigned rather than
    re-derived later, because every diagnostic array above is masked to NaN for a
    refused point -- so the evidence a reason could be inferred from is exactly
    what refusal removes."""
    visibility: NDArray[np.float64]
    """The weaker of the two views' reported visibilities, per point."""
    landmarks: tuple[Landmark, ...]
    geometry: StereoGeometry
    space: LandmarkSpace
    slow_motion_factor: float
    report: ReconstructionReport

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def index(self) -> dict[Landmark, int]:
        return {landmark: column for column, landmark in enumerate(self.landmarks)}

    def track(self, landmark: Landmark) -> NDArray[np.float64]:
        """One landmark's 3D trajectory, `(frames, 3)`, NaN where refused."""
        return self.points[:, self.index[landmark], :]

    def speed(self, landmark: Landmark) -> NDArray[np.float64]:
        """One landmark's speed in metres per second, `(frames,)`."""
        return np.linalg.norm(self.velocity[:, self.index[landmark], :], axis=1)


def measured_pixel_sigma(
    filtered: FilteredSequence, landmarks: tuple[Landmark, ...]
) -> tuple[float, bool]:
    """Per-view landmark uncertainty in pixels, measured from this clip.

    The filter reports, per landmark and per axis, the RMS distance between the
    raw detections and the trajectory it fitted through them. That is what the
    landmarks actually scattered by on this footage, in frame widths; multiplying
    by the frame width puts it in pixels.

    It is an **upper bound** on the fitted position's own error, and is used as
    one: smoothing averages several samples, so the fitted point is better
    determined than any single detection. Overstating the input noise overstates
    the output uncertainty, which is the direction to be wrong in.

    Returns the figure and whether it was measured. A residual of exactly zero is
    not evidence of perfect landmarks -- it is what a window holding as many
    samples as the polynomial has coefficients produces, every time -- so that
    case falls back to the scatter Phase 3 measured from real footage and says
    so.
    """
    width = float(filtered.geometry.width)
    residuals: list[float] = []
    for landmark in landmarks:
        entry = filtered.landmarks.get(landmark)
        if entry is None:
            continue
        # x and y only. The z channel is MediaPipe's own depth guess, which is
        # not a measurement in this camera's geometry and does not describe the
        # scatter of anything that is triangulated.
        residuals.extend(value for value in entry.report.residual_rms[:2] if np.isfinite(value))

    if residuals:
        measured = float(np.median(residuals)) * width
        if measured > 0.0:
            return measured, True
    return _MEASURED_LANDMARK_SIGMA_FRAME_WIDTHS * width, False


def require_stereo_rig(
    rig: CameraRig | None, reference_role: CameraRole, target_role: CameraRole
) -> None:
    """Refuse a project that cannot be triangulated, before any work is done.

    Separated from `_require_rig` so a caller can ask the question **first**. A
    reconstruction otherwise extracts poses for two clips and fits an alignment
    between them before discovering that the rig was never calibrated, and then
    reports the last thing that went wrong rather than the first: the message a
    person gets is "no extracted poses for this clip", when the actual blocker is
    a missing calibration that was knowable from one database read.

    Every check here reads the project and none reads the footage, which is what
    makes it cheap enough to run up front.
    """
    if rig is None or rig.status() is not CalibrationStatus.STEREO or rig.stereo is None:
        status = rig.status().value if rig is not None else "none"
        raise ReconstructionError(
            f"Triangulation needs both cameras calibrated and their relative pose "
            f"measured; this project has '{status}'.",
            remediation=(
                "Calibrate each camera with `analyzer calibrate camera`, then measure the "
                "pair with `analyzer calibrate stereo`. One calibrated camera turns a pixel "
                "into a direction, and two directions are what meet at a point."
            ),
        )

    stereo = rig.stereo
    ends = {stereo.reference_role, stereo.target_role}
    if {reference_role, target_role} != ends:
        raise ReconstructionError(
            f"The stereo calibration relates {stereo.reference_role.value} to "
            f"{stereo.target_role.value}, and this reconstruction is between "
            f"{reference_role.value} and {target_role.value}.",
            remediation="Reconstruct the pair the extrinsics were measured for.",
        )

    if rig.usable_camera(reference_role) is None or rig.usable_camera(target_role) is None:
        missing = reference_role if rig.usable_camera(reference_role) is None else target_role
        raise ReconstructionError(
            f"The {missing.value} camera has no usable calibration.",
            remediation="Re-calibrate it; `analyzer calibrate show` says what it failed on.",
        )


def _require_rig(
    rig: CameraRig | None,
    reference_role: CameraRole,
    target_role: CameraRole,
    reference: FilteredSequence,
    target: FilteredSequence,
) -> StereoGeometry:
    """Refuse anything short of a stereo rig that describes this footage.

    The checks above, plus the ones that need the clips: a calibration measured
    at another frame size does not describe this footage, and nothing but the
    footage says what size it is.
    """
    require_stereo_rig(rig, reference_role, target_role)
    assert rig is not None and rig.stereo is not None  # noqa: S101 - narrowed by the call above
    stereo = rig.stereo

    first = rig.usable_camera(reference_role)
    second = rig.usable_camera(target_role)
    assert first is not None and second is not None  # noqa: S101 - narrowed by the call above

    for role, camera, sequence in (
        (reference_role, first, reference),
        (target_role, second, target),
    ):
        if not camera.intrinsics.applies_to(sequence.geometry.width, sequence.geometry.height):
            raise ReconstructionError(
                f"The {role.value} calibration was measured on "
                f"{camera.intrinsics.image_width}x{camera.intrinsics.image_height} frames "
                f"and its clip is {sequence.geometry.width}x{sequence.geometry.height}.",
                remediation=(
                    "Calibrate that camera at the setting the swing was filmed at. "
                    "Rescaling intrinsics is only right if the sensor was scaled rather "
                    "than cropped, which the file does not record."
                ),
            )

    # The stereo model states its pose from its own reference role, which need
    # not be the role this reconstruction calls the reference. Inverting it here
    # is the only place that composition happens.
    if stereo.reference_role is reference_role:
        rotation = stereo.rotation_matrix()
        translation = stereo.translation_vector()
    else:
        rotation = stereo.rotation_matrix().T
        translation = -rotation @ stereo.translation_vector()

    return StereoGeometry(
        reference=first.intrinsics,
        target=second.intrinsics,
        rotation=rotation,
        translation=translation,
    )


def _pairing_summary(
    target_track: pairing_module.ViewTrack,
    sampled: pairing_module.SampledTrack,
    time_map: TimeMap,
    times: NDArray[np.float64],
    *,
    resampled: bool,
) -> PairingSummary:
    """What relating the two clocks cost, in the unit it contaminates.

    The Phase 8 conversion, applied to a subject that cannot be held still: the
    map's uncertainty in seconds times the landmark's image speed in pixels per
    second is a displacement in pixels, directly comparable with the reprojection
    error.
    """
    speeds = sampled.velocity[sampled.valid]
    magnitudes = np.linalg.norm(speeds, axis=1) if speeds.size else np.zeros(0)
    median_speed = float(np.median(magnitudes)) if magnitudes.size else 0.0
    max_speed = float(np.max(magnitudes)) if magnitudes.size else 0.0

    uncertainties = [
        value
        for value in (time_map.uncertainty_at(float(instant)) for instant in times)
        if value is not None
    ]
    sync_s = float(np.median(uncertainties)) if uncertainties else None
    interval_s = target_track.median_interval_s

    return PairingSummary(
        method="resampled" if resampled else "nearest_frame",
        resampled_frames=int(np.count_nonzero(sampled.valid.any(axis=1))),
        outside_overlap=int(np.count_nonzero(~sampled.inside)),
        median_interval_ms=float(interval_s * 1000.0) if np.isfinite(interval_s) else float("nan"),
        median_landmark_speed_px_s=median_speed,
        max_landmark_speed_px_s=max_speed,
        sync_uncertainty_ms=(sync_s * 1000.0) if sync_s is not None else None,
        median_pairing_error_px=(sync_s * median_speed) if sync_s is not None else None,
        max_pairing_error_px=(sync_s * max_speed) if sync_s is not None else None,
        # A nearest-frame pairing rounds to within half an interval, and the
        # error is uniform over it, so a quarter of an interval is its mean
        # magnitude. Quoted at the **fastest** landmark rather than the median,
        # because the median is dominated by the parts of a body that barely
        # move and every metric worth computing is anchored to the parts that do.
        # This is the number that decided `ReconstructionConfig.resample`.
        nearest_frame_error_px=(0.25 * interval_s * max_speed if np.isfinite(interval_s) else None),
    )


def _warnings(report: ReconstructionReport, sigma_measured: bool) -> list[str]:
    """Facts a reader should not have to derive from the tables."""
    notes: list[str] = [
        "These points are metres in the reference camera's frame, not a scene-fixed "
        "frame: nothing here measures which way is up or which way the target line "
        "runs. Lengths, angles and speeds between reconstructed points are unaffected, "
        "because they do not depend on the frame; anything stated as 'vertical' or "
        "'towards the target' would."
    ]

    quality = report.quality
    if quality is None:
        return notes

    if not sigma_measured:
        notes.append(
            "The filter's residual did not supply a landmark scatter for this clip -- which "
            f"is what a window holding exactly as many samples as the polynomial has "
            f"coefficients produces -- so the uncertainties below use the "
            f"{_MEASURED_LANDMARK_SIGMA_FRAME_WIDTHS:g} frame widths Phase 3 measured from "
            "real footage instead."
        )

    unstable = [entry for entry in quality.bones if not entry.stable]
    if unstable:
        worst_bone = max(unstable, key=lambda entry: entry.variation)
        notes.append(
            f"{len(unstable)} of {len(quality.bones)} reconstructed segments change length "
            f"through the clip by more than the bound, worst {worst_bone.name} at "
            f"{worst_bone.variation:.0%} ({worst_bone.spread_m * 100:.1f} cm). A bone does not "
            "change length, so this is reconstruction error -- and it is the error the "
            "reprojection residual cannot see, because a landmark sliding along its own ray "
            "reprojects perfectly at the wrong depth."
        )

    disagreeing = [entry for entry in quality.symmetry if not entry.agrees]
    if disagreeing:
        worst_pair = max(disagreeing, key=lambda entry: entry.disagreement)
        notes.append(
            f"The left and right {worst_pair.segment} reconstruct "
            f"{worst_pair.disagreement:.0%} apart ({worst_pair.left_m * 100:.1f} cm against "
            f"{worst_pair.right_m * 100:.1f} cm). Real people are not that asymmetric, so one "
            "side is being reconstructed systematically worse than the other -- typically the "
            "side one camera has occluded throughout."
        )

    if quality.median_convergence_deg is not None and quality.median_convergence_deg < 30.0:
        notes.append(
            f"The two rays meet at {quality.median_convergence_deg:.0f} degrees at the median, "
            "which is a shallow intersection: depth error scales as 1/sin of it, so this "
            "capture costs "
            f"{1.0 / max(np.sin(np.radians(quality.median_convergence_deg)), 1e-6):.1f}x the "
            "error a right-angled pair would give. Moving one camera further round the "
            "player is worth more here than any amount of re-calibration."
        )

    pairing = report.pairing
    if (
        pairing is not None
        and pairing.max_pairing_error_px is not None
        and quality.median_reprojection_px is not None
        and pairing.max_pairing_error_px > 0.5 * quality.median_reprojection_px
    ):
        notes.append(
            f"Not knowing exactly when each frame was taken displaces the fastest landmark by "
            f"{pairing.max_pairing_error_px:.1f} px, against a "
            f"{quality.median_reprojection_px:.1f} px reprojection error. The alignment, not "
            "the calibration, is what limits this reconstruction -- and the hands are where "
            "that lands, which is where the metrics are."
        )

    if report.frames and report.reconstructed_frames < 0.5 * report.frames:
        notes.append(
            f"Only {report.reconstructed_frames} of {report.frames} reference frames produced "
            "any reconstructed landmark at all, so most of this swing has no 3D positions."
        )

    return notes


def reconstruct_pair(
    reference: FilteredSequence,
    target: FilteredSequence,
    rig: CameraRig | None,
    time_map: TimeMap | None,
    *,
    reference_role: CameraRole,
    target_role: CameraRole,
    reference_name: str = "reference",
    target_name: str = "target",
    landmarks: tuple[Landmark, ...] | None = None,
    config: ReconstructionConfig | None = None,
) -> ReconstructedSequence:
    """Triangulate every landmark of one swing from two calibrated, aligned clips.

    `reference` sets the clock and the frame numbering; `target` is resampled
    onto it. Which clip is which changes the coordinate frame the points come out
    in and nothing else, so the natural choice is whichever clip the swing events
    were located in.
    """
    resolved = config or ReconstructionConfig()
    geometry = _require_rig(rig, reference_role, target_role, reference, target)

    if time_map is None:
        raise ReconstructionError(
            "The two clips' clocks have not been related, so nothing here knows which "
            "target frame shows the same instant as a given reference frame.",
            remediation=(
                "Align the pair with `analyzer project sync`. Triangulating two views of "
                "*different* instants produces a confident answer about a body that was "
                "never in that position."
            ),
        )

    selected = landmarks if landmarks is not None else tuple(Landmark)
    reference_track = pairing_module.track_from(reference, selected)
    target_track = pairing_module.track_from(target, selected)

    times = reference_track.t
    mapped = np.array([time_map.to_target(float(instant)) for instant in times])
    sampler: Callable[
        [pairing_module.ViewTrack, NDArray[np.float64]], pairing_module.SampledTrack
    ] = pairing_module.resample if resolved.resample else pairing_module.nearest
    sampled = sampler(target_track, mapped)

    frames, width = len(reference_track), len(selected)
    shape = (frames, width)

    # --- gather the pairs, flattened, so one triangulation call does the clip --
    reference_px = reference_track.pixels.reshape(-1, 2)
    target_px = sampled.pixels.reshape(-1, 2)
    reference_velocity = reference_track.velocity.reshape(-1, 2)
    target_velocity = sampled.velocity.reshape(-1, 2)

    visibility = np.minimum(reference_track.visibility, sampled.visibility)
    seen = (
        reference_track.valid
        & sampled.valid
        & (reference_track.visibility >= resolved.min_visibility)
        & (sampled.visibility >= resolved.min_visibility)
    )
    outside = ~np.broadcast_to(sampled.inside[:, None], shape) & ~seen

    attempt = seen.reshape(-1)
    masked_reference = np.where(attempt[:, None], reference_px, np.nan)
    masked_target = np.where(attempt[:, None], target_px, np.nan)

    points = triangulate(
        masked_reference,
        masked_target,
        geometry,
        refine_points=resolved.refine,
        iterations=resolved.max_refine_iterations,
    )
    residual = reprojection_errors(points, masked_reference, masked_target, geometry)
    convergence = convergence_angles(points, geometry)

    sigma_reference, measured_reference = measured_pixel_sigma(reference, selected)
    sigma_target, measured_target = measured_pixel_sigma(target, selected)
    # The worse of the two views. A point is located by the disagreement between
    # them, so the noisier view bounds how well that disagreement is known.
    sigma_px = max(sigma_reference, sigma_target)
    sigma_measured = measured_reference and measured_target
    if resolved.pixel_sigma_px is not None:
        sigma_px = resolved.pixel_sigma_px
        sigma_measured = False

    uncertainty = positional_uncertainty(points, geometry, sigma_px)
    velocity = velocity_from_image(points, reference_velocity, target_velocity, geometry)

    # --- the gates, in the order the module docstring argues for ---------------
    located = np.isfinite(points).all(axis=1)
    ill = located & ~(convergence >= resolved.min_convergence_deg)
    bad_residual = located & ~ill & ~(residual <= resolved.max_reprojection_px)
    uncertain = (
        located
        & ~ill
        & ~bad_residual
        & (
            ~(uncertainty <= resolved.max_uncertainty_m)
            if resolved.max_uncertainty_m is not None
            else np.zeros(points.shape[0], dtype=np.bool_)
        )
    )
    accepted = located & ~ill & ~bad_residual & ~uncertain

    # Plain strings rather than the enum members: a `StrEnum` is iterable, and
    # numpy assigns an iterable into an object array element-wise, which would
    # scatter the characters of "not_seen" across eight cells.
    reasons = np.full(points.shape[0], "", dtype="<U20")
    reasons[outside.reshape(-1)] = RefusalReason.OUTSIDE_OVERLAP.value
    reasons[~attempt & ~outside.reshape(-1)] = RefusalReason.NOT_SEEN.value
    reasons[attempt & ~located] = RefusalReason.NOT_SEEN.value
    reasons[ill] = RefusalReason.ILL_CONDITIONED.value
    reasons[bad_residual] = RefusalReason.REPROJECTION.value
    reasons[uncertain] = RefusalReason.UNCERTAIN.value
    reasons[accepted] = ""

    def _shape(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(accepted, values, np.nan).reshape(shape)

    valid = accepted.reshape(shape)
    shaped_reasons = reasons.reshape(shape)
    kept_points = np.where(accepted[:, None], points, np.nan).reshape(frames, width, 3)
    kept_velocity = np.where(accepted[:, None], velocity, np.nan).reshape(frames, width, 3)

    report = _build_report(
        reference=reference,
        selected=selected,
        valid=valid,
        points=kept_points,
        uncertainty=_shape(uncertainty),
        convergence=_shape(convergence),
        residual=_shape(residual),
        reasons=shaped_reasons,
        geometry=geometry,
        rig=rig,
        pairing=_pairing_summary(
            target_track, sampled, time_map, times, resampled=resolved.resample
        ),
        sigma_px=sigma_px,
        sigma_measured=sigma_measured,
        reference_role=reference_role,
        target_role=target_role,
        reference_name=reference_name,
        target_name=target_name,
        config=resolved,
    )

    return ReconstructedSequence(
        t=times,
        points=kept_points,
        velocity=kept_velocity,
        valid=valid,
        uncertainty_m=_shape(uncertainty),
        convergence_deg=_shape(convergence),
        reprojection_px=_shape(residual),
        refusal=shaped_reasons,
        visibility=np.where(valid, visibility, 0.0),
        landmarks=selected,
        geometry=geometry,
        space=LandmarkSpace.CAMERA,
        slow_motion_factor=reference.slow_motion_factor,
        report=report,
    )


def _build_report(
    *,
    reference: FilteredSequence,
    selected: tuple[Landmark, ...],
    valid: NDArray[np.bool_],
    points: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    convergence: NDArray[np.float64],
    residual: NDArray[np.float64],
    reasons: NDArray[np.str_],
    geometry: StereoGeometry,
    rig: CameraRig | None,
    pairing: PairingSummary,
    sigma_px: float,
    sigma_measured: bool,
    reference_role: CameraRole,
    target_role: CameraRole,
    reference_name: str,
    target_name: str,
    config: ReconstructionConfig,
) -> ReconstructionReport:
    """Assemble the contract a caller reads, from the arrays a caller does not."""
    frames = int(valid.shape[0])
    attempted = int(valid.size)
    produced = int(np.count_nonzero(valid))

    per_landmark: list[LandmarkReconstruction] = []
    for column, landmark in enumerate(selected):
        column_valid = valid[:, column]
        counts: dict[RefusalReason, int] = {}
        for reason in RefusalReason:
            hits = int(np.count_nonzero(reasons[:, column] == reason.value))
            if hits:
                counts[reason] = hits
        per_landmark.append(
            LandmarkReconstruction(
                landmark=int(landmark),
                name=landmark.name.lower(),
                frames=frames,
                reconstructed=int(np.count_nonzero(column_valid)),
                coverage=float(np.count_nonzero(column_valid) / frames) if frames else 0.0,
                median_convergence_deg=_median(convergence[:, column]),
                median_reprojection_px=_median(residual[:, column]),
                median_uncertainty_m=_median(uncertainty[:, column]),
                refused=counts,
            )
        )

    bones = skeleton.bone_consistency(points, valid, selected, config)
    quality = (
        ReconstructionQuality(
            points_attempted=attempted,
            points_reconstructed=produced,
            coverage=produced / attempted if attempted else 0.0,
            median_reprojection_px=_median(residual),
            max_reprojection_px=_extreme(residual, np.nanmax),
            median_convergence_deg=_median(convergence),
            min_convergence_deg=_extreme(convergence, np.nanmin),
            median_uncertainty_m=_median(uncertainty),
            p95_uncertainty_m=_percentile(uncertainty, 95.0),
            bones=bones,
            symmetry=skeleton.symmetry(bones, config),
            worst_bone_variation=(max(entry.variation for entry in bones) if bones else None),
            pixel_sigma_px=sigma_px,
            methodology=(
                "Landmarks undistorted below the filter, converted to pixels, and "
                "triangulated by DLT"
                + (" then refined on reprojection error in both views" if config.refine else "")
                + ". Convergence is the angle between the two rays at the fitted point; "
                "uncertainty is sigma * (J'J)^-1 in the worst direction, with sigma "
                + (
                    "measured from this clip's filter residual"
                    if sigma_measured
                    else "taken from the scatter Phase 3 measured on real footage"
                )
                + f" at {sigma_px:.2f} px. Bone variation is the interquartile range over "
                "the median of each segment's reconstructed length, which the reprojection "
                "error cannot measure."
            ),
        )
        if produced
        else None
    )

    report = ReconstructionReport(
        reconstructed=produced > 0,
        reference_role=reference_role,
        target_role=target_role,
        reference_name=reference_name,
        target_name=target_name,
        frames=frames,
        reconstructed_frames=int(np.count_nonzero(valid.any(axis=1))),
        slow_motion_factor=reference.slow_motion_factor,
        calibration=CalibrationStatus.STEREO,
        baseline_m=geometry.baseline_m,
        convergence_deg=(rig.stereo.convergence_deg if rig is not None and rig.stereo else 0.0),
        quality=quality,
        pairing=pairing,
        landmarks=per_landmark,
        reconstructed_at=datetime.now(UTC),
        config=config,
        refusal=(
            None
            if produced
            else (
                "No landmark could be triangulated at any instant. Either the two clips do "
                "not overlap once their clocks are related, or no landmark was tracked in "
                "both views at the same moment."
            )
        ),
    )
    report.warnings = _warnings(report, sigma_measured)
    return report


def _median(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _percentile(values: NDArray[np.float64], percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def _extreme(
    values: NDArray[np.float64], reducer: Callable[[NDArray[np.float64]], np.float64]
) -> float | None:
    finite = values[np.isfinite(values)]
    return float(reducer(finite)) if finite.size else None
