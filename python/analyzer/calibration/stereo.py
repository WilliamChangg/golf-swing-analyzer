"""Where the second camera stands, and what not having a genlock costs.

Stereo extrinsics need two views of the board **at the same instant**, and two
phones started by hand do not share a clock. Phase 7 relates those clocks and
says how well it knows the relation; this module is where that uncertainty stops
being a millisecond figure in a report and becomes a quantity in the same unit
as the thing it damages.

## Sync error, in pixels

A pairing that is wrong by `dt` seconds matters exactly as much as the board
moved in `dt` seconds. So the cost of pairing frame A with frame B is

    pairing_error_px  =  effective_time_error_s  x  board_image_speed_px_s

and that is a length, directly comparable with the reprojection error it would
otherwise be mistaken for -- a pair contributing two pixels of pairing error
into a fit whose residual is a fifth of a pixel is contributing noise ten times
larger than the signal, and no reprojection error will say so, because the fit
will absorb it into the pose.

The effective time error has two parts and they are different kinds of thing:
the **known** gap between the two frames once mapped onto one clock, and the
**unknown** error in the map itself, which Phase 7 reports as
`TimeMap.uncertainty_at`. They combine in quadrature, which is the usual
treatment for one measured offset and one standard error.

## The capture instruction that follows

`board_image_speed_px_s` is the term a person controls, and it is the one worth
controlling: hold the board still for a second at each position and the speed
goes to nearly zero, taking the entire cost of not having a genlock with it.
Wave it around and no achievable sync quality rescues the pair.

This is why stereo calibration from ordinary unsynchronised video is possible at
all, and the system measures it rather than assuming it -- a held board and a
waved one produce the same detections, the same corner counts and the same
confident-looking residual, and differ only here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.calibration.detect import BoardView
from analyzer.calibration.intrinsics import coverage
from analyzer.contracts.calibration import (
    BoardSpec,
    CalibrationConfig,
    CalibrationQuality,
    CameraIntrinsics,
    PairingReport,
    StereoCalibration,
)
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.sync import TimeMap

# How far the board must have moved, in either camera, for a pair to count as
# new evidence rather than another frame of the same position.
_DISTINCT_PAIR_SEPARATION_PX = 40.0

# A pair must share at least this many board corners. Six matches the per-view
# floor for intrinsics: below it a pair constrains the relative pose so weakly
# that it mostly contributes its own detection noise.
_MIN_SHARED_CORNERS = 6


class StereoError(ValueError):
    """Stereo extrinsics could not be computed."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class ViewPair:
    """Two views of the board believed to show the same instant."""

    reference: BoardView
    target: BoardView
    time_error_s: float
    """Known gap between the two, on one clock, after the map is applied."""
    map_uncertainty_s: float
    """Phase 7's own uncertainty in that map, at this instant. Zero when unknown."""
    board_speed_px_s: float
    """How fast the board was moving in the image around this pair."""

    @property
    def effective_time_error_s(self) -> float:
        return float(np.hypot(self.time_error_s, self.map_uncertainty_s))

    @property
    def pairing_error_px(self) -> float:
        """Board displacement attributable to the pairing being imperfect."""
        return self.effective_time_error_s * self.board_speed_px_s

    def shared_corners(self) -> NDArray[np.int32]:
        """Corner ids both views saw, sorted, so the correspondence is unambiguous."""
        return np.intersect1d(
            self.reference.corner_ids.ravel(), self.target.corner_ids.ravel()
        ).astype(np.int32)


def board_speed(views: list[BoardView], index: int) -> float:
    """How fast the board is moving in the image, around one view, in px/s.

    A central difference of the board's image centroid over its neighbouring
    views, falling back to a one-sided difference at the ends. The centroid is a
    crude summary of the board's motion -- it is blind to rotation about its own
    centre, which also displaces corners -- so this under-reports a board being
    twisted in place. It is still the right quantity for the dominant case,
    which is a board being carried from one position to the next.

    Zero when there is nothing to difference against, which is the honest answer
    for a single view rather than a claim that it was stationary.

    **`views` must be every detection, not a selected subset**, and getting that
    wrong defeats the whole mechanism. Selection keeps one view per board
    position, so differencing selected views measures how fast the board was
    carried *between* positions -- which is large however patiently it was held
    at either end. The quantity wanted is the speed at the paired instant, and
    the only frames that carry it are the neighbouring ones in time, which are
    exactly the frames selection discards as repetitive.
    """
    if len(views) < 2:
        return 0.0

    low = max(0, index - 1)
    high = min(len(views) - 1, index + 1)
    if low == high:
        return 0.0

    span_s = views[high].timestamp_s - views[low].timestamp_s
    if span_s <= 0.0:
        return 0.0

    start = np.array(views[low].centroid)
    end = np.array(views[high].centroid)
    return float(np.linalg.norm(end - start) / span_s)


def pair_by_time_map(
    reference_views: list[BoardView],
    target_views: list[BoardView],
    time_map: TimeMap,
    config: CalibrationConfig,
    *,
    select: bool = True,
) -> tuple[list[ViewPair], list[str]]:
    """Match each reference view to the target view closest to it in time.

    Greedy nearest-neighbour, with each target view used at most once. A target
    view is not the best match for two reference views unless the reference
    camera is much the faster of the two, and in that case pairing both would
    put the same evidence into the fit twice under two different board poses.

    **Both lists must be every detection, unselected.** Two things here need the
    frames that selection throws away: the nearest target view in time is the
    nearest of all of them, not of a thinned subset, and `board_speed` is
    measured from the neighbouring frames in time, which is what makes holding
    the board still worth anything. Distinctness is then selected on the
    *pairs*, which is the right place for it -- a pair is the unit the stereo
    fit consumes, and two pairs showing the board in one position are one piece
    of evidence however many frames they came from.
    """
    pairs: list[ViewPair] = []
    dropped: list[str] = []
    taken: set[int] = set()

    target_times = np.array([view.timestamp_s for view in target_views])

    for index, view in enumerate(reference_views):
        if target_times.size == 0:
            break
        expected = time_map.to_target(view.timestamp_s)
        order = np.argsort(np.abs(target_times - expected))
        choice = next((int(slot) for slot in order if int(slot) not in taken), None)
        if choice is None:
            break

        error_s = float(abs(target_times[choice] - expected))
        if error_s * 1000.0 > config.max_pair_time_error_ms:
            dropped.append(
                f"Reference frame {view.frame}: the nearest target view sits "
                f"{error_s * 1000:.0f} ms away on the shared clock, past the "
                f"{config.max_pair_time_error_ms:.0f} ms bound."
            )
            continue

        uncertainty = time_map.uncertainty_at(view.timestamp_s) or 0.0
        speed = max(
            board_speed(reference_views, index),
            board_speed(target_views, choice),
        )
        pair = ViewPair(
            reference=view,
            target=target_views[choice],
            time_error_s=error_s,
            map_uncertainty_s=float(uncertainty),
            board_speed_px_s=speed,
        )

        if pair.pairing_error_px > config.max_pairing_error_px:
            dropped.append(
                f"Reference frame {view.frame}: the board was moving at "
                f"{speed:.0f} px/s and the pairing is known to "
                f"{pair.effective_time_error_s * 1000:.0f} ms, so this pair carries "
                f"{pair.pairing_error_px:.1f} px of displacement error -- past the "
                f"{config.max_pairing_error_px:.1f} px bound. Hold the board still."
            )
            continue

        if pair.shared_corners().size < _MIN_SHARED_CORNERS:
            dropped.append(
                f"Reference frame {view.frame}: the two cameras share only "
                f"{pair.shared_corners().size} board corner(s). Both must see the same "
                "part of the board."
            )
            continue

        taken.add(choice)
        pairs.append(pair)

    if select:
        pairs, repeated = _select_distinct_pairs(pairs)
        dropped.extend(repeated)

    return pairs, dropped


def _select_distinct_pairs(pairs: list[ViewPair]) -> tuple[list[ViewPair], list[str]]:
    """Keep pairs that show the board somewhere new, in either camera.

    The same reasoning as `detect.select_distinct`, applied one level up: a
    board held still for a second at 30 fps produces thirty pairs of one
    position, and thirty copies of one piece of evidence weight the fit thirty
    times over without constraining it any further.

    A pair counts as new if *either* camera saw the board somewhere new, because
    the two views are not redundant -- a board that moved along one camera's
    optical axis barely moves in that camera's image and moves a long way in the
    other's.
    """
    kept: list[ViewPair] = []
    dropped: list[str] = []

    for pair in pairs:
        distinct = True
        for existing in kept:
            reference_moved = float(
                np.linalg.norm(
                    np.array(pair.reference.centroid) - np.array(existing.reference.centroid)
                )
            )
            target_moved = float(
                np.linalg.norm(np.array(pair.target.centroid) - np.array(existing.target.centroid))
            )
            if max(reference_moved, target_moved) < _DISTINCT_PAIR_SEPARATION_PX:
                distinct = False
                break
        if distinct:
            kept.append(pair)
        else:
            dropped.append(
                f"Reference frame {pair.reference.frame}: the board was in a position an "
                "earlier pair already covered."
            )

    return kept, dropped


def pair_explicitly(
    reference_views: list[BoardView],
    target_views: list[BoardView],
    frame_pairs: list[tuple[int, int]],
    config: CalibrationConfig,
) -> tuple[list[ViewPair], list[str]]:
    """Match views the caller has named, asserting simultaneity rather than measuring it.

    The escape hatch for a rig that genuinely is synchronised -- a clapperboard,
    a flash, two cameras on one trigger -- where the pairing is known by means
    outside this system. Nothing here can check the assertion, so
    `PairingReport.median_time_error_ms` is None for this path rather than zero:
    zero would read as a measurement of perfect synchronisation.
    """
    by_frame_reference = {view.frame: (index, view) for index, view in enumerate(reference_views)}
    by_frame_target = {view.frame: (index, view) for index, view in enumerate(target_views)}

    pairs: list[ViewPair] = []
    dropped: list[str] = []

    for reference_frame, target_frame in frame_pairs:
        found_reference = by_frame_reference.get(reference_frame)
        found_target = by_frame_target.get(target_frame)
        if found_reference is None or found_target is None:
            missing = "reference" if found_reference is None else "target"
            frame = reference_frame if found_reference is None else target_frame
            dropped.append(f"No board was detected in {missing} frame {frame}.")
            continue

        pair = ViewPair(
            reference=found_reference[1],
            target=found_target[1],
            time_error_s=0.0,
            map_uncertainty_s=0.0,
            board_speed_px_s=max(
                board_speed(reference_views, found_reference[0]),
                board_speed(target_views, found_target[0]),
            ),
        )
        if pair.shared_corners().size < _MIN_SHARED_CORNERS:
            dropped.append(
                f"Frames {reference_frame}/{target_frame} share only "
                f"{pair.shared_corners().size} board corner(s)."
            )
            continue
        pairs.append(pair)

    del config
    return pairs, dropped


def _correspondences(
    pairs: list[ViewPair], spec: BoardSpec
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Per pair: the shared board corners, and where each camera saw them."""
    from analyzer.calibration.board import object_points

    board_corners = object_points(spec)
    objects: list[NDArray[np.float32]] = []
    reference_points: list[NDArray[np.float32]] = []
    target_points: list[NDArray[np.float32]] = []

    for pair in pairs:
        shared = pair.shared_corners()
        reference_index = {
            int(identifier): slot
            for slot, identifier in enumerate(pair.reference.corner_ids.ravel())
        }
        target_index = {
            int(identifier): slot for slot, identifier in enumerate(pair.target.corner_ids.ravel())
        }

        objects.append(board_corners[shared].astype(np.float32).reshape(-1, 1, 3))
        reference_points.append(
            pair.reference.corners_px.reshape(-1, 2)[
                [reference_index[int(identifier)] for identifier in shared]
            ]
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )
        target_points.append(
            pair.target.corners_px.reshape(-1, 2)[
                [target_index[int(identifier)] for identifier in shared]
            ]
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )

    return objects, reference_points, target_points


def calibrate_stereo(
    pairs: list[ViewPair],
    spec: BoardSpec,
    reference_intrinsics: CameraIntrinsics,
    target_intrinsics: CameraIntrinsics,
    *,
    reference_role: CameraRole,
    target_role: CameraRole,
    dropped: list[str] | None = None,
    candidates: int | None = None,
    method: str = "time_map",
    config: CalibrationConfig | None = None,
) -> StereoCalibration:
    """Fit the relative pose of two calibrated cameras from paired board views.

    The intrinsics are **held fixed**. Re-fitting them here would let the
    optimiser trade a focal length against a baseline -- the two are nearly
    interchangeable over a small angular range -- and quietly replace two
    calibrations that were each measured from a proper spread of board views
    with a joint fit determined by whatever the two cameras happened to see at
    once. The intrinsics are the better-determined quantity and are kept.
    """
    resolved = config or CalibrationConfig()

    if len(pairs) < resolved.min_stereo_pairs:
        raise StereoError(
            f"{len(pairs)} usable board pair(s); {resolved.min_stereo_pairs} are required.",
            remediation=(
                "Record both cameras while the board is held still at several positions "
                "across the shared field of view. Holding it still is what makes the pairing "
                "cost nothing; moving between positions is what makes the pairs different."
            ),
        )

    objects, reference_points, target_points = _correspondences(pairs, spec)
    image_size = (reference_intrinsics.image_width, reference_intrinsics.image_height)

    result = cv2.stereoCalibrate(
        objects,
        reference_points,
        target_points,
        reference_intrinsics.matrix(),
        reference_intrinsics.distortion_vector(),
        target_intrinsics.matrix(),
        target_intrinsics.distortion_vector(),
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms = float(result[0])
    rotation = np.asarray(result[5], dtype=np.float64)
    translation = np.asarray(result[6], dtype=np.float64).reshape(3)

    per_pair, worst = _stereo_residuals(
        pairs,
        objects,
        reference_points,
        target_points,
        reference_intrinsics,
        target_intrinsics,
        rotation,
        translation,
    )

    speeds = [pair.board_speed_px_s for pair in pairs]
    pairing_errors = [pair.pairing_error_px for pair in pairs]
    time_errors = [pair.time_error_s * 1000.0 for pair in pairs]

    pairing = PairingReport(
        pairs=len(pairs),
        candidates=candidates if candidates is not None else len(pairs),
        method=method,
        median_time_error_ms=(float(np.median(time_errors)) if method == "time_map" else None),
        worst_pairing_error_px=(float(max(pairing_errors)) if method == "time_map" else None),
        median_board_speed_px_s=float(np.median(speeds)) if speeds else None,
        dropped=list(dropped or []),
    )

    tilts = [_pair_tilt_deg(pair, spec, reference_intrinsics) for pair in pairs]
    corners = sum(int(pair.shared_corners().size) for pair in pairs)
    # Each shared corner gives two coordinates in each of two cameras; the pose
    # of each board and the one relative pose are what is fitted.
    parameters = 6 + 6 * len(pairs)

    quality = CalibrationQuality(
        rms_reprojection_px=rms,
        max_reprojection_px=worst,
        per_view_rms_px=per_pair,
        coverage=coverage([pair.reference for pair in pairs], image_size, tilts),
        degrees_of_freedom=4 * corners - parameters,
    )

    baseline = float(np.linalg.norm(translation))
    convergence = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))

    usable, refusal, warnings = _stereo_gate(quality, pairing, baseline, resolved)

    return StereoCalibration(
        reference_role=reference_role,
        target_role=target_role,
        rotation=[[float(value) for value in row] for row in rotation],
        translation_m=[float(value) for value in translation],
        baseline_m=max(baseline, 1e-9),
        convergence_deg=convergence,
        quality=quality,
        pairing=pairing,
        calibrated_at=datetime.now(UTC),
        usable=usable,
        refusal=refusal,
        warnings=warnings,
    )


def _pair_tilt_deg(pair: ViewPair, spec: BoardSpec, intrinsics: CameraIntrinsics) -> float:
    """Board tilt in the reference camera, recovered per pair by PnP."""
    from analyzer.calibration.board import object_points

    shared = pair.shared_corners()
    index = {
        int(identifier): slot for slot, identifier in enumerate(pair.reference.corner_ids.ravel())
    }
    image = pair.reference.corners_px.reshape(-1, 2)[
        [index[int(identifier)] for identifier in shared]
    ]
    ok, rvec, _ = cv2.solvePnP(
        object_points(spec)[shared].astype(np.float64),
        image.astype(np.float64),
        intrinsics.matrix(),
        intrinsics.distortion_vector(),
    )
    if not ok:
        return 0.0
    normal = cv2.Rodrigues(rvec)[0] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))


def _stereo_residuals(
    pairs: list[ViewPair],
    objects: list[NDArray[np.float32]],
    reference_points: list[NDArray[np.float32]],
    target_points: list[NDArray[np.float32]],
    reference_intrinsics: CameraIntrinsics,
    target_intrinsics: CameraIntrinsics,
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> tuple[list[float], float]:
    """Per-pair RMS and the worst single corner, in the target camera.

    Each board's pose is recovered in the reference camera by PnP, carried into
    the target camera through the fitted relative pose, and reprojected there.
    That measures exactly what the extrinsics claim -- a point located by one
    camera lands where the other camera sees it -- rather than re-measuring the
    intrinsics, which this fit did not touch.
    """
    per_pair: list[float] = []
    worst = 0.0

    for _pair, board, reference_image, target_image in zip(
        pairs, objects, reference_points, target_points, strict=True
    ):
        ok, rvec, tvec = cv2.solvePnP(
            board.reshape(-1, 3).astype(np.float64),
            reference_image.reshape(-1, 2).astype(np.float64),
            reference_intrinsics.matrix(),
            reference_intrinsics.distortion_vector(),
        )
        if not ok:  # pragma: no cover - PnP failing on a view that already calibrated
            per_pair.append(float("nan"))
            continue

        board_rotation = cv2.Rodrigues(rvec)[0]
        in_target_rotation = rotation @ board_rotation
        in_target_translation = rotation @ tvec.reshape(3) + translation

        projected, _ = cv2.projectPoints(
            board.reshape(-1, 3).astype(np.float64),
            cv2.Rodrigues(in_target_rotation)[0],
            in_target_translation,
            target_intrinsics.matrix(),
            target_intrinsics.distortion_vector(),
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - target_image.reshape(-1, 2), axis=1)
        per_pair.append(float(np.sqrt(np.mean(errors**2))))
        worst = max(worst, float(np.max(errors)))

    return per_pair, worst


def _stereo_gate(
    quality: CalibrationQuality,
    pairing: PairingReport,
    baseline_m: float,
    config: CalibrationConfig,
) -> tuple[bool, str | None, list[str]]:
    """Whether the extrinsics may be used, and why not."""
    warnings: list[str] = []

    if quality.rms_reprojection_px > config.max_rms_reprojection_px:
        return (
            False,
            (
                f"A point located by one camera lands {quality.rms_reprojection_px:.2f} px "
                f"from where the other camera sees it, past the "
                f"{config.max_rms_reprojection_px:.2f} px bound. Either the two cameras were "
                "not looking at the board at the same moment, or one of them moved between "
                "the calibration and now."
            ),
            warnings,
        )

    if quality.degrees_of_freedom <= 0:
        return (
            False,
            (
                "The fit has no spare degrees of freedom, so its residual is zero by "
                "construction and nothing checked the result. More paired views are needed."
            ),
            warnings,
        )

    if baseline_m < 0.05:
        warnings.append(
            f"The two optical centres are {baseline_m * 100:.0f} cm apart. Triangulation "
            "over a baseline that short is ill-conditioned: a small error in either ray "
            "becomes a large error in depth."
        )

    worst = pairing.worst_pairing_error_px
    if worst is not None and worst > 0.25 * config.max_pairing_error_px:
        warnings.append(
            f"The worst-paired view carries {worst:.2f} px of displacement from the clocks "
            "not being exactly related, which is a substantial fraction of the "
            f"{quality.rms_reprojection_px:.2f} px residual. Holding the board still at each "
            "position removes this term entirely."
        )

    return True, None, warnings
