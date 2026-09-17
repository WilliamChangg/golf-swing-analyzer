"""Finding the board in footage, and choosing which views to keep.

Two jobs that look like one. Detection is mechanical: hand each frame to
OpenCV's Charuco detector and record what came back. Selection is not, and it is
where a calibration is usually won or lost.

## Why selection is not "use everything"

Calibration footage is a person holding a board in front of a camera for thirty
seconds, which at 30 fps is nine hundred frames of which perhaps twenty are
distinct. The other eight hundred and eighty are the same view again, and they
are not free:

**They cost accuracy.** Every view is one sample of the board's position, and
two hundred samples of *one* position is still one position. The fit weights
them equally, so a set dominated by whatever pose the person rested on is a fit
dominated by it -- the optimiser will happily sacrifice the eight distinct views
to improve the two hundred identical ones.

**They cost honesty.** `CoverageReport.views` is read as evidence that the lens
was sampled. Nine hundred near-identical views reported as nine hundred would
make the most degenerate possible capture look like the most thorough one.

So views are selected for **being different from each other**, greedily, by how
far the board's image position and apparent size have moved. That is a
deliberately crude measure of distinctness -- it does not know about tilt, which
`CoverageReport` measures separately and which matters more -- but it is the
measure that survives having no calibration yet, which is the situation
selection runs in. Tilt cannot be computed before the intrinsics exist, and
selecting on a quantity estimated from the intrinsics being fitted is circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.calibration.board import detector, object_points
from analyzer.contracts.calibration import BoardObservation, BoardSpec, DetectionReport
from analyzer.ingestion.reader import FrameSource, open_video
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker

# Image extensions a still-image calibration set may use. Deliberately short:
# these are what a phone and a camera produce, and an unrecognised extension is
# better reported as "no images found" than silently handed to a decoder.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class BoardView:
    """One frame in which the board was found, with the correspondences it gives.

    `corner_ids` index into `board.object_points(spec)`, so the pairing between
    what was seen and where it sits on the board is carried rather than
    reconstructed. That is the whole reason Charuco is worth its complexity over
    a chessboard: the correspondence is never in doubt, so a partial view is
    usable instead of wasted.
    """

    frame: int
    timestamp_s: float
    corner_ids: NDArray[np.int32]
    corners_px: NDArray[np.float32]
    image_size: tuple[int, int]
    """Displayed (width, height). Carried per view so a set mixing two frame
    sizes is caught here rather than producing a calibration for neither."""

    def __len__(self) -> int:
        return int(self.corner_ids.size)

    @property
    def centroid(self) -> tuple[float, float]:
        mean = self.corners_px.reshape(-1, 2).mean(axis=0)
        return float(mean[0]), float(mean[1])

    @property
    def extent_px(self) -> float:
        """Diagonal of the board's bounding box in the image.

        A proxy for apparent size, and so for distance: it halves when the board
        moves twice as far away. Used for selection, where only the ratio
        matters, never as a distance -- the metric one comes from the board pose
        once intrinsics exist.
        """
        points = self.corners_px.reshape(-1, 2)
        span = points.max(axis=0) - points.min(axis=0)
        return float(np.hypot(span[0], span[1]))

    def object_corners(self, spec: BoardSpec) -> NDArray[np.float64]:
        """The board-frame coordinates of the corners this view saw, in metres."""
        return object_points(spec)[self.corner_ids.ravel()]


def detect_in_image(
    image: NDArray[np.uint8],
    spec: BoardSpec,
    *,
    frame: int = 0,
    timestamp_s: float = 0.0,
    charuco_detector: cv2.aruco.CharucoDetector | None = None,
) -> BoardView | None:
    """Find the board in one image, or None if it is not there.

    Grayscale is what the detector wants and colour is what every source here
    produces, so the conversion happens once at the boundary. It is also the
    only place colour order could go wrong, and unlike pose estimation -- where
    BGR fed as RGB produces plausible, wrong landmarks -- a board detector
    handed the wrong channel order finds the board anyway, because a
    black-and-white pattern is the same in both. The conversion is still
    explicit, so nobody has to work that out.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    found = charuco_detector or detector(spec)
    corners, ids, _, _ = found.detectBoard(gray)

    if corners is None or ids is None or len(ids) == 0:
        return None

    height, width = gray.shape[:2]
    return BoardView(
        frame=frame,
        timestamp_s=timestamp_s,
        corner_ids=np.asarray(ids, dtype=np.int32).reshape(-1),
        corners_px=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        image_size=(int(width), int(height)),
    )


def select_distinct(
    views: list[BoardView],
    *,
    min_separation_px: float = 40.0,
    min_scale_ratio: float = 1.15,
    limit: int = 40,
) -> list[BoardView]:
    """Keep views that differ from the ones already kept.

    Greedy and order-preserving: walk the views in time and keep one whenever it
    is far enough from **every** view kept so far, either by where the board sits
    in the frame or by how large it appears. Distance alone would reject a view
    taken at the same place but much closer, which is one of the more valuable
    views there is -- it is what gives the set a range of scales.

    `limit` caps the result because the fit's cost grows with views and its
    accuracy stops improving well before forty distinct ones; the cap is a
    ceiling on work, not a judgement that a forty-first view would be harmful.

    The thresholds are in pixels and are frame-size relative only by accident,
    which is a real limitation: 40 px is a larger fraction of a 720p frame than
    of a 4K one. It is left in pixels because the quantity that matters is how
    far the board moved across the sensor, and the alternative -- a fraction of
    frame width -- would make a 4K capture need four times the physical movement
    to count as a new view.
    """
    kept: list[BoardView] = []
    for view in views:
        if len(kept) >= limit:
            break
        centre = np.array(view.centroid)
        extent = view.extent_px
        distinct = True
        for existing in kept:
            moved = float(np.linalg.norm(centre - np.array(existing.centroid)))
            if existing.extent_px <= 0.0 or extent <= 0.0:
                ratio = 1.0
            else:
                ratio = max(extent / existing.extent_px, existing.extent_px / extent)
            if moved < min_separation_px and ratio < min_scale_ratio:
                distinct = False
                break
        if distinct:
            kept.append(view)
    return kept


def _observations(views: list[BoardView], used: set[int], reason: str) -> list[BoardObservation]:
    """One record per detected view, marking which ones survived selection."""
    entries: list[BoardObservation] = []
    for view in views:
        centre = view.centroid
        entries.append(
            BoardObservation(
                frame=view.frame,
                corners=len(view),
                centroid_x=centre[0],
                centroid_y=centre[1],
                used=view.frame in used,
                dropped_reason=None if view.frame in used else reason,
            )
        )
    return entries


def detect_in_video(
    path: Path,
    spec: BoardSpec,
    *,
    stride: int = 1,
    min_corners: int = 6,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
    select: bool = True,
) -> tuple[list[BoardView], DetectionReport]:
    """Scan a clip for the board and return the views worth calibrating from.

    `stride` skips frames before detection rather than after, because detection
    is the cost: a thirty-second clip at 240 fps is seven thousand frames and
    the board is not in a meaningfully different place in any two adjacent ones.
    Selection then runs on what survives, so the stride is an optimisation and
    the selection is the decision -- a large stride cannot make the set more
    diverse than the footage was, and it cannot make it less diverse than
    selection would have.
    """
    tracker = ProgressTracker(
        reporter or NullReporter(), task="detect_board", request_id=request_id
    )
    source: FrameSource = open_video(path)
    found: list[BoardView] = []
    scanned = 0
    charuco = detector(spec)
    short_views = 0

    try:
        total = source.metadata.timing.frame_count
        tracker.report(
            "scanning", 0, total, f"looking for a {spec.squares_x}x{spec.squares_y} board"
        )
        for frame in source.frames(step=max(1, stride)):
            scanned += 1
            view = detect_in_image(
                frame.image,
                spec,
                frame=frame.index,
                timestamp_s=frame.timestamp_s,
                charuco_detector=charuco,
            )
            if view is not None:
                if len(view) >= min_corners:
                    found.append(view)
                else:
                    short_views += 1
            tracker.report("scanning", frame.index + 1, total, f"{len(found)} views")
    finally:
        source.close()

    selected = select_distinct(found) if select else found
    used = {view.frame for view in selected}

    warnings: list[str] = []
    if short_views:
        warnings.append(
            f"{short_views} frame(s) showed the board with fewer than {min_corners} corners "
            "and were skipped. That is usually the board leaving the frame or sitting too "
            "far away for its markers to resolve."
        )
    if found and len(selected) < len(found):
        warnings.append(
            f"{len(found)} frames showed the board and {len(selected)} distinct views were "
            "kept. Views that repeat a position already covered are dropped: they add "
            "weight to the fit without adding evidence, and counting them would make a "
            "static capture look thorough."
        )

    report = DetectionReport(
        frames_scanned=scanned,
        frames_with_board=len(found),
        views_used=len(selected),
        corners_total=sum(len(view) for view in selected),
        board=spec,
        observations=_observations(found, used, "a view already covered this position and scale"),
        warnings=warnings,
    )
    tracker.report("detected", scanned, scanned, f"{len(selected)} views kept")
    return selected, report


def detect_in_directory(
    directory: Path,
    spec: BoardSpec,
    *,
    min_corners: int = 6,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
    select: bool = False,
) -> tuple[list[BoardView], DetectionReport]:
    """Scan a directory of stills for the board.

    Selection defaults to **off** here, which is the opposite of the video path
    and is deliberate. A directory of stills is a set someone assembled on
    purpose, and dropping half of it because two shots look alike would be
    overriding a decision that has already been made by a person. Video is the
    other way round: nobody chose those nine hundred frames.
    """
    files = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No images in {directory}. Looked for: {', '.join(sorted(IMAGE_SUFFIXES))}."
        )

    tracker = ProgressTracker(
        reporter or NullReporter(), task="detect_board", request_id=request_id
    )
    charuco = detector(spec)
    found: list[BoardView] = []
    short_views = 0

    for index, file in enumerate(files):
        loaded = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if loaded is None:
            continue
        image = np.asarray(loaded, dtype=np.uint8)
        view = detect_in_image(
            image, spec, frame=index, timestamp_s=float(index), charuco_detector=charuco
        )
        if view is not None:
            if len(view) >= min_corners:
                found.append(view)
            else:
                short_views += 1
        tracker.report("scanning", index + 1, len(files), f"{len(found)} views")

    selected = select_distinct(found) if select else found
    used = {view.frame for view in selected}

    warnings: list[str] = []
    if short_views:
        warnings.append(
            f"{short_views} image(s) showed the board with fewer than {min_corners} corners "
            "and were skipped."
        )

    report = DetectionReport(
        frames_scanned=len(files),
        frames_with_board=len(found),
        views_used=len(selected),
        corners_total=sum(len(view) for view in selected),
        board=spec,
        observations=_observations(found, used, "dropped by selection"),
        warnings=warnings,
    )
    tracker.report("detected", len(files), len(files), f"{len(selected)} views kept")
    return selected, report
