"""The physical board, and the OpenCV object that describes it.

One place builds a `cv2.aruco.CharucoBoard` from a `BoardSpec`, so the board
that is generated for printing and the board that is looked for in footage can
never be two different boards. That failure is worth designing out rather than
testing for: a detector configured for a board the user did not print finds
nothing at all, and "no corners detected" looks identical to bad lighting, a
dark room, or a board held too far away.

`generate_board_image` exists for the same reason. A board downloaded from a
calibration-pattern website is a board whose dictionary, square count and legacy
flag are all guesses; one generated here from the spec that will be used to
detect it is none of those things, and it prints its own parameters along the
bottom edge so the sheet on the wall says what it is.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.calibration import BoardFamily, BoardSpec


class BoardError(ValueError):
    """A board specification cannot describe a physical board."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def _dictionary(family: BoardFamily) -> cv2.aruco.Dictionary:
    """The ArUco dictionary named by a family.

    Resolved through `getattr` against the `cv2.aruco` constants rather than a
    hand-written mapping, because the enum member names are exactly OpenCV's own
    and a mapping would be a second list to keep in step with the first.
    """
    constant = getattr(cv2.aruco, family.value, None)
    if constant is None:  # pragma: no cover - the enum is closed over cv2's names
        raise BoardError(
            f"OpenCV {cv2.__version__} has no ArUco dictionary called {family.value}.",
            remediation="Choose another board family.",
        )
    return cv2.aruco.getPredefinedDictionary(constant)


def validate(spec: BoardSpec) -> None:
    """Refuse a spec that cannot describe a board that would work.

    Two checks, both catching mistakes that produce a board which detects badly
    rather than one that fails to build.
    """
    if spec.marker_length_m >= spec.square_length_m:
        raise BoardError(
            f"A marker of {spec.marker_length_m * 1000:.0f} mm cannot fit inside a "
            f"{spec.square_length_m * 1000:.0f} mm square.",
            remediation=(
                "Make the marker about 0.75 of the square, which leaves the white "
                "margin the detector needs to find each marker's border."
            ),
        )

    dictionary = _dictionary(spec.family)
    # Charuco fills the black squares of the chessboard, which is half of them.
    required = (spec.squares_x * spec.squares_y) // 2
    available = int(dictionary.bytesList.shape[0])
    if required > available:
        raise BoardError(
            f"A {spec.squares_x}x{spec.squares_y} board needs {required} markers and "
            f"{spec.family.value} contains {available}.",
            remediation="Use a larger dictionary, such as DICT_5X5_250, or a smaller board.",
        )


def build(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    """The OpenCV board described by a spec.

    Lengths are handed over in **metres**, which makes every board pose OpenCV
    returns metric without a conversion anywhere. That is where the scale of
    every later 3D claim comes from, and keeping it in one unit from the ruler to
    the triangulation is what stops a factor of a thousand appearing in the
    middle of it.
    """
    validate(spec)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        _dictionary(spec.family),
    )
    # The marker layout changed in OpenCV 4.6. A board printed from an older
    # generator and detected as a modern one yields corners in the wrong places,
    # which is far worse than yielding none: the calibration succeeds and is
    # wrong. The flag is carried on the spec so the sheet's origin decides.
    board.setLegacyPattern(spec.legacy_pattern)
    return board


def detector(spec: BoardSpec) -> cv2.aruco.CharucoDetector:
    """A detector for this board, with OpenCV's default parameters.

    The parameters are left alone deliberately. Tuning a detector against the
    handful of clips available here would fit it to this room's lighting, and
    the failure that produces -- fewer corners on someone else's footage -- is
    invisible from inside this project.
    """
    return cv2.aruco.CharucoDetector(build(spec))


def object_points(spec: BoardSpec) -> NDArray[np.float64]:
    """The board's chessboard corners in its own frame, in metres.

    Ordered as OpenCV ids them, so index `i` is the corner with id `i` and a
    detection's ids index straight into this array. Z is zero for every corner:
    the board is a plane, and that is the assumption the whole method rests on
    -- which is why it has to be mounted flat rather than held in one hand.
    """
    return np.asarray(build(spec).getChessboardCorners(), dtype=np.float64)


def board_texture(
    spec: BoardSpec, *, board_px: tuple[int, int], margin_px: int = 40
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    """Render the board inside a white margin, and say how large it came out.

    Returns the image and the **actual** board rectangle in pixels, which is not
    always the one asked for. OpenCV's renderer refuses some combinations of
    board size and square count outright -- it computes each marker's rectangle
    by rounding and occasionally rounds one a pixel outside the image -- so a
    request that fails is retried a pixel larger until one is accepted.

    The returned size is what a caller must use to map board metres onto texture
    pixels. Assuming the requested size is the failure this signature exists to
    prevent: the rendering would be a pixel or two off, which is invisible in
    the picture and shows up as a systematic bias in everything measured from it.

    The margin is not decorative. The outermost markers need white around them
    or they cannot be detected at all, which on a printed sheet is the
    difference between a board that calibrates and one that never quite does.
    """
    validate(spec)
    if margin_px < 1:
        raise BoardError(
            "A board needs a white margin around it.",
            remediation=(
                "Use a margin of at least a few pixels: the outermost markers cannot be "
                "detected without white space around their borders."
            ),
        )

    board = build(spec)
    width, height = board_px
    for attempt in range(16):
        candidate = (width + attempt, height + attempt)
        try:
            image = board.generateImage(
                (candidate[0] + 2 * margin_px, candidate[1] + 2 * margin_px),
                marginSize=margin_px,
            )
        except cv2.error:
            continue
        return np.asarray(image, dtype=np.uint8), candidate

    raise BoardError(  # pragma: no cover - 16 consecutive refusals has not been observed
        f"OpenCV would not render a {spec.squares_x}x{spec.squares_y} board at any size "
        f"near {board_px[0]}x{board_px[1]} pixels.",
        remediation="Ask for a different resolution, or a board with a different square count.",
    )


def generate_board_image(
    spec: BoardSpec, *, pixels_per_metre: float = 4000.0, margin_px: int = 40
) -> NDArray[np.uint8]:
    """Render the board for printing, at a stated scale.

    The default is about 100 dots per inch of board, which is enough for the
    marker borders to survive an office printer. What it is *not* is a promise
    about the printed size: printers scale to fit the page, so the length that
    matters is the one measured on the sheet afterwards with a ruler and put
    into `BoardSpec.square_length_m`. Every metric claim this system ever makes
    descends from that measurement, and it is the one step of the process no
    software can check.
    """
    image, _ = board_texture(
        spec,
        board_px=(
            round(spec.width_m * pixels_per_metre),
            round(spec.height_m * pixels_per_metre),
        ),
        margin_px=margin_px,
    )
    return image
