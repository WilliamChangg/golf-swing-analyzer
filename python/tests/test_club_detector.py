"""The per-frame detector, against a shaft whose angle is known by construction.

`tests/synthetic_club.py` draws the club at an angle that is an input, so an
angular error here is a subtraction. What the fixture does not contain is a
photograph -- no defocus, no compression, a flat background -- so the rates below
are a **floor**.

The tests that matter most are the ones asserting a **negative**, which is the
shape Phases 8 and 9 arrived at one and two layers down:

* that a stationary background line through the hands scores **as much edge
  support as the club and more length**, so ranking by evidence puts a door frame
  above a golf club;
* that a smeared club stops being a line rather than becoming a weaker one, so
  there is no support threshold that recovers it;
* that both ends of a line through the hands are offered as candidates, because
  a door frame and a club pointing the other way are the same pixels.

Those three are why the tracker exists, and they fail if anyone later decides the
Hough score is the confidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analyzer.club import HoughShaftDetector, angle_difference_deg
from analyzer.club.detector import ClubDetectionError, DetectionResult, GripAnchor, ShaftCandidate
from analyzer.contracts.club import ClubConfig, ShaftRefusal
from analyzer.ingestion.reader import VideoFrame
from tests import synthetic_club as fixture

# Every clean-frame assertion below is made against this, and it is loose by an
# order of magnitude: the measured error on a sharp shaft is about half a degree
# at every angle tested. A bound at the measurement would fail on a rounding
# change; one at ten times it still fails if the detector starts finding
# something else.
CLEAN_TOLERANCE_DEG = 3.0


def _best(detector: HoughShaftDetector, item: fixture.RenderedFrame) -> ShaftCandidate:
    result = detector.detect(item.frame, item.anchor)
    assert result.candidates, f"no candidate at {item.angle_deg} deg: {result.refusal}"
    return result.candidates[0]


class TestCleanDetection:
    """A sharp shaft, in every direction a swing reaches."""

    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0, 170.0, -45.0, -90.0, -135.0])
    def test_the_direction_is_recovered(self, angle: float) -> None:
        detector = HoughShaftDetector()
        item = fixture.render(angle_deg=angle)
        best = _best(detector, item)
        assert abs(angle_difference_deg(best.angle_deg, angle)) < CLEAN_TOLERANCE_DEG

    @pytest.mark.parametrize("angle", [30.0, 90.0, -120.0])
    def test_the_length_is_recovered(self, angle: float) -> None:
        detector = HoughShaftDetector()
        best = _best(detector, fixture.render(angle_deg=angle))
        assert best.length_px == pytest.approx(fixture.SHAFT_PX, rel=0.05)

    def test_support_is_measured_along_the_whole_ray(self) -> None:
        """A clean shaft is supported end to end, which is what 1.0 means.

        It is worth pinning because the obvious implementation scores 0.005
        here: Canny finds the two *sides* of a shaft and a ray sampled down its
        middle lands between them. See `_SUPPORT_BAND_TORSO`.
        """
        detector = HoughShaftDetector()
        assert _best(detector, fixture.render(angle_deg=90.0)).support > 0.95

    def test_the_detector_is_stateless(self) -> None:
        """Two calls on one frame agree, and the order frames arrive in is free.

        The Protocol requires it and the tracker depends on it: it visits frames
        in order of how good their evidence is, not in time order, so a detector
        carrying state between calls would make the track depend on the seed.
        """
        detector = HoughShaftDetector()
        items = [fixture.render(index=i, angle_deg=angle) for i, angle in enumerate((20.0, 80.0))]
        forward = [detector.detect(item.frame, item.anchor) for item in items]
        backward = [detector.detect(item.frame, item.anchor) for item in reversed(items)]
        assert forward[0].candidates == backward[1].candidates
        assert forward[1].candidates == backward[0].candidates


class TestBlur:
    """What motion smear does, which is the whole of this phase's difficulty."""

    def test_a_little_smear_is_survived(self) -> None:
        detector = HoughShaftDetector()
        item = fixture.render(angle_deg=45.0, sweep_deg=1.0)
        assert item.blur_px < 10.0
        best = _best(detector, item)
        assert abs(angle_difference_deg(best.angle_deg, 45.0)) < CLEAN_TOLERANCE_DEG

    def test_a_smeared_shaft_stops_being_a_line_rather_than_a_weak_one(self) -> None:
        """**The negative.** There is no support threshold that recovers it.

        A detector whose confidence fell off gradually with blur could be tuned:
        lower the bound and take the noisy answers. This one does not degrade, it
        disappears -- a fan of faint gradients has no edge for Canny to find and
        nothing for the transform to vote on. The consequence is that blur is a
        capture problem with no processing fix, which is why `data/README.md`
        asks for a shutter speed rather than offering a sensitivity setting.
        """
        detector = HoughShaftDetector()
        heavy = fixture.render(angle_deg=45.0, sweep_deg=8.0)
        assert heavy.blur_px > 20.0
        result = detector.detect(heavy.frame, heavy.anchor)

        near_club = [
            candidate
            for candidate in result.candidates
            if abs(angle_difference_deg(candidate.angle_deg, 45.0)) < 20.0
        ]
        assert not near_club, "a heavily smeared shaft should not be found at all"


class TestClutter:
    """Straight background features, which is what the margin check is for."""

    def test_a_door_frame_outranks_the_club_on_evidence(self) -> None:
        """**The negative this phase is built around.**

        A vertical line through the hands has the same support as the club and is
        longer, so the ranking a Hough transform naturally produces puts it
        first. Nothing about the *image* separates them; only what the frame is
        compared against does.
        """
        detector = HoughShaftDetector()
        item = fixture.render(
            angle_deg=45.0, clutter_x=(fixture.ADDRESS_GRIP_PX[0],), grip=fixture.ADDRESS_GRIP_PX
        )
        result = detector.detect(item.frame, item.anchor)

        club = [c for c in result.candidates if abs(angle_difference_deg(c.angle_deg, 45.0)) < 10.0]
        post = [c for c in result.candidates if abs(abs(c.angle_deg) - 90.0) < 10.0]
        assert club and post, "both the club and the door frame should be candidates"
        assert post[0].support >= club[0].support
        assert post[0].length_px > club[0].length_px
        assert result.candidates[0] is post[0], "evidence alone ranks the door frame first"

    def test_a_line_through_the_hands_offers_both_of_its_ends(self) -> None:
        """A door frame and a club pointing the other way are the same pixels.

        The detector therefore offers both readings rather than picking one. With
        no temporal information their supports are equal and the tracker refuses
        the frame as ambiguous, which is the honest answer; with a tracked
        neighbour the prediction separates them. Taking the farther end silently,
        which the first version of this detector did, reported the shaft
        backwards at a confidence of 1.00.
        """
        detector = HoughShaftDetector()
        item = fixture.render(
            angle_deg=45.0, clutter_x=(fixture.ADDRESS_GRIP_PX[0],), grip=fixture.ADDRESS_GRIP_PX
        )
        result = detector.detect(item.frame, item.anchor)
        angles = [candidate.angle_deg for candidate in result.candidates]
        opposed = [
            (a, b)
            for a in angles
            for b in angles
            if abs(abs(angle_difference_deg(a, b)) - 180.0) < 10.0
        ]
        assert opposed, f"expected two opposed readings of the door frame, got {angles}"

    def test_a_distant_background_line_is_rejected_on_distance(self) -> None:
        """The grip test does most of the work, before anything else is asked."""
        detector = HoughShaftDetector()
        far = fixture.ADDRESS_GRIP_PX[0] + 0.6 * fixture.TORSO_PX
        item = fixture.render(angle_deg=45.0, clutter_x=(far,), grip=fixture.ADDRESS_GRIP_PX)
        result = detector.detect(item.frame, item.anchor)
        vertical = [c for c in result.candidates if abs(abs(c.angle_deg) - 90.0) < 10.0]
        assert not vertical


class TestGeometry:
    """The bounds, each stated in torso lengths so framing does not change them."""

    def test_a_shaft_that_is_too_short_is_not_a_candidate(self) -> None:
        detector = HoughShaftDetector(ClubConfig(min_shaft_length_torso=2.0))
        result = detector.detect(*_frame_and_anchor(fixture.render(angle_deg=45.0)))
        assert not result.candidates
        assert result.refusal is ShaftRefusal.NO_CANDIDATE

    def test_a_shaft_that_is_too_long_is_not_a_candidate(self) -> None:
        detector = HoughShaftDetector(ClubConfig(max_shaft_length_torso=0.5))
        result = detector.detect(*_frame_and_anchor(fixture.render(angle_deg=45.0)))
        assert not result.candidates

    def test_the_bounds_scale_with_the_subject(self) -> None:
        """The same configuration, a subject twice as far away, the same verdict.

        Not a restatement of the arithmetic: it is the reason the bounds are in
        torso lengths at all. A detector configured in pixels would find a club
        on a close-framed clip and refuse the identical swing filmed from further
        back, with nothing in the output saying why.
        """
        detector = HoughShaftDetector()
        near = fixture.render(angle_deg=60.0)
        assert detector.detect(near.frame, near.anchor).candidates

        half = _downscale(near.frame)
        anchor = GripAnchor(
            x=near.anchor.x / 2.0, y=near.anchor.y / 2.0, torso_px=near.anchor.torso_px / 2.0
        )
        result = detector.detect(half, anchor)
        assert result.candidates
        assert abs(angle_difference_deg(result.candidates[0].angle_deg, 60.0)) < CLEAN_TOLERANCE_DEG


class TestRefusals:
    """Each reason names a different thing to change about the capture."""

    def test_an_empty_frame_has_no_edges(self) -> None:
        detector = HoughShaftDetector()
        blank = VideoFrame(
            index=0,
            timestamp_s=0.0,
            image=np.full((400, 400, 3), 200, dtype=np.uint8),
        )
        result = detector.detect(blank, GripAnchor(x=200.0, y=200.0, torso_px=100.0))
        assert result.refusal is ShaftRefusal.NO_EDGES
        assert not result.candidates

    def test_an_anchor_outside_the_frame_refuses_rather_than_searching(self) -> None:
        detector = HoughShaftDetector()
        item = fixture.render(angle_deg=45.0)
        anchor = GripAnchor(x=-5000.0, y=-5000.0, torso_px=fixture.TORSO_PX)
        assert detector.detect(item.frame, anchor).refusal is ShaftRefusal.NO_EDGES

    def test_a_result_carries_candidates_or_a_reason_and_never_both(self) -> None:
        with pytest.raises(ClubDetectionError):
            DetectionResult(candidates=(), refusal=None, edge_pixels=0, lines_found=0)
        with pytest.raises(ClubDetectionError):
            DetectionResult(
                candidates=(ShaftCandidate(1.0, 1.0, 1.0, 1.0, 0.0),),
                refusal=ShaftRefusal.NO_EDGES,
                edge_pixels=0,
                lines_found=0,
            )

    def test_an_anchor_needs_a_positive_scale(self) -> None:
        """Every bound is a multiple of it, so a zero makes them all zero."""
        with pytest.raises(ClubDetectionError):
            GripAnchor(x=0.0, y=0.0, torso_px=0.0)
        with pytest.raises(ClubDetectionError):
            GripAnchor(x=0.0, y=0.0, torso_px=float("nan"))


class TestProvenance:
    def test_the_detector_records_what_ran(self) -> None:
        """Measured from the loaded library, not copied from the dependency pin."""
        import cv2

        info = HoughShaftDetector().info
        assert info.name == "hough_shaft"
        assert info.opencv_version == cv2.__version__


def _frame_and_anchor(item: fixture.RenderedFrame) -> tuple[VideoFrame, GripAnchor]:
    return item.frame, item.anchor


def _downscale(frame: VideoFrame) -> VideoFrame:
    """Half-size, which is the same swing filmed from twice the distance."""
    import cv2

    height, width = frame.image.shape[:2]
    resized = cv2.resize(frame.image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    return VideoFrame(index=frame.index, timestamp_s=frame.timestamp_s, image=resized)


def test_the_fixture_smear_matches_the_geometry_it_claims() -> None:
    """A fixture whose own arithmetic is wrong measures nothing.

    `blur_px` is the arc the tip sweeps during the exposure, and every table in
    `scripts/benchmark_club.py` is plotted against it. Checking it against the
    definition costs one line and stops a renderer change quietly rescaling every
    published figure.
    """
    item = fixture.render(angle_deg=0.0, sweep_deg=10.0)
    assert item.blur_px == pytest.approx(math.radians(10.0) * fixture.SHAFT_PX)
