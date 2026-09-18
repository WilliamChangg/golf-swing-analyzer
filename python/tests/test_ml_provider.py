"""Finding the landmarks behind a labelled clip.

The seam between a label -- which names a clip -- and a dataset, which needs that
clip's filtered trajectories. Small, and the only part of the path from
`analyzer label` to `analyzer train` that touches the filesystem, so it is the
part where a labelled set quietly becomes a smaller labelled set.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.labels import ClipLabel, LabelProvenance
from analyzer.contracts.pose import LANDMARK_COUNT
from analyzer.ml.dataset import DatasetError
from analyzer.ml.provider import cached_pose_provider
from analyzer.pose.store import cached_sequence_path, write_sequence
from tests import synthetic
from tests.conftest import requires_pose_model

MODEL = "pose_landmarker_full"


def label_for(key: ContentKey | None, *, fps: float = 120.0, path: str = "/gone.mov") -> ClipLabel:
    return ClipLabel(
        clip_path=path,
        content_key=key,
        player_id="p1",
        session_id="s1",
        swing_id="w1",
        provenance=LabelProvenance.HUMAN,
        labeller="wc",
        labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
        is_swing=False,
        frames=100,
        fps=fps,
    )


@requires_pose_model
def test_a_clip_is_found_by_its_contents_not_its_path(isolated_cache: Path) -> None:
    """A labelled clip that has been renamed or moved keeps its labels.

    That is the whole reason a `ContentKey` is stored on the label, and this is
    the test of it: the recorded path points at nothing at all.
    """
    key = ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest="a" * 64, size_bytes=10)
    sequence = synthetic.swing_sequence(fps=120.0)
    write_sequence(sequence, cached_sequence_path(key, MODEL), landmark_count=LANDMARK_COUNT)

    filtered = cached_pose_provider(MODEL)(label_for(key))
    assert len(filtered.t) == len(sequence.frames)


@requires_pose_model
def test_a_low_frame_rate_clip_is_filtered_with_a_window_it_can_support(
    isolated_cache: Path,
) -> None:
    """Otherwise every 30 fps clip in a set is dropped for having no torso."""
    key = ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest="b" * 64, size_bytes=10)
    write_sequence(
        synthetic.swing_sequence(fps=30.0),
        cached_sequence_path(key, MODEL),
        landmark_count=LANDMARK_COUNT,
    )

    filtered = cached_pose_provider(MODEL)(label_for(key, fps=30.0))
    assert filtered.report.config.smoothing.window_s > 0.1


@requires_pose_model
def test_a_missing_extraction_names_the_command_that_makes_one(isolated_cache: Path) -> None:
    """A provider that extracted on demand would turn one command into hours."""
    key = ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest="c" * 64, size_bytes=10)
    with pytest.raises(DatasetError, match="analyzer extract"):
        cached_pose_provider(MODEL)(label_for(key))


@requires_pose_model
def test_a_label_with_no_content_key_and_no_clip_is_refused(isolated_cache: Path) -> None:
    with pytest.raises(DatasetError, match="No extraction found"):
        cached_pose_provider(MODEL)(label_for(None))
