"""Finding the landmarks behind a labelled clip.

A label names a clip; a dataset needs that clip's filtered trajectories. This is
the seam between them, and it is a separate module because the two obvious
implementations are genuinely different: the tests hand the builder sequences
they generated, and the CLI has to find an extraction on disk that somebody ran
earlier.

**It does not extract.** A provider that silently ran pose estimation would turn
`analyzer dataset` into an hours-long command that looks like it has hung, on a
corpus where one missing extraction is indistinguishable from a hundred. It says
which clip is missing and which command produces it.

**It finds clips by content, then by path.** `ContentKey` is the identity: a
labelled clip that has been renamed or moved still matches its extraction, which
is the whole reason the key is stored on the label. The recorded path is the
fallback for a label written before a clip had a key, and it is a hint rather
than an identity -- two different clips can sit at one path on two machines.
"""

from __future__ import annotations

from pathlib import Path

from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.labels import ClipLabel
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import FilteredSequence, filter_sequence
from analyzer.ingestion import probe_video
from analyzer.ml.dataset import DatasetError, filter_config_for
from analyzer.pose.store import cached_sequence_path, read_sequence


def cached_pose_provider(
    model: str | None = None,
    *,
    config: FilterConfig | None = None,
    search: tuple[Path, ...] = (),
):
    """A `PoseProvider` reading extractions out of the pose cache.

    `search` lists directories to look in when a label's recorded path is not
    where the clip is any more, which is the ordinary state of a dataset that
    outlives one machine. Each candidate is probed and its content key compared
    against the label's, so a clip is only accepted when its bytes match -- a
    file of the right name is not the right file.
    """
    from analyzer.pose.estimator import resolve_model

    entry, _ = resolve_model(model)

    def provide(label: ClipLabel) -> FilteredSequence:
        poses = _locate(label, entry.name, search)
        sequence = read_sequence(poses)
        return filter_sequence(
            sequence,
            filter_config_for(label.fps, config),
            space=LandmarkSpace.FRAME_WIDTHS,
            slow_motion_factor=label.slow_motion_factor,
        )

    return provide


def _locate(label: ClipLabel, model_name: str, search: tuple[Path, ...]) -> Path:
    """The pose file for a labelled clip, or an error naming what to run."""
    if label.content_key is not None:
        direct = cached_sequence_path(label.content_key, model_name)
        if direct.exists():
            return direct

    candidates = [Path(label.clip_path)]
    candidates.extend(directory / Path(label.clip_path).name for directory in search)
    for candidate in candidates:
        if not candidate.exists():
            continue
        metadata = probe_video(candidate)
        if (
            label.content_key is not None
            and metadata.content_key.digest != label.content_key.digest
        ):
            continue
        poses = cached_sequence_path(metadata.content_key, model_name)
        if poses.exists():
            return poses
        raise DatasetError(
            f"{candidate.name} is the clip this label describes, and it has not been "
            f"pose-extracted with '{model_name}'. Run: analyzer extract {candidate} "
            f"--model {model_name}"
        )

    raise DatasetError(
        f"No extraction found for {label.clip_key} and the clip is not at "
        f"{label.clip_path}. Point --search at the directory holding the footage, or "
        f"re-extract it with: analyzer extract <clip> --model {model_name}"
    )
