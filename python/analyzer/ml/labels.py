"""Reading and writing hand-labelled swings.

One JSON file per labelled clip, in a directory. Not a database and not one
combined file, for reasons that are all about what happens to a labelled set
over years rather than about what is convenient to write.

A label is small, hand-made and irreplaceable, and the format it lives in should
be one a person can open, read, diff and fix with a text editor when something
goes wrong at three in the morning. One file per clip means two labellers working
in parallel never touch the same file, a version control history shows which
clip's judgement changed, and a corrupted file costs one clip rather than the
set.

The cost is that loading the set is a directory walk, which for tens of thousands
of labels would be the wrong trade. It is not the wrong trade at any size this
project will reach by hand: a labeller marking four events per clip produces a
few hundred files in a day.

**A file that this build does not understand is refused, not adapted.** Reading a
label written under a future schema and interpreting the fields that happen to
still be there is how a set quietly acquires two meanings for one field.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from analyzer.contracts.labels import LABEL_SCHEMA_VERSION, ClipLabel, LabelSet
from analyzer.paths import labels_dir

# Characters allowed in the part of a filename taken from an id. Ids come from a
# person typing at a prompt, so they can contain anything; the file name they
# produce must not be able to escape the directory or collide with a sibling by
# normalisation.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class LabelStoreError(RuntimeError):
    """A label file could not be read, or is not one this build understands."""


def _safe(part: str) -> str:
    cleaned = _UNSAFE.sub("-", part.strip()).strip("-.")
    return cleaned or "unnamed"


def label_filename(label: ClipLabel) -> str:
    """The file one label is written to.

    Named from the three ids that identify the clip within the set, so the
    directory listing is readable and a re-label of the same clip overwrites its
    own file instead of accumulating near-duplicates that would then both be
    loaded and both be counted.
    """
    return f"{_safe(label.player_id)}__{_safe(label.session_id)}__{_safe(label.swing_id)}.json"


def write_label(label: ClipLabel, root: Path | None = None) -> Path:
    """Write one label, returning the file it went to."""
    directory = root if root is not None else labels_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / label_filename(label)
    payload = json.dumps(label.model_dump(mode="json"), indent=2, sort_keys=True)
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def read_label(path: Path) -> ClipLabel:
    """Read one label file, refusing anything this build cannot interpret."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelStoreError(f"{path.name} could not be read as JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise LabelStoreError(f"{path.name} does not contain a label object.")

    version = raw.get("schema_version")
    if version != LABEL_SCHEMA_VERSION:
        raise LabelStoreError(
            f"{path.name} is schema version {version!r}; this build reads "
            f"{LABEL_SCHEMA_VERSION}. The file is not being guessed at: a field whose "
            "meaning changed between versions would be read as the wrong quantity."
        )

    try:
        return ClipLabel.model_validate(raw)
    except ValidationError as exc:
        raise LabelStoreError(f"{path.name} is not a valid label: {exc}") from exc


def load_label_set(root: Path | None = None, *, strict: bool = True) -> LabelSet:
    """Every label under a directory, as one set.

    `strict` decides what an unreadable file does. The default refuses the whole
    set, because a set that silently drops the files it could not parse reports
    counts that are smaller than the work that went into it, and nobody notices
    until a split is made from the wrong population. `strict=False` exists for
    the CLI's listing command, where showing what *can* be read is the point.
    """
    directory = root if root is not None else labels_dir()
    if not directory.exists():
        return LabelSet(clips=[])

    clips: list[ClipLabel] = []
    problems: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            clips.append(read_label(path))
        except LabelStoreError as exc:
            if strict:
                raise
            problems.append(str(exc))

    duplicates = _duplicate_keys(clips)
    if duplicates:
        raise LabelStoreError(
            "Two label files describe the same clip: "
            + ", ".join(sorted(duplicates))
            + ". One of them is a copy made under a different id, which would put the "
            "same swing on both sides of a split."
        )
    return LabelSet(clips=clips)


def _duplicate_keys(clips: Iterable[ClipLabel]) -> set[str]:
    """Clip keys, and clip content keys, that appear more than once.

    Content keys are checked as well as ids because the failure worth catching is
    the same footage labelled twice under two names -- a file copied into two
    session folders, which no id comparison can see.
    """
    seen_keys: set[str] = set()
    seen_content: dict[str, str] = {}
    duplicates: set[str] = set()
    for clip in clips:
        if clip.clip_key in seen_keys:
            duplicates.add(clip.clip_key)
        seen_keys.add(clip.clip_key)

        if clip.content_key is None:
            continue
        digest = clip.content_key.digest
        previous = seen_content.get(digest)
        if previous is not None and previous != clip.clip_key:
            duplicates.add(f"{previous} and {clip.clip_key} (same file contents)")
        seen_content[digest] = clip.clip_key
    return duplicates
