"""Persisting pose sequences as Parquet.

Long format -- one row per (frame, space, landmark) -- rather than 198 columns
of `left_wrist_image_x`. Long format survives schema change: adding a landmark
or a coordinate space adds rows, not columns, so a file written today still
reads after Phase 9 introduces a triangulated space. It is also the shape
pandas, polars and DuckDB all want for a group-by, which is what every analysis
over these does.

Frames in which nothing was detected are written as rows of NaN rather than
omitted. That costs space on a clip where the subject is mostly absent, and buys
two things worth more: the file stays rectangular, so reading needs no
reconstruction of which frames are missing; and a gap arrives at the filtering
layer already represented as NaN, which is exactly the representation Phase 3
needs in order to refuse to interpolate across it.

Everything needed to interpret the file -- schema version, source video, model
digest, thresholds, run statistics -- is written into the Parquet key-value
metadata, so the file is self-describing and a result can always be traced back
to the artifact that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.pose import (
    POSE_SCHEMA_VERSION,
    LandmarkPoint,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)

# Metadata keys. Prefixed so they cannot collide with anything a Parquet writer
# adds of its own accord.
_META_PREFIX = b"gsa."
_KEY_SCHEMA_VERSION = _META_PREFIX + b"schema_version"
_KEY_VIDEO_PATH = _META_PREFIX + b"video_path"
_KEY_CONTENT_KEY = _META_PREFIX + b"video_content_key"
_KEY_MODEL = _META_PREFIX + b"model"
_KEY_EXTRACTED_AT = _META_PREFIX + b"extracted_at"
_KEY_STATS = _META_PREFIX + b"stats"

# float32 throughout: landmark coordinates are normalised model outputs with far
# fewer than seven significant digits of real precision, so float64 would store
# noise at twice the size.
SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int32(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        pa.field("detected", pa.bool_(), nullable=False),
        pa.field("space", pa.dictionary(pa.int8(), pa.string()), nullable=False),
        pa.field("landmark", pa.int16(), nullable=False),
        pa.field("x", pa.float32(), nullable=True),
        pa.field("y", pa.float32(), nullable=True),
        pa.field("z", pa.float32(), nullable=True),
        pa.field("visibility", pa.float32(), nullable=True),
        pa.field("presence", pa.float32(), nullable=True),
    ]
)


class PoseStoreError(RuntimeError):
    """A pose file could not be read, or is not one this build understands."""


def _rows_for_space(
    frame: PoseFrame, space: LandmarkSpace, points: list[LandmarkPoint], landmark_count: int
) -> dict[str, list[Any]]:
    """Build the column fragments for one frame in one space."""
    if frame.detected and points:
        return {
            "landmark": list(range(len(points))),
            "x": [p.x for p in points],
            "y": [p.y for p in points],
            "z": [p.z for p in points],
            "visibility": [p.visibility for p in points],
            "presence": [p.presence for p in points],
        }
    # Absent: a full set of landmark slots, every value NaN.
    return {
        "landmark": list(range(landmark_count)),
        "x": [None] * landmark_count,
        "y": [None] * landmark_count,
        "z": [None] * landmark_count,
        "visibility": [None] * landmark_count,
        "presence": [None] * landmark_count,
    }


def to_table(sequence: PoseSequence, *, landmark_count: int) -> pa.Table:
    """Flatten a sequence into the long-format table, metadata attached."""
    columns: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}

    for frame in sequence.frames:
        for space, points in (
            (LandmarkSpace.IMAGE, frame.image),
            (LandmarkSpace.HIP_LOCAL, frame.hip_local),
        ):
            fragment = _rows_for_space(frame, space, points, landmark_count)
            count = len(fragment["landmark"])
            columns["frame_index"].extend([frame.frame_index] * count)
            columns["timestamp_s"].extend([frame.timestamp_s] * count)
            columns["detected"].extend([frame.detected] * count)
            columns["space"].extend([space.value] * count)
            for name, values in fragment.items():
                columns[name].extend(values)

    metadata = {
        _KEY_SCHEMA_VERSION: str(sequence.schema_version).encode(),
        _KEY_VIDEO_PATH: sequence.video_path.encode(),
        _KEY_CONTENT_KEY: sequence.video_content_key.model_dump_json().encode(),
        _KEY_MODEL: sequence.model.model_dump_json().encode(),
        _KEY_EXTRACTED_AT: sequence.extracted_at.isoformat().encode(),
        _KEY_STATS: sequence.stats.model_dump_json().encode(),
    }
    return pa.Table.from_pydict(columns, schema=SCHEMA).replace_schema_metadata(metadata)


def write_sequence(sequence: PoseSequence, path: Path, *, landmark_count: int) -> Path:
    """Write a sequence to Parquet, atomically.

    Write-then-rename so an interrupted run cannot leave a half-written file
    that a later read would have to defend against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = to_table(sequence, landmark_count=landmark_count)

    temporary = path.with_suffix(".parquet.tmp")
    # zstd over snappy: these files are long and highly repetitive (a frame
    # index repeated 66 times, runs of NaN), which zstd exploits far better, and
    # nothing here is read in a latency-sensitive path.
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    return path


def _require(metadata: dict[bytes, bytes], key: bytes, path: Path) -> bytes:
    value = metadata.get(key)
    if value is None:
        raise PoseStoreError(
            f"{path.name} is missing the '{key.decode()}' metadata key, so it cannot be "
            "interpreted. It was probably not written by this system."
        )
    return value


def read_sequence(path: Path) -> PoseSequence:
    """Read a sequence back, refusing anything this build cannot interpret."""
    try:
        table = pq.read_table(path)
    except FileNotFoundError as exc:
        raise PoseStoreError(f"No pose file at {path}.") from exc
    except pa.ArrowInvalid as exc:
        raise PoseStoreError(f"{path.name} is not a readable Parquet file: {exc}") from exc

    metadata = table.schema.metadata or {}

    version_raw = _require(metadata, _KEY_SCHEMA_VERSION, path)
    try:
        version = int(version_raw)
    except ValueError as exc:
        raise PoseStoreError(
            f"{path.name} declares a non-numeric schema version {version_raw!r}."
        ) from exc

    if version != POSE_SCHEMA_VERSION:
        # Refused rather than coerced: a field may have changed meaning between
        # versions, and re-extracting is cheap next to getting that wrong.
        raise PoseStoreError(
            f"{path.name} was written with pose schema version {version}, but this build "
            f"understands {POSE_SCHEMA_VERSION}. Re-run the extraction."
        )

    model = PoseModelInfo.model_validate_json(_require(metadata, _KEY_MODEL, path))
    content_key = ContentKey.model_validate_json(_require(metadata, _KEY_CONTENT_KEY, path))
    stats = PoseExtractionStats.model_validate_json(_require(metadata, _KEY_STATS, path))
    extracted_at = _require(metadata, _KEY_EXTRACTED_AT, path).decode()
    video_path = _require(metadata, _KEY_VIDEO_PATH, path).decode()

    return PoseSequence(
        schema_version=version,
        video_path=video_path,
        video_content_key=content_key,
        model=model,
        extracted_at=extracted_at,  # type: ignore[arg-type]  # pydantic parses the ISO string
        stats=stats,
        frames=_frames_from_table(table),
    )


def _frames_from_table(table: pa.Table) -> list[PoseFrame]:
    """Rebuild `PoseFrame`s from long-format rows.

    Rows are grouped by frame index in a single pass rather than sorted, because
    the writer emits them in order and re-sorting a million-row table to
    rediscover that would be wasted work.
    """
    data = table.to_pydict()
    frames: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for row in range(table.num_rows):
        index = int(data["frame_index"][row])
        entry = frames.get(index)
        if entry is None:
            entry = {
                "timestamp_s": float(data["timestamp_s"][row]),
                "detected": bool(data["detected"][row]),
                LandmarkSpace.IMAGE.value: [],
                LandmarkSpace.HIP_LOCAL.value: [],
            }
            frames[index] = entry
            order.append(index)

        if not entry["detected"]:
            continue

        x = data["x"][row]
        if x is None:
            # A detected frame with no landmarks in this space: HIP_LOCAL is
            # absent on some model builds, and an empty list says so exactly.
            continue

        entry[str(data["space"][row])].append(
            LandmarkPoint(
                x=float(x),
                y=float(data["y"][row]),
                z=float(data["z"][row]),
                visibility=float(data["visibility"][row]),
                presence=float(data["presence"][row]),
            )
        )

    return [
        PoseFrame(
            frame_index=index,
            timestamp_s=frames[index]["timestamp_s"],
            detected=frames[index]["detected"],
            image=frames[index][LandmarkSpace.IMAGE.value],
            hip_local=frames[index][LandmarkSpace.HIP_LOCAL.value],
        )
        for index in order
    ]
