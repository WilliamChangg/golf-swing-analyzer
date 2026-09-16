# ADR-0006: Pin model artifacts by sha256, keep weights out of git

**Status:** Accepted · 2026-09-15

## Context

The pose landmarker weights total ~46 MB across three variants. They are
redistributable build artifacts, not source. Committing them would bloat every
clone and put binaries in a repository whose history should stay reviewable.

At the same time, "the model" is an input to every measurement this project will
publish. A benchmark or accuracy figure is meaningless if the artifact it was
produced with is unidentified.

Upstream publishes to a `latest` channel
(`.../pose_landmarker_full/float16/latest/...`), so the bytes behind a URL can
change without notice.

## Decision

- Weights are gitignored; `models/manifest.json` **is** committed.
- The manifest pins each artifact by `sha256` and `size_bytes`, recorded from
  the files actually downloaded and used during development.
- `scripts/download_models.py` fetches and verifies against the manifest.
- The doctor probes on-disk models and reports:
  - **OK** — present, size and hash match
  - **ERROR** — size mismatch (a truncated or corrupt download)
  - **DEGRADED** — hash mismatch: the file loads, but it is not the artifact the
    recorded measurements were taken against
  - **MISSING** / **DEGRADED** — absent, depending on whether it is required

## Consequences

- A hash mismatch is surfaced rather than silently accepted, but is not treated
  as a hard failure: on a `latest` channel it most likely means a new upstream
  release, not tampering.
- Re-pinning is deliberate (`--update-hashes`), and the script says plainly that
  recorded benchmarks no longer apply afterwards.
- Downloads stream to a `.part` file and are moved into place only on success,
  so an interrupted transfer cannot leave a truncated file where a valid model
  is expected.
