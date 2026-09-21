# Pose models

Weights are downloaded, not committed. [`manifest.json`](manifest.json) pins each
artifact by exact size and SHA-256. The installed variants are MediaPipe's lite,
full and heavy pose landmarker models; full is required and is the default.
The source is Google's MediaPipe model distribution. See the manifest and
[ADR-0006](../docs/decisions/ADR-0006-model-pinning.md) for provenance and pinning.

## Install and repair

From the repository root, with the Python environment installed:

```bash
uv run --project python python scripts/download_models.py
uv run --project python python scripts/download_models.py --all
uv run --project python python scripts/download_models.py --only pose_landmarker_heavy
uv run --project python python scripts/download_models.py --only pose_landmarker_full --force
```

The default fetches required models. `--all` includes optional variants; `--force`
re-downloads an installed file. The desktop's **Environment → Manage models**
provides equivalent manifest-pinned installation, verification and repair.
`GSA_MODELS_DIR` can relocate the directory but must also contain `manifest.json`.

Both entry points use a streaming installer with a unique temporary `.part` file,
size/hash verification and atomic replacement. A timeout, partial transfer or
mismatched artifact preserves the existing file and removes the partial download.

**Update installs the application's pin**, not any arbitrary new upstream release.
The developer-only `--update-hashes` flag intentionally adopts downloaded bytes
and changes the manifest; repeat validation and benchmarks before publishing new
measurements. Do not use it merely to hide a failed verification.

## What verification means

The manager reports `verified`, `missing` or `mismatch`. The doctor uses broader
component statuses: a required absent file is `MISSING`, an optional absent file
is `DEGRADED`, size mismatch is `ERROR`, hash mismatch is `DEGRADED`, and matching
bytes are `OK`. A byte check alone does **not** establish that a model loads.
**Re-check** and `analyzer doctor` separately perform the runtime probe.

The full variant remains the default without a measured landmark-accuracy ranking
on golfers. Timings and detection coverage do not establish that another variant
is more or less anatomically accurate. Current execution measurements are in
[the generated appendix](../docs/benchmarks.md).

MediaPipe runs with an explicit CPU delegate. Torch device selection concerns
locally trained temporal models, whose checkpoints and model cards live in the
application data directory rather than this manifest. `analyzer models` lists
that temporal registry. See [Modeling](../docs/modeling.md) for labels, splits and
current evaluation limits.
