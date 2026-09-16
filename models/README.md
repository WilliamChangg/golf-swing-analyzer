# Models

Model weights are **not committed** to this repository. `manifest.json` is, and
it pins every artifact by sha256 — see
[ADR-0006](../docs/decisions/ADR-0006-model-pinning.md) for why.

## Fetching

```bash
python scripts/download_models.py            # required models only
python scripts/download_models.py --all      # every variant
python scripts/download_models.py --only pose_landmarker_heavy
```

Downloads stream to a `.part` file and are moved into place only after a
successful transfer, so an interrupted download cannot leave a truncated file
where a valid model is expected.

## Current models

| Name                    | Size    | Precision | Required |
| ----------------------- | ------- | --------- | -------- |
| `pose_landmarker_lite`  | 5.8 MB  | float16   | no       |
| `pose_landmarker_full`  | 9.4 MB  | float16   | **yes**  |
| `pose_landmarker_heavy` | 30.7 MB | float16   | no       |

Source: Google MediaPipe Pose Landmarker, distributed from
`storage.googleapis.com/mediapipe-models/`. Licensed under Apache 2.0.

`full` is the default: the accuracy/latency balance suits per-frame video
processing better than `heavy`, and `lite` loses landmark precision that the
biomechanics layer depends on. That default is a starting point, not a measured
conclusion — Phase 2 will benchmark all three and the choice will be revisited
against real numbers.

## Verification

`analyzer doctor` checks every model in the manifest and reports:

| Result   | Meaning                                               |
| -------- | ----------------------------------------------------- |
| OK       | present; size and sha256 match                        |
| ERROR    | size mismatch — truncated or corrupt                  |
| DEGRADED | hash mismatch — loads, but is not the pinned artifact |
| MISSING  | absent and required                                   |

Upstream publishes to a `latest` channel, so a hash mismatch most likely means a
new upstream release rather than tampering. Re-pin deliberately with
`--update-hashes`, and re-run any benchmarks afterwards: recorded measurements
do not carry over to a different artifact.
