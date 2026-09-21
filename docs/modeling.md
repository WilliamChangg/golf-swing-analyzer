# Models, labels and temporal learning

There are two distinct model systems: downloaded pose artifacts and locally
trained temporal classifiers. Neither the presence of a model file nor a passing
inference test establishes accuracy on golfers.

## Pose model registry

[`models/manifest.json`](../models/manifest.json) pins the lite, full and heavy
MediaPipe pose artifacts by size and SHA-256. `pose_landmarker_full` is the default.
The manifest and adapter specify backend, precision, input expectations and
explicit CPU execution. An upstream `latest` URL is a location, not a version;
the digest is the artifact identity.

**Environment → Manage models** reports installed and expected hashes and offers
download, verify, reinstall and update to the pinned artifact. Downloads stream
to a temporary file and replace the installed file only after verification.
The developer's `--update-hashes` operation deliberately changes the pin and
requires renewed validation. See [Model operations](../models/README.md).

Pose cache reuse checks video and model identity, and downstream report caches
include the extraction identity. Installing another artifact must not make an
old report appear to have been computed with it. Runtime probing is separate
from checking bytes: **Re-check** executes inference.

The existing real-clip benchmarks establish execution, throughput and detection
coverage. They do not compare landmark accuracy across model variants. Claims
that lite loses a measured amount of precision or that full is an empirically
optimal accuracy/latency compromise would exceed the current evidence.

## Human labels and dataset state

No labelled real dataset adequate for training and held-out accuracy claims is
provided. Local reference videos are uncommitted and do not become labels merely
because the heuristic detector produced events.

```bash
uv run --project python analyzer label swing.mov --player player-01 --session range-01 --labeller reviewer-01
uv run --project python analyzer labels
uv run --project python analyzer extract swing.mov
uv run --project python analyzer dataset
uv run --project python analyzer train
uv run --project python analyzer models
```

`label` opens an interactive frame-labelling window. Record the true player and
capture session consistently, and provide a known slow-motion factor when
needed. Labels carry content identity, provenance and event uncertainty;
no-swing clips are meaningful examples. Derived training features require pose
extraction for the labelled clips. A set that cannot be honestly split is refused.

Labels live under the application data directory's `labels/`; checkpoints and
model cards live under `ml/`. They are not disposable cache files. `analyzer
models` shows temporal model cards; use the desktop's model manager for pose
artifacts.

## Feature contract

[`ml/features.py`](../python/analyzer/ml/features.py) defines the feature order:
hand height, lateral position, speed, horizontal/vertical velocity, acceleration,
shoulder tilt, hip tilt, torsion and validity. Body distances use torso-length
normalization; angle channels use the documented angular scale. The torso
reference is estimated from the clip, and preprocessing is for offline analysis.

Samples are interpolated onto a configured uniform real-time grid (currently
60 Hz), with validity rules that prevent interpolation across invalid source
samples. Slow-motion factors therefore affect the feature clock. The feature
specification is hashed and accompanies datasets/checkpoints; changing feature
order, rate, normalization or interpolation changes the input contract.

Resampling introduces its own timing floor. Reporting events more precisely than
that grid or the source frames support would not establish greater accuracy.
The deterministic filter works on irregular timestamps; the learned model's
uniform feature grid is a separate, explicit step.

## Temporal architecture

[`ml/tcn.py`](../python/analyzer/ml/tcn.py) implements a residual stack of dilated
one-dimensional convolutions with GroupNorm, GELU and dropout. The configuration
in [`contracts/ml.py`](../python/analyzer/contracts/ml.py) determines width,
kernel size, layer count and receptive field. Receptive field is reported in
seconds as well as samples.

The default network is non-causal: predictions use context before and after an
instant. Feature preprocessing and normalization also use recorded-clip context;
a convolutional causal flag alone is not a validated streaming pipeline.

Outputs distinguish none, address, backswing, downswing and follow-through.
The none class covers non-swing frames instead of mislabelling them as address.
A provider interface supports evaluating temporal predictions alongside the
heuristic detector; the shipped ordinary analysis path remains heuristic.

## Splits, training and evaluation

[`ml/splits.py`](../python/analyzer/ml/splits.py) groups by **player**, which also
keeps that player's sessions together. Neighbouring swings from one player on
both sides of a split would measure recognition of seen players, not transfer
to new golfers. Leakage checks inspect the resulting assignment. A deliberately
leaky splitter exists for synthetic demonstrations, not admissible evaluation.

Training records configuration, seed, features and split provenance. Evaluation
reports classification and event-localization results with label/resampling
uncertainty and sample support. Model cards distinguish synthetic evidence from
real claims and enforce minimum support gates. Those gates are policy floors,
not a proof that a dataset is representative or statistically sufficient.

[`scripts/benchmark_ml.py`](../scripts/benchmark_ml.py) exercises this apparatus
on a synthetic corpus; its results cannot serve as golfer accuracy numbers.
There is no publishable real accuracy, F1 or event-timing score at this stage.

## Device selection

Training defaults to CPU for reproducibility. `--device auto`, `cpu`, `cuda` or
`mps` selects through the shared runtime probe. Available accelerators must pass
an allocation and matrix-operation check; failed selection falls back to CPU
with a reason. `GSA_FORCE_CPU=1` overrides an explicit accelerator request.

The selected device is recorded and used for model and input tensors. A kernel
failure during training is surfaced rather than silently restarting training on
CPU. MediaPipe's CPU delegate is independent of torch device selection, and the
macOS MediaPipe runtime still requires platform graphics services.

The [generated appendix](benchmarks.md) includes a forced-CPU execution run and
pinned inventory evidence. Its small temporal forward pass checks finite output,
not learning quality, training throughput or reproducibility across devices.

## Next evidence required

Collect diverse players and capture sessions with stable provenance, independently
label event brackets and disagreements, establish held-out players before tuning,
and compare learned and heuristic detectors on identical real inputs. Publish
sample counts, refusals and uncertainty alongside errors. A valid model card
must keep the scope of its claim tied to that evaluation.
