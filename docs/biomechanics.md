# Biomechanics and coaching

Metrics describe what can be measured from this recording. They do not assign
swing quality. The authoritative catalogue is
[`biomechanics/registry.py`](../python/analyzer/biomechanics/registry.py), with
contracts in [`contracts/metrics.py`](../python/analyzer/contracts/metrics.py).

## Preconditions and reference frames

A pose extraction, filtered motion, usable swing anchors and tracked body scale
are required. A missing swing or torso produces named refusals. Image-based
calculations use `FRAME_WIDTHS`: x and y share the frame-width scale and y points
up. Distances are normalized by torso length where stated. This removes image
scale, not perspective or out-of-plane motion.

Camera view is measured from shoulder width relative to torso length at address.
A broadside line supports face-on interpretation; a collapsed line supports
down-the-line interpretation. Ambiguous cases remain `UNKNOWN`. The session's
camera role is a capture declaration, not this measurement. The view detector's
anatomical assumptions have not been validated on a labelled population.

See [Coordinate systems](coordinate-systems.md) for handedness and origins.
Hip-local model coordinates are not a substitute for calibrated spatial input.

## Implemented metric catalogue

Names below are contract identifiers, grouped by method. Individual outputs also
state their anchor or interval, units, camera interpretation and any refusal.

| Family             | Identifiers                                                                                 | Units and meaning                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Posture            | `spine_tilt`, `left_knee_flex`, `right_knee_flex`                                           | Degrees in the image plane; spine interpretation depends on view  |
| Translation        | `hip_sway`, `head_sway`, `head_lift`                                                        | Torso lengths relative to address; image directions               |
| Rotation           | `shoulder_turn`, `pelvis_turn`, `x_factor`                                                  | Degrees inferred from foreshortening; unsigned and view dependent |
| Body-line tilt     | `shoulder_tilt`, `pelvis_tilt`                                                              | Projected line angles in degrees                                  |
| Hand motion        | `hand_depth`, `hand_path_length`, `peak_hand_speed`                                         | Torso lengths, or torso lengths per real second                   |
| Arms               | `lead_arm_angle`, `trail_arm_angle`                                                         | Projected joint angles in degrees                                 |
| Timing             | `backswing_duration`, `downswing_duration`, `follow_through_duration`, `takeaway_to_impact` | Real seconds between detected events                              |
| Tempo              | `tempo_ratio`                                                                               | Backswing duration divided by downswing duration                  |
| Spatial rotation   | `shoulder_turn_3d`, `pelvis_turn_3d`, `x_factor_3d`                                         | Degrees about the measured address spine axis                     |
| Spatial arms       | `lead_arm_angle_3d`, `trail_arm_angle_3d`                                                   | Joint angles between reconstructed vectors, in degrees            |
| Spatial hand speed | `peak_hand_speed_3d`                                                                        | Metres per real second from calibrated reconstruction             |

The formulas and view-specific interpretations are part of the registry rather
than duplicated in the UI. Configuration thresholds are structural validity
policies; they are not population golf norms.

## What a projected measurement means

A spine angle relative to image vertical reads as lateral bend in a face-on view
and forward posture in a down-the-line view, assuming the corresponding camera
placement. A tilted camera changes that reference. Knee and elbow angles are
angles between projected segments; depth motion can change their anatomical
values without changing their image values.

Shoulder and pelvis turn use the change in apparent line span from address.
Foreshortening cannot distinguish a turn from its mirror image, and an oblique
address underestimates the true reference span. These turns and projected
X-factor refuse a down-the-line view. Projected X-factor is a difference of
unsigned turn estimates, not a signed 3D thorax/pelvis measurement.

Hand depth is interpreted only from the supported camera view. Path length and
peak speed are image-plane quantities normalized by torso length; they are not
club-head speed and cannot be converted to metres without spatial geometry.
An unknown lead side or insufficient support can independently refuse arm metrics.

## Spatial measurements

`metrics VIDEO --project ID` can add spatial metrics when the project's aligned
pair and calibration produce usable reconstruction. Their `basis` identifies
calibrated 3D. Rotations reference the address spine axis, not an assumed gravity
axis or a motion-capture anatomical convention. Joint angles and speed are
invariant to a rigid camera-frame change.

There is no gravity-aligned `WORLD` frame and no calibrated gravity-relative 3D
spine tilt. Small reprojection residual does not bound depth error. Spatial
arithmetic is checked on synthetic ground truth in the
[generated metric appendix](benchmarks.md#synthetic-spatial-metric-error); no real
stereo dataset establishes its golfer accuracy.

## Confidence, uncertainty and refusals

Each metric retains source frames, methodology, basis, units and camera view.
Its confidence combines observation, anchor and method factors multiplicatively,
so good tracking cannot hide an unusable anchor. These factors are engineering
quality indicators, not probabilities calibrated against labelled truth.

An uncertainty value or timing bracket covers only the mechanism named by its
method. Pose bias, blur, perspective and uncertain event detection can contribute
additional error. In particular, a low polynomial fitting residual can be zero
on minimally supported data without the underlying landmarks being accurate.

Refusals remain in the result with reasons. Missing information must not be
presented as a zero angle, no movement or an ideal swing. The desktop lets the
reader seek to a measurement's source frames for inspection.

## Coaching rules and evidence

[`coaching/registry.py`](../python/analyzer/coaching/registry.py) declares the
shipped rules, thresholds, provenance, population and allowed measurement basis.
Timing rules can operate on supported clips. Most posture, rotation and arm rules
refuse: a borrowed 3D threshold cannot be applied to a projected quantity, and a
quantity without an applicable published threshold is not assigned one.

Even the calibrated X-factor rule remains limited by measurement-method mismatch:
rotation about the address spine axis is not automatically the anatomical method
used to establish a published norm. See
[ADR-0017](decisions/ADR-0017-borrowed-thresholds-and-the-guard.md) for the rule
catalogue's reasoning. This document does not independently endorse its sources
as universal coaching targets.

Findings cite their computed evidence, frames and comparison bracket. A value
near a threshold is not declared beyond it unless the supported difference clears
that bracket. A rejected finding is recorded as a refusal. The system makes no
injury prediction or diagnosis.

The optional phrasing provider receives structured findings and proposes wording.
The deterministic guard rejects numbers not present in the evidence and keeps
fallback text when the proposal fails. This is a numeric consistency check, not
a general semantic truth detector. The local provider checks a loopback endpoint;
no real model quality study has been performed. Phrasing is off by default.

## Comparing recordings

`comparison/` builds a common phase-relative clock anchored at takeaway, top,
impact and finish. Before drawing a projected channel it checks camera views and
address geometry; a refused channel has no plotted samples. Camera mismatch can
refuse planar comparisons while leaving timing comparisons available.

Uncertainty bounds from separate recordings are added because their systematic
errors need not cancel. Trajectory brackets account for signal variation over
the timing ambiguity interval, including interpolation support. A difference
inside the bracket is unresolved; this does not prove equivalence.

The bounds do not fully model per-sample landmark error. Noise can therefore
produce resolved differences even when the underlying swing is unchanged.
Body-shape differences can also trigger camera-span gates. Comparison is most
interpretable for the same player under repeatable capture conditions, and it
still has no overall score or correctness claim.

## Verification and next evidence

Use `analyzer extract`, then `analyzer metrics … --json` or `analyzer coach …
--json` to inspect the complete contract. Numerical invariants cover geometry,
unit handling and known synthetic trajectories; the fixture catalogue is in
[Testing](testing.md). Overlay inspection on local recordings checks visible
agreement but supplies no labelled error distribution.

Validation still needs independent landmark labels, event labels with inter-rater
ambiguity, player-held-out evaluation, real calibration/triangulation truth and
method-matched coaching norms. Those gaps remain after documentation is complete.
