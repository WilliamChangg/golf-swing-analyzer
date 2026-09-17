# Coordinate systems

Every number this system reports is a measurement taken in some reference frame,
and most of the ways to get one wrong produce a plausible number rather than an
error. This document is the single statement of which frames exist, what their
conventions are, which are reachable today, and what each one may be used to
claim.

The code is `python/analyzer/coordinates.py`; the enum is
`LandmarkSpace` in `python/analyzer/contracts/pose.py`.

## The frames

| Frame          | Units              | y      | Origin         | Isotropic | Metric | Status              |
| -------------- | ------------------ | ------ | -------------- | --------- | ------ | ------------------- |
| `IMAGE`        | x/W, y/H           | down   | top-left       | **no**    | no     | stored              |
| `FRAME_WIDTHS` | x/W, (H−y_px)/W    | **up** | bottom-left    | yes       | no     | derived on read     |
| `HIP_LOCAL`    | approximate metres | up     | hip midpoint   | yes       | approx | stored              |
| `CAMERA`       | metres             | down   | camera centre  | yes       | **yes** | **triangulated**   |
| `WORLD`        | metres             | —      | fixed to scene | yes       | yes    | **absent**          |

`W` and `H` are the **displayed** frame dimensions, after rotation. A phone clip
stores frames sideways with a display matrix; the coded and displayed dimensions
differ by a transpose, and using the coded pair would invert the aspect
correction rather than merely scale it.

## IMAGE

What a pose estimator emits and what the Parquet store holds. x is divided by
the frame width and y by the frame height, both landing in [0, 1], with y
increasing towards the bottom of the frame.

It is the only frame a landmark can be drawn in without conversion, and it is a
bad frame to measure in, for two reasons that both fail silently.

**It is anisotropic.** On a 1080×1920 clip, one pixel across is 1/1080 and one
pixel down is 1/1920 — the same displacement measures 0.5625 times as much
vertically. Any Euclidean quantity mixing the two is wrong by a factor that
depends only on the shape of the recording. A line genuinely at 45° reads as
**29.4°**.

**Its y points down.** Every rule in a swing analysis is about something being
high or low. A missing flip inverts the top of the backswing into the bottom.

## FRAME_WIDTHS

Both axes divided by the frame **width**, y increasing upward, origin at the
bottom-left. This is the frame every measurement in the system is taken in.

```
x' = x
y' = (1 − y) · (H / W)
z' = z
```

A displacement of _n_ pixels measures the same whichever way it points, so
distances, speeds and angles all mean what they look like. 1.0 is the width of
the frame.

**The conversion happens once, below the filter** (`pose/series.py`), and that
placement is the point rather than an implementation detail:

- A consumer cannot forget it. Before Phase 6 the correction lived in the
  biomechanics layer, which meant phase detection — sitting below it — measured
  hand travel and torso length in the uncorrected frame.
- The derivatives come out right for free. The filter is linear, so a sign flip
  and a scale applied to positions before fitting emerge correctly in the
  velocity and the acceleration. Converting after the fit means correcting the
  derivative separately and keeping that in step with the position conversion by
  hand, which is a second thing to get wrong.

### What it is not

**Not metric.** It is a picture measured in units of its own width. The same
swing filmed from twice the distance gives half the numbers. Anything that must
survive a change of camera position divides by something the subject brings with
them — the biomechanics layer uses the torso length — rather than by a frame
width.

## HIP_LOCAL

MediaPipe's "world landmarks": approximate metres, centred on the hip midpoint,
oriented to the body. Stored alongside IMAGE.

**They are not calibrated world coordinates.** They carry no information about
where the camera was, how far away the subject stood, or which way the target
line ran, and no metric-scale claim rests on them. Both the phase detector and
the biomechanics layer refuse them, for a reason beyond the missing calibration:
the origin moves with the body, so a hip displacement measured in them is zero
by construction and a head height is measured from a moving datum.

## CAMERA

**Produced, from Phase 9 on, and never by a conversion.**

Metres in three dimensions, centred on the reference camera, from triangulating
two calibrated views of the same instant (`analyzer/reconstruction`). It is the
only frame in this table whose unit is a real metre, and the scale descends from
one number: the printed board square measured with a ruler in Phase 8.

It is still listed in `UNREACHABLE_SPACES`, and that is the distinction worth
keeping. `landmark_series` refuses it, because a stored pose sequence is **one
clip**, which is one projection — asking it for CAMERA would mean inventing the
depth the projection destroyed. The frame is a measurement made from two
sequences, not a reading of one, so it does not live behind `convert`.

## WORLD

**Named here, and not produced — and the obstacle is not arithmetic.**

A scene-fixed frame needs two directions that a calibrated stereo pair does not
supply:

- **which way is up.** The cameras do not know their own attitude. Nothing in a
  reconstruction distinguishes a level floor from a sloped one.
- **which way the target line runs.** A body does not declare one, and Phase 6
  already established that a down-the-line recording cannot even tell which *end*
  of the target line the camera stood at.

Both fall out of a capture that lays the calibration board flat on the ground in
the hitting area with one edge along the target line: the board's plane gives the
ground, its normal gives up, and its own axes give the line. That is a change to
`data/README.md` and a later phase, not a missing function — and until it
happens, naming the frame and refusing it beats rotating into axes that were
assumed.

This costs something real and it is worth being specific about what. A **3D spine
tilt** — the forward posture angle a coach actually talks about — needs a
vertical, so it is not in the spatial metric family. What does not need one is
anything measured against the body's own axes or between two reconstructed
points: the 3D rotations are taken about the measured address spine axis, joint
angles and lengths are invariant to the frame entirely, and speeds in metres per
second are too.

Both are in the enum deliberately. A later phase should add a capability, not a
concept — and asking for either raises an error naming what supplies it rather
than reporting an unsupported value.

## Conversions

Implemented in `analyzer/coordinates.py`, all round-trip exactly:

```
image_to_frame_widths   ↔  frame_widths_to_image
frame_widths_to_pixels      (for drawing)
image_to_pixels             (for drawing)
```

`convert(points, source, target, geometry)` dispatches and refuses everything
else. In particular **there is no conversion between IMAGE and HIP_LOCAL in
either direction**, and none into CAMERA: recovering either from a picture means
recovering the depth the picture lost, which is exactly what a single camera
cannot do. CAMERA arrives from `analyzer/reconstruction`, which is a measurement
taken from two clips rather than a function of one.

The tests in `python/tests/test_coordinates.py` check three things: facts rather
than tolerances (equal pixel displacements must measure equal; a true 45° line
must read 45°), round trips (which catch a correction applied twice or in the
wrong order where a one-way check cannot), and the refusals.

## Camera views

Distinct from the reference frame, and the thing that decides what a projected
measurement _means_. Same arithmetic, different anatomy.

| View            | Shoulder line at address | Rotation measurable | Frame's x axis runs   |
| --------------- | ------------------------ | ------------------- | --------------------- |
| `FACE_ON`       | broadside, wide          | yes                 | along the target line |
| `DOWN_THE_LINE` | end-on, collapsed        | no                  | towards the ball      |
| `UNKNOWN`       | between the two          | decided per metric  | undetermined          |

`DOWN_THE_LINE` means _along_ the target line and does not say which end. A
camera behind the player and one in front foreshorten the shoulder line
identically, and nothing here separates them; the measurable consequences are
the same either way. What it costs is the sign of anything measured along the
frame's horizontal axis, so those are reported as image directions rather than
as "towards the player". The reference clip `data/dtl/iron_dtl.mp4` is filmed
from in front, despite its name.

The view is **measured**, from the projected width of the shoulder line at
address in torso lengths. On the reference clips those are **0.83** (face-on)
and **0.10** (down-the-line) — a factor of eight, which is a verdict rather than
a close call. Thresholds are 0.55 and 0.30, with the band between them reported
as `UNKNOWN` rather than rounded to the nearer label.

Every `Metric` carries the view it was measured in and an `interpretation`
stating what it corresponds to on the body from there. Spine tilt is the worked
example: identical computation, **+4.6°** of lateral side bend on the face-on
clip and **+35.5°** of forward posture angle on the down-the-line one.

Metrics a view cannot support are refused rather than relabelled. Shoulder turn,
pelvis turn and X-factor are refused down the line, because the shoulder line
points at the camera and its rotation does not appear in the picture. Hand depth
is refused face-on, because the axis it measures along runs along the target
line there and is a different quantity under the same name.
