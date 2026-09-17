# ADR-0010: Measuring biomechanics from one uncalibrated camera

**Status:** Accepted
**Date:** 2026-09-16
**Phase:** 5

## Context

Phase 5 has to produce the numbers a coach would recognise — shoulder turn,
X-factor, spine angle, tempo — from a single phone camera with no calibration,
no second view, and no depth. Those quantities are defined in three dimensions.
The input has two.

Three responses were available.

1. **Wait for Phase 9.** Produce nothing until stereo triangulation exists. That
   leaves the whole system unable to say anything about a swing for another four
   phases, and leaves the metric contract, the registry and the confidence
   plumbing undesigned until the moment they are hardest to change.
2. **Report 3D-sounding numbers from 2D.** Compute a shoulder angle, call it
   "shoulder rotation", and let the reader assume it is the rotation of their
   shoulders. This is what most single-camera swing apps do.
3. **Measure the projection, and make the number say so.** Compute what the
   image actually determines, and carry the distinction in the type system
   rather than in a disclaimer.

Option 2 is excluded by the project's first design commitment: every displayed
number comes from a real computation, and capabilities are reported as measured.
A projected angle presented as a body angle fails that even though the
arithmetic is correct, because the label is the part that is wrong.

## Decision

**Every metric carries a `basis` saying what kind of claim it is**, as a typed
field on the contract rather than as prose:

| basis                 | what it is                                          |
| --------------------- | --------------------------------------------------- |
| `TEMPORAL`            | a duration or a ratio of durations                  |
| `IMAGE_PLANE`         | a distance or speed between two points in the frame |
| `PROJECTED_ANGLE`     | the angle between two segments as they appear       |
| `FORESHORTENED_ANGLE` | rotation inferred from how much a segment shortened |

`TEMPORAL` is the only basis in this phase that measures the body rather than a
picture of it. A consumer that wants to say "your shoulders turned 50 degrees"
has to pass a field that says the number is a foreshortening estimate.

**Rotation is measured by foreshortening, against an address baseline.** A rigid
segment of length L seen rotated by θ projects to L·cos θ, so

    θ = arccos(span / span_at_address)

recovers the turn from two lengths measured in the same image, with no camera
parameters at all — the focal length, the distance and the sensor size all
cancel in the ratio. Three consequences are accepted and recorded rather than
worked around:

- **It is blind to direction.** cos is even, so a turn and its mirror image
  shorten the line identically. Values are magnitudes.
- **It is ill-conditioned near zero.** dθ/d(span) goes as 1/sin θ, so a small
  turn is swamped by landmark noise while a large one is measured sharply. This
  is not a caveat in a document; `sin θ` _is_ the `method` confidence factor.
- **It needs the player square to the camera at address.** That premise is
  checked, not assumed — see below.

**Confidence is three measured factors, and none of them is a policy constant.**

| factor        | what it measures                                              |
| ------------- | ------------------------------------------------------------- |
| `observation` | reported visibility of the landmarks used, on the frames used |
| `anchor`      | the Phase 4 confidence of the event or phase measured at      |
| `method`      | how sharply the method itself pins the quantity down          |

`method` is computed per basis from something in the data: for a duration, the
fraction of it that is not frame-rate quantisation; for a projected angle, the
fraction of the segment lying in the image plane; for a foreshortened rotation,
the sine of the angle found. `overall` is the product, matching Phase 4.

The factors answer _how well this method determined this quantity as defined_ —
not how close the quantity is to an anatomical truth. That second question is
what `basis` is for, and mixing the two into one scalar would make it
uninterpretable: "the landmark was blurry" and "a camera cannot see rotation"
have completely different fixes.

**Lengths are in torso lengths.** Not pixels, not frame widths. A frame width
halves when the camera moves twice as far away; the player's torso does not.

## The baseline check, and why it refuses

`span_at_address` stands in for the segment's true width, which is right only if
the player was square to the camera at address. That is true of a face-on
recording and false of a down-the-line one, where the shoulders start nearly
end-on and open up through the backswing.

So the premise is tested: if the segment projects more than
`max_reference_excess` (default 1.25) times wider anywhere in the clip than it
did at address, the rotation metrics are **refused**, with the measured ratio in
the reason. Refused rather than reported with a warning, because the result
would not be a slightly-wrong version of the truth — it would be the angle away
from a pose the player never held.

On the reference footage the two cases separate by a wide margin rather than a
marginal one: 1.12× on the face-on clip (landmark noise) against 10.61× on the
down-the-line clip.

### What was tried first

The baseline was originally the **widest view anywhere in the clip**, with the
premise checked by asking whether that frame fell in the address phase. It is
worse, and the reference footage showed why on the first run: a maximum over a
clip is a maximum over that clip's landmark noise, so it selects the single
frame where the estimator most overstated the span. On `PW_face-on.mp4` that is
frame 49, just past impact, where the shoulders read 12% wider than the player
can physically be. Every rotation in the clip was then measured against an
error, and — because that frame is after the takeaway — the clip was _refused_
as not face-on when it plainly is.

Taking the baseline from where the premise says it should be, and checking the
premise separately, fixes both.

## Consequences

- Single-camera metrics exist now, and are labelled in a way that survives being
  passed to a coaching layer or a language model in Phase 13.
- The `Metric`, registry and confidence machinery is built and exercised before
  Phase 9 has to extend it, rather than being retrofitted around triangulated
  data.
- Down-the-line clips get tilts, posture, hands and timing, and no rotation.
  That is a real capability loss and it is the correct one: that view does not
  contain the measurement.
- When Phase 8 and Phase 9 land, the foreshortening estimators are replaced
  rather than corrected, and `basis` gains a value meaning "triangulated". The
  refusal logic here becomes unnecessary rather than wrong.

## Alternatives rejected

**A fixed confidence penalty for projected quantities** (say, cap every
uncalibrated angle at 0.6). Rejected because it is a made-up number that makes
the result less interpretable, not more honest — and because a measured
conditioning factor was available for every basis. Nothing in the confidence
model is a constant someone chose.

**Reporting shoulder turn with a sign inferred from the z channel.** IMAGE-space
z is MediaPipe's single-camera depth estimate on an unstated scale. Using it to
sign a rotation would put a guessed quantity inside a measured one with no way
to tell afterwards which part was which.

**Labelling arms "lead" and "trail" by assuming right-handedness.** Rejected in
favour of measuring it: at the top of the backswing the hands sit over the trail
shoulder, which is visible, and projecting the hands' offset onto the shoulder
line names the side. Where the view does not show it — down the line, where the
shoulder line points at the camera — no side is named and those metrics are
refused.
