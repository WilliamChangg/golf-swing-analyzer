# ADR-0020: A comparison is between two recordings, so the camera is gated before the swing is

**Status:** Accepted
**Date:** 2026-09-18
**Phase:** 16 — Swing comparison

## Context

Every layer below this one measures **one** recording. Phase 16 is the first
whose output is a statement about a pair, and the whole difficulty is contained
in one sentence: two recordings differ for reasons that have nothing to do with
the two swings.

There are four such reasons and they are not equally obvious. A slow-motion
factor is supplied rather than measured, so a duration from such a clip is a
guess multiplied by a measurement — Phase 13 already gates on that. A lens
correction applied to one clip and not the other displaces every landmark by tens
of pixels near the frame edge — Phase 8 already measured that. Each clip's own
frame rate bounds what it can resolve — Phase 13's `bracket` already computes
that.

The fourth is the one this phase had to find, and it is much larger than any of
them.

### The measurement

`scripts/benchmark_compare.py --sweep camera`, Apple M1 Pro / macOS 26.4.1. One
**unchanged** synthetic swing — the same `body_at` function, the same instants,
the same joint angles — projected through cameras that differ only in where they
stand. Landmark scatter is the 2.7 px Phase 3 measured on real footage; median of
five seeds:

| azimuth | address span | span disagreement | **shoulder turn reported** | comparison's verdict |
| ------- | ------------ | ----------------- | -------------------------- | -------------------- |
| 0°      | 1.05 torso   | 0%                | **57.7°**                  | unresolved           |
| 2°      | 1.05         | 1%                | 57.6°                      | unresolved           |
| 5°      | 1.04         | 1%                | 56.9°                      | unresolved           |
| 10°     | 1.02         | 3%                | 55.1°                      | unresolved           |
| 15°     | 0.99         | 6%                | 52.1°                      | unresolved           |
| 20°     | 0.95         | 11%               | 47.7°                      | **camera_moved**     |
| 30°     | 0.83         | 24%               | **32.9°**                  | **camera_moved**     |

**The same turn reads as 57.7° and as 32.9°.** Nothing about the body changed
between the first row and the last. A 25-degree difference in the single most
quoted number in golf instruction, produced entirely by moving a tripod through
an arc a person setting up a phone would not think to measure.

The arithmetic behind it is not subtle once stated. A foreshortening rotation
compares a body line's projected span against its span at address; from a camera
`a` degrees off broadside, a line that has truly turned `t` projects `cos(a + t)`
against an address span of `cos(a)`, so the reported angle is

    arccos( cos(a + t) / cos(a) )

which equals `t` only when `a` is zero, and departs from it fast. Fifty degrees
seen from ten degrees off reads as fifty-six; from twenty degrees, sixty-five.

### Why the view label cannot be the gate

Phase 6 measures a `CameraView` — `face_on`, `down_the_line`, `unknown` — from
the address shoulder span, and it is exactly right for what it does: it decides
whether a recording _contains_ a measurement at all. It is far too coarse for
this. Every row of that table above is `face_on`. A camera can move thirty
degrees round a player and stay comfortably inside one label, and by then it has
moved the reported shoulder turn by nearly half.

### Why the derived azimuth cannot be the gate either

`ViewEstimate.openness` is the address span divided by the widest that line was
ever seen in the clip, which is the cosine of how far off broadside the shoulders
were at address. Its arccosine is an estimate of the camera's azimuth, and it is
tempting to gate on that directly.

It does not survive contact. The arccosine is flat near broadside, so an openness
of 0.99 — one per cent, which is landmark noise — comes out as 8.1 degrees of
azimuth. Phase 5 already recorded the reference footage reading 12% wider than
the player can physically be at the widest frame; that is 28 degrees. A gate on
the derived angle would refuse every pair ever filmed, which is not carefulness,
it is uselessness.

## Decision

**Gate on the measurement and bracket with the derivation. Refuse a projected
comparison when the two clips' address shoulder spans disagree by more than 10%;
below that, add what the residual azimuth alone would do to the reported value as
a named term in the bracket.**

Four parts.

**1. The span is the verdict.** `CameraAgreement.span_disagreement` is the
absolute difference of the two clips' address shoulder spans over their mean. It
is a directly measured quantity in a well-conditioned unit — torso lengths — with
no arccosine between it and the data. Above `max_span_disagreement` every
`IMAGE_PLANE`, `PROJECTED_ANGLE` and `FORESHORTENED_ANGLE` comparison is refused
as `CAMERA_MOVED`, and the refusal prints both spans so the reader can see how
far apart they were.

**2. The azimuth is a bracket term, computed exactly.** For a foreshortened
rotation, `camera_term_deg` recovers the true turn from the reference clip's own
azimuth, re-reads it from the target's, and returns the difference. Both azimuths
are unsigned — foreshortening is identical from either side of broadside and
nothing here separates them — so the worst case over the sign assignments is
taken. **Two equal azimuths therefore do not cancel**, and that is deliberate:
two cameras each five degrees off may have stood in one place or ten degrees
apart, and a term that assumed the first would be assuming the answer.

**3. Brackets are summed across clips, not combined in quadrature.**
`coaching.combined_bracket` uses quadrature and justifies it by the two
measurements sharing a systematic error — a foreshortening baseline taken at
address is the same baseline at both anchors of one clip, and a shared error
largely cancels in a difference. **Two separate recordings share nothing.** A sum
of bounds is a bound; a quadrature of bounds is not. Same shape of question, two
clips apart, opposite answer.

**4. The gate that is a fact about the pair runs before the gate that is a fact
about one clip.** `VIEW_MISMATCH`, `CAMERA_MOVED`, `CALIBRATION_MISMATCH` and
`SUPPLIED_TIMEBASE` are all evaluated before `MISSING`, `LOW_CONFIDENCE`,
`NO_BRACKET` and `UNRESOLVED`. The ordering is the same lesson Phase 13 learned
about `BASIS_NOT_PERMITTED` sitting above `NO_METRIC`: a reader told "one clip
did not measure this" goes and re-films it, and if the cameras were in different
places the re-filmed clip is refused for the camera instead. They have shot
footage to learn what the first answer could have told them.

### And there is no score

`MetricDifference` carries a direction — higher or lower — and nothing else. No
severity, no rank, no total. A number summarising how far apart two swings are
would be the single most quoted output of this system and the least defensible
one, for two separate reasons: it would need a scale relating degrees of shoulder
turn to seconds of tempo, which nobody has measured, and it would need to know
which direction of each quantity is _desirable_, which Phase 13 established no
source supplies. Every rotation threshold in that registry turned out to be a
convention, a population summary, or a number with no published protocol. None of
them is a target.

`Direction` has two members rather than three, and that is the same argument once
more. There is no "the same": two values closer together than the pair can
resolve produce `UNRESOLVED`, which says the recordings could not tell them apart
rather than that the swings agreed. The first would be a measurement. The second
is the absence of one, and they belong in different lists.

## Consequences

**The reference footage in this repository supports almost no comparison, and
that is the correct answer.** `--sweep clips` over the three pairs that can be
built: two are a face-on camera against a down-the-line one, so every projected
quantity refuses and four timing differences survive. The third is two swings by
one player from one position — and it refuses too, because the second clip begins
at the takeaway, a clip with no address phase has no measured view, and a
comparison that cannot say where either camera stood will not compare a
projection.

That last one is a capture instruction rather than a defect, and it is the most
useful thing this phase produced for anybody holding a camera: **start recording
before the player is set up to the ball.** The address phase is where the view,
the rotation baseline and the hand-path origin all come from.

**The span gate assumes both clips show one body.** A broader-shouldered player
projects a broader line from the same place, so two different golfers filmed from
one tripod can fail the gate. This is stated rather than fixed: nothing in either
recording separates a camera that moved from a player built differently, and both
explain a difference the swing did not make. Refusing is the honest response to
an ambiguity, not a limitation to be engineered around.

**The trajectory bracket covers _when_ a sample was taken, not _how well_ the
value at it was measured.** It is the event ambiguity carried onto the normalised
axis and through the signal's own local range — which is exact for what it
describes, and describes only the clock. The azimuth-0 row of the camera sweep
measures what that leaves: **5% of sampled positions report a resolved difference
on a swing that did not change**, from landmark noise alone at two different
seeds. Quantifying that would need a per-sample landmark uncertainty, and Phase 4
already recorded why the filter's residual cannot supply one — it is exactly zero
whenever the smoothing window holds as many samples as the polynomial has
coefficients, which is every clip below about 60 fps at the shipped defaults.

**Everything above the camera gate is still floor-level.** The 3D fixture's body
is an input, there is no pose estimator in it, and the noise is an explicit
displacement rather than an estimator's structured error. A real estimator loses
the shoulders to motion blur exactly where this fixture is perfect, so the
degrees in the table are the least this effect can be worth, not the most.
