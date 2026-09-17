# ADR-0014: The strongest line in the frame is not the club

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 10 — Club tracking

## Context

A golf shaft is the first thing this engine measures that is not a body. Every
layer up to here reads landmarks a model was trained to find, which arrive with
the model's own opinion of how well it saw them. A shaft has no model, no
landmark index and no reported visibility. It has edges.

The classical pipeline for finding it is not in doubt and is what Phase 10.2
asks for: restrict to a region around the hands, run Canny, run a probabilistic
Hough transform, filter the segments geometrically. The question this ADR records
an answer to is the one after that — **what the system should report about how
much it believes the line it found**, and what it should refuse.

The obvious answer is the one the transform hands over. `HoughLinesP` scores a
line by how many edge pixels voted for it; a detector can report that, or the
fraction of the segment that lies on an edge, and call it confidence. It has the
right shape. It is high when things are going well.

Phases 8 and 9 had each already found that the analogous number one and two
layers down is blind to the failure that matters, and that the fix is to gate on
something about the **capture** rather than about the fit — coverage of the frame
for a calibration, ray convergence angle for a reconstruction. Phase 10 asked the
same question again and got the same shape and two different facts.

## Decision

**Three numbers are reported and each is labelled with the question it answers.
Support is a floor, not a ranking. The gate is per-phase coverage.**

```
support        how much edge evidence backs this line        the fit
margin         whether anything else in frame fits as well   a check
phase coverage whether the frames that matter have any       the gate
```

Concretely:

- A candidate is chosen on `support x continuity`, never on support alone.
- `margin` is measured on that same combined score, not on evidence alone.
- A frame emits nothing when the product of the three falls below
  `ClubConfig.min_confidence`, and the refusal is named.
- `ClubTrackingReport.coverage` is reported and documented as the number **not**
  to read on its own; `phase_coverage` is what the verdict rests on, and the
  club-head impact estimate is refused outright when downswing coverage is below
  half.

## Why: support ranks a door frame above a golf club

A vertical background line passing through the hands — a door frame, a fence
post, a window mullion — satisfies every geometric test a shaft does. It begins
near the grip, it is the right length, and it is _straight and stationary_, which
makes it a better line than a club by every measure a Hough transform has.
Measured on a single rendered frame with a sharp club at 45° and a door frame
through the grip:

| candidate  | support  | length | rank on evidence |
| ---------- | -------- | ------ | ---------------- |
| door frame | **1.00** | 670 px | **first**        |
| the club   | 1.00     | 275 px | second           |

It is worse than a tie, and it gets worse in the direction that matters. The club
is the fastest thing in the frame through the downswing and therefore the
blurriest; the door frame is stationary and stays perfectly sharp. So evidence
ranks the background _above_ the club exactly where the club matters most, and a
detector confident in proportion to support would be most confident when it was
most wrong:

| club state              | candidates found        |
| ----------------------- | ----------------------- |
| sharp                   | door frame **and** club |
| blurred (14 px smear)   | door frame only         |
| heavily blurred (38 px) | door frame only         |

`margin` is what notices, and it has to be measured on the combined score for it
to notice the right thing. Scored on evidence alone, the frame where a smeared
club sits at 0.6 against a door frame at 1.0 reads as ambiguous and a **correct**
detection is refused. Scored on `support x continuity`, the door frame scores
almost nothing — it is not rotating — the margin is high, and the club is kept.

`scripts/benchmark_club.py --sweep clutter` measures what that buys against
picking the strongest line, which is what a per-frame detector alone would do:

| background lines | tracker: kept / wrong | evidence only: kept / wrong |
| ---------------- | --------------------- | --------------------------- |
| 0                | 289 / **0**           | 296 / **0**                 |
| 1                | 285 / **0**           | 294 / **4**                 |
| 2                | 285 / **0**           | 294 / **7**                 |
| 3                | 277 / **0**           | 297 / **15**                |
| 4                | 279 / **0**           | 298 / **17**                |

The tracker keeps fewer frames and gets them all right. The frames it gives up are
the ones a per-frame detector gets wrong, which is the whole of the difference:
a stationary line scores almost nothing on continuity however good its edges are.

## Why: the aggregate detection rate hides the downswing

The second fact is not that a number is blind. It is that the failure is
**structured**, and structured in a direction that makes the obvious summary
flattering.

Detection is easiest where the club is slowest — at address, where it is
stationary, and near the top, where it pauses. It is hardest through the
downswing, where it moves fastest. Address and the follow-through are most of a
clip. So the clip-wide rate is an average dominated by the frames nobody wants to
measure, and it can be high while the downswing contains nothing at all.

`scripts/benchmark_club.py --sweep shutter`, at 120 fps, sweeping the exposure as
a fraction of the frame interval:

| shutter | max smear | **overall** | address | backswing | **DOWNSWING** | follow |
| ------- | --------- | ----------- | ------- | --------- | ------------- | ------ |
| 0.03    | 0.8 px    | 100%        | 100%    | 99%       | **100%**      | 100%   |
| 0.125   | 3.5 px    | 99%         | 100%    | 99%       | **100%**      | 98%    |
| 0.25    | 6.9 px    | 93%         | 100%    | 98%       | **83%**       | 73%    |
| 0.5     | 13.9 px   | 86%         | 100%    | 93%       | **65%**       | 62%    |
| 1.0     | 27.7 px   | 64%         | 100%    | 58%       | **25%**       | 23%    |

Read the last row. A tracker reporting "64% coverage" has tracked a quarter of
the downswing, and the address column is 100% in every row — it carries no
information about the capture at all.

The consequence beyond reporting is that the club-head impact estimate is gated
on downswing coverage rather than on anything local to itself. Without that gate,
a clip whose club head blurred away through impact reports the lowest point of
whatever survived, which the synthetic swing put **38 frames late** with both of
the local checks — interiority in the observed set, and adjacency of the frames
either side of the minimum — passing.

## Why: the blur limit is a capture bound with no processing fix

The third measurement is the one that makes the first two actionable.

A smeared shaft does not become a weaker line. It stops being a line. An exposure
integrates the scene, so a rotating club is drawn as a fan whose density falls
off with the sweep; there is no edge for Canny to find and nothing for the
transform to vote on. `--sweep blur`:

| smear at the head | found    | support | angle error |
| ----------------- | -------- | ------- | ----------- |
| 0.0 px            | **100%** | 1.00    | 0.44°       |
| 4.8 px            | **100%** | 1.00    | 0.88°       |
| 9.6 px            | **100%** | 1.00    | 0.90°       |
| 14.4 px           | **0%**   | —       | —           |
| 28.8 px           | **0%**   | —       | —           |

The cliff sits between 9 and 14 px and there is no grey zone either side of it:
where the club is found at all, the angle is right to about half a degree, and
where it is not, lowering a threshold recovers nothing. **That is why there is no
sensitivity setting in `ClubConfig` that trades detection rate against accuracy.**
There is nothing to trade.

It also turns one line of `data/README.md` from advice into a measurement.
Smear is `club-head image speed x exposure`, and the exposure cannot outlast the
frame interval — so `ClubTrackingQuality.max_blur_px` reports the smear at a
360° shutter, which is the worst any camera does, and a shutter _n_ times faster
divides it by _n_. Nothing in a video file records the exposure, so that upper
bound is the most that can be said from the file alone, and it is enough to
explain a refused downswing.

## What this does not catch, and says so

Continuity rejects a _stationary_ rival. It cannot reject one the tracker has
already stepped onto, because a stationary line agrees perfectly with a
prediction extrapolated from two frames that are already on it — and once the
real club is hidden, there is no rival to bring the margin down either.

`--sweep occlusion` measures exactly that case, with a rectangular occluder whose
boundary runs near the hands at the top of the backswing:

| occluder    | tracked | downswing | wrong | **conf. when wrong** | head seen |
| ----------- | ------- | --------- | ----- | -------------------- | --------- |
| nothing     | 93%     | 79%       | 0     | —                    | 97%       |
| upper third | 54%     | 44%       | 18    | **0.98**             | 8%        |
| upper half  | 47%     | 31%       | 4     | **0.81**             | 97%       |
| a wide band | 64%     | 38%       | 0     | —                    | 100%      |

Read the confidence column. Those eighteen frames score 0.98 against the 0.99 a
correct frame carries — all three factors are satisfied, and the answer is the
occluder's edge. **Nothing in this design notices**, and no threshold in
`ClubConfig` separates them.

This is recorded rather than fixed, in the way Phase 7 recorded that Phase 4's
event confidence does not detect a badly located event. Two things would help and
neither belongs here. A rule that the shaft must rotate during the backswing
would catch it and is a golf norm, which `PhaseConfig` and `MetricConfig` both
refuse to encode without a labelled set. And a detector that has _seen_ a blurred
or half-occluded club can find one where a straight-line transform provably
cannot, which is Phase 12's learned detector rather than a threshold.

**It is not hypothetical, and the real footage found it on the first run.** On
`rory_face_on.mp4`, nine of 139 tracked frames follow the vertical **edge of the
yardage sign** behind the player instead of the club, at a median confidence of
0.98. The sign's edge is long, sharp, stationary and passes through the hands at
the finish — precisely where the club has gone behind the player's head and there
is no rival left to bring the margin down. The synthetic sweep above predicted
the shape of that failure and the number it would be reported at.

The fixture's occluder is also harder than a person: a rectangle has a perfectly
straight, frame-spanning boundary and a torso does not. That makes the row a
worst case rather than a prediction — and it is a worst case that a bag, a mat
edge or a shadow line reproduces exactly.

## Consequences

**Most of what this phase produces on real footage is refusals**, and the design
treats that as the normal outcome rather than the failure case. `ClubFrame`
carries an observation or a named reason there is none, `scripts/overlay_club.py`
draws the refused frames stamped with the reason, and the per-frame result is
kept for every frame so a gap is visible rather than inferred.

**The detector is never told what the tracker believes.** A search narrowed by
the current belief finds what it expects, confirms it, and narrows further, which
is how a classical tracker locks onto a door frame and stays locked. Keeping the
detector stateless costs a little work per frame and means a lost tracker's
candidate list still contains the club.

**Both ends of a line through the hands are offered as candidates.** A door frame
and a club pointing the other way are the same pixels, so taking the farther
endpoint silently picks one — and measured, it picks wrong for runs of several
frames and reports it at a confidence of 1.00. Offering both puts them against
each other: with no prediction the margin collapses and the frame is refused,
which is the honest answer, and with a tracked neighbour the prediction separates
them.

## Alternatives considered

**Report the Hough vote count as the confidence.** Rejected on the clutter table
above: it is the quantity that ranks a door frame first, and it ranks it first
most decisively in the frames where the club is hardest to see.

**Gate on the clip-wide detection rate.** Rejected on the shutter table. The
address column is 100% in every row, and address is most of a clip.

**Widen `maxLineGap` to bridge the break a blurred club head leaves.** Rejected:
it also bridges the gap between a shaft and a fence post behind it, merging two
objects into one long confident line. The break is real information and the
support score is where it belongs.

**Let the prediction supply a direction where no candidate passes.** Rejected for
the reason Phase 3 refuses to interpolate across a long gap: a value produced
where there is no observation is an invention however smooth it looks, and the
result would be a continuous track through a club nobody could see.

**Fit the whole swing's shaft angle as a smooth function of time and read it off.**
This would produce a value everywhere and is the same objection one step further
along — and worse, because it would look like a measurement of the downswing
specifically. Phase 12 may learn a shaft detector, which is a different thing: a
model that has seen blurred clubs can find one, where a straight-line transform
provably cannot.

**Undistort the frame before searching it.** Not rejected, not done. A straight
club in the world is a _curved_ line in a distorted image, so the straight-line
model the transform rests on is itself violated near the frame edge — and the
correction is a remap per frame rather than a transform per landmark. What is
ruled out is the half-measure: undistorting the hand anchor while searching the
raw frame would point the search at a place in the image where the hands are not.
`dispatch._track_club` records that.

## References

- `analyzer/contracts/club.py` — the argument, as a module docstring
- `analyzer/club/hough.py` — the four stages, and what `support` measures
- `analyzer/club/track.py` — seeding, growth, and the three factors
- `scripts/benchmark_club.py` — `--sweep blur`, `--sweep shutter`, `--sweep clutter`
- `python/tests/test_club_detector.py` — the door frame outranking the club, as a test
- `python/tests/test_club_track.py` — the refusals, as tests
- [ADR-0012](ADR-0012-calibration-coverage.md) — gate on the capture, two layers down
- [ADR-0013](ADR-0013-epipolar-blindness.md) — the same shape, one layer down
