# ADR-0015: The ball is measured by its absence, and impact is not averaged

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 11 — Ball detection

## Context

By Phase 10 this engine could locate impact two ways and neither of them saw it.
`SwingPhases` reports the peak of hand speed and says in its own contract that
this is a kinematic estimate rather than an observation, because the hands reach
their peak before the club head reaches the ball. `ClubTrackingReport` reports
the lowest point the club head reached after the top, and refuses on most real
footage because it needs the head visible through the blurriest frames in the
clip. `docs/architecture.md` promised that Phases 10 and 11 would replace that
corroboration with real evidence.

Two questions had to be answered to do that, and they are separable.

**What does a ball detector detect?** The obvious answer is the ball, tracked
through the frame. The obvious answer is unavailable: a ball leaves the face at
around 70 m/s, so at a 1/60 s exposure it is a metre-long smear, and at any
consumer frame rate it is past the edge of the picture before it has been drawn
sharply once. Phase 10 had already met this wall with the club head and named it
— blur is a cliff, not a decline.

**What does the system report when three estimates disagree?** Averaging is the
reflex. There was no precedent in the project for it, and `contracts/phases.py`
and `contracts/club.py` had each already refused a weaker version of the same
thing: tempo was kept out of Phase 4, "transition to impact" out of Phase 5, and
the club-head estimate was explicitly forbidden from overwriting the kinematic
one.

## Decision

**The ball is measured where it is easiest — at rest — and impact is read off the
frame it stops being there. The four estimates are ranked rather than averaged,
and every disagreement is kept as a measurement.**

Concretely:

- A departure is the boundary between the last frame carrying a ball and the
  first carrying none. Both ends are reported. `BallDeparture.interval_s` is the
  gap between them and is documented as a **bracket** — impact is inside it.
- Identification and instant are separate, and so are their confidences.
  `BallConfidence` scores one frame's observation; `DepartureConfidence` scores
  the instant. They are computed from disjoint evidence.
- `analyzer/impact.py` reports one instant from one source, with the source
  named, the error bar attached, and every other estimate listed beside it with
  its delta. Nothing is blended.
- `FusedImpact.observed` is true only for a ball departure, so a consumer can
  gate a claim on whether the event was seen.

## Why: the absence is the measurable thing

A teed ball is still, round, high-contrast and in one place for the whole of
address, the backswing and the downswing — hundreds of frames of unambiguous
evidence about one small object. The event being timed is not something it does.
It is the frame after the last one it appears in.

The consequence is the property no other impact estimate in this project has:

| estimate         | uncertainty                         | kind      |
| ---------------- | ----------------------------------- | --------- |
| ball departure   | one frame interval                  | a bracket |
| club head        | the sampling of the observed arc    | a floor   |
| hand low / speed | the smoothing window the fps forced | a scale   |

Nothing in the ball path estimates a peak, fits a curve or smooths a signal, so
nothing in it has a resolution that degrades. At 30 fps the bracket is 33 ms and
at 240 fps it is 4 ms, and in both cases impact is **inside** it. Phase 4's
uncertainty is the width of the window its own frame rate forced, which on a
24 fps clip is about as long as the downswing being measured — and that is a
scale on which the peak could have moved, not an interval it is inside. The two
are not comparable, which is why `FusedImpact.uncertainty_is_bracket` exists
rather than a single number a reader would be free to misread.

## Why: the identification is circular, and the circle is closed rather than hidden

Nothing else in a golf frame is a small still round object that disappears once
and never returns. That fact is what picks the ball out of the stationary
candidates — a tee marker, a white shoe, a bright mat seam, a daisy — and it is
also the thing being reported. Scoring the confidence of the departure on the
departure would be an argument with itself.

So the evidence is partitioned, and the partition is enforced by the two
confidences being different objects:

```
BallConfidence        contrast x margin x stillness     one frame's observation
DepartureConfidence   establishment x abruptness x permanence     the instant
```

Neither reads the other's inputs. `test_ball_track.py` asserts this directly:
halving the contrast of every frame in a clip changes the observation quality and
must not move the departure's confidence at all.

What is left over is the one thing a circular identification genuinely cannot
check — if **two** stationary candidates depart, the clip cannot say which was
the ball — and that is reported rather than solved. It is `EstablishedBall.margin`,
and a low value there is the failure that looks perfect from every individual
frame's point of view.

## Why: precedence rather than averaging

Three of the four estimates are biased, in a known direction, and one is not.
Averaging an unbiased observation with a biased proxy moves the answer away from
the truth by a fraction of the bias, and produces a number whose provenance is "a
bit of each" — which no reader can reason about and no later phase can correct.

The order is a property of what each source measures, not of how confident any
particular clip's version of it happens to be. An observation beats a measurement
of a quantity that coincides with the event; that beats a proxy; and among
proxies, the one with no known bias beats the one with a known one. A confident
proxy is still a proxy, and `test_impact.py` pins that: a ball departure scored
at 0.1 still outranks a hand-speed peak scored at 0.8.

The disagreements are then not diagnostics. They are **the measurement this phase
makes possible**: `kinematic − ball` is how the bias of the hand-speed peak gets
quantified, on a clip and eventually across a set of them. Discarding them in an
average would have destroyed the only number Phase 11 adds that Phase 12 will
need.

`HAND_LOW` above `HAND_SPEED` is the one step in the order that rests on a
measurement rather than an argument, and it is the question the roadmap left open
after the tour-pro footage arrived. It is now measured rather than estimated. On
`data/rory/face-on/rory_face_on.mp4`, with the ball bracketed between frames 360 and
361:

| source           | frame | delta vs the observation |
| ---------------- | ----- | ------------------------ |
| `ball_departure` | 361   | **reported**             |
| `hand_low`       | 362   | +5 ms / **+1 frame**     |
| `hand_speed`     | 381   | +95 ms / **+20 frames**  |

The arc low lands within a frame; peak hand speed lands twenty frames away and
**late**, which is the opposite of the direction `contracts/phases.py` has warned
about since Phase 4 — the hands are supposed to peak _before_ the head arrives.
That reversal is not explained here and is not explained away: it is one clip, it
is slow-motion footage whose factor is supplied rather than measured, and a bias
measured once is an anecdote with a number attached. It is enough to set a
precedence and not enough to be a correction.

**Phase 4's own primary estimate is unchanged by this.** Nothing in the fusion reaches back down and
rewrites what `detect_phases` returns, because an engine in which a number
changes depending on which other analyses happened to run is an engine whose
outputs cannot be compared across clips.

## What this cannot claim

**The ball is not observed being struck. It is observed stopping being visible.**
For a struck ball those are the same instant; they are not the same statement,
and the difference is where every failure of this phase lives.

The one that sounds fatal: at impact the club head is at the ball, so a ball
hidden behind it and a ball that has left are the same picture. That does not
matter for this measurement — both happen at impact. It would matter if the club
covered the ball appreciably _before_ contact, which it can on a down-the-line
view, and `DepartureConfidence.abruptness` is what notices: a struck ball is as
visible in its final frame as in its first, and a covered one dims first.

Measured on the fixture, the failure is real and it is reported rather than
fixed. A ball faded over eight frames is located two frames early, with the
abruptness factor at 0.43, the permanence factor under 1.0 and two warnings
naming the cause.

That permanence factor is measured on the **detector's candidates** rather than
on the frames the tracker accepted, and that choice was made the other way first.
A dimming ball falls under the confidence bound and ends the run early; every
frame after that is then one the tracker rejected, so an acceptance-based score
reads 1.0 while the ball is still plainly in the picture. The dim ball is still a
candidate. Reading candidates notices; reading verdicts does not.

## Consequences

- Impact acquires a provenance and an error bar, and `FusedImpact.observed` lets
  Phase 13 refuse to phrase a finding as "at impact" when nothing saw it.
- The bias of the kinematic estimate becomes measurable on any clip with a
  visible ball, which is the input Phase 12 needs to correct it.
- A new failure mode enters the system that no earlier phase has: a confident
  instant read off the wrong object. It is bounded by `EstablishedBall.margin`
  and by the requirement that a departure land between the top and the finish,
  and neither is a substitute for the labelled set Phase 12 builds.
- `data/README.md` acquires a capture requirement it did not have: one ball in
  the hitting area, on a surface it contrasts with, and two seconds of recording
  after the strike so the absence can be verified.
