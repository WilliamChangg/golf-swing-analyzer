# ADR-0017: The number with no protocol, the protocol with no number, and the guard above the model

**Status:** Accepted
**Date:** 2026-09-18
**Phase:** 13 — Coaching engine

## Context

Every layer below this one measures something and says what kind of measurement
it is. Phase 5 gave each metric a `basis` so that a shoulder turn inferred from
foreshortening could never be mistaken for a shoulder turn measured in three
dimensions. Phase 13 is where those measurements become sentences a person acts
on, and it is the first layer where being wrong is expensive rather than
embarrassing: a wrong angle is a wrong number, and a wrong finding is a golfer
changing their swing.

The obvious build is a table of thresholds and a language model to read them
out. Both halves of that fail in ways nobody notices:

**The thresholds.** Golf coaching runs on numbers with no papers behind them.
"Ninety degrees of shoulder turn", "forty-five degrees of X-factor", "three to
one tempo". Hard-coded as `THRESHOLD = 90.0` they look identical, and they are
not: one is instruction-book convention with no stated axis or instrument, one is
a magazine article that named a quantity before anybody measured it, and one is a
frame count off video that can be reproduced. Where a threshold _does_ come from
a measurement, the measurement is almost always three-dimensional motion capture
— which is a number about a quantity this engine does not have.

**The model.** A language model handed structured findings will produce fluent
prose containing, somewhere, a number nobody measured. Rarely, which is the
problem: a system wrong once in fifty cannot be trusted anywhere without
checking, and the checking is the expensive part.

## Decision

**A threshold declares how it was measured, and that decides what it may be
compared against. A convention permits nothing. Prose may contain no number its
evidence does not, and that rule is enforced on this engine's own sentences
before it is enforced on any model's.**

Concretely:

- `ThresholdSource` is required on every rule and carries `method`,
  `population`, `measures` and `permitted_bases`. The engine refuses any rule
  whose metric's `basis` is not in that list, **before** it looks at whether the
  clip produced the measurement.
- `ThresholdMethod.CONVENTION` sources declare `permitted_bases = []`. A number
  with no published measurement protocol has no quantity attached to it, so
  nothing can be compared against it. Six of the twelve rules rest on one, and
  they are in the registry, refused by name.
- `ThresholdMethod.SAME_CLIP` is the only kind that borrows nothing: one quantity
  against itself at two anchors of one recording, where the threshold is the two
  measurements' own combined uncertainty.
- Every band states its derivation. The tempo band is not "3:1 give or take"; it
  is what the source's own published frame counts permit once one frame of
  ambiguity at the top is allowed for.
- A comparison must clear a **bracket** before it is reported as a comparison.
  Durations get one frame interval; the tempo ratio gets the much larger figure
  that one frame at the top moves it by; everything else gets the metric's
  measured uncertainty, or is refused.
- `analyzer/coaching/guard.py` checks numbers, units, unmeasured quantities,
  ball-flight claims and causal claims, and it runs over `Finding.observation` in
  the test suite as well as over every model candidate. A candidate with one
  offence is discarded whole.
- The phrasing layer is off by default, never sees a frame or a landmark or a
  path, and refuses any endpoint that is not loopback.

## Why the basis gate is checked before the clip

The gate order runs from facts about the world to facts about this recording,
and `BASIS_NOT_PERMITTED` sits above `NO_METRIC` deliberately. A reader told
"this clip could not measure your shoulder turn" will go and re-film, and the
re-filmed clip will be refused for the basis instead. Telling them the real
reason first saves them the trip.

## Why a convention permits nothing, rather than a low confidence

The tempting middle path is to let the folklore rules fire with a heavy caveat.
It does not work, and the reference footage says why.

On `data/rory/face-on/rory_face_on.mp4` — a tour professional — this engine
measures a **shoulder turn of 53.0 ± 5.5 degrees at the top**. The convention the
rule would compare against is about ninety. The swing is not restricted; the
measurement is a foreshortening estimate from one camera, which under-reports by
tens of degrees and says so in its `basis`. A rule firing here with a caveat
attached would tell a tour player to turn more, and no wording of the caveat
changes what the reader takes away.

The same clip gives an X-factor of 25.2 ± 15.8 degrees against a popularised
forty-five. Both numbers are correct measurements of what the camera saw. Neither
is a measurement of the quantity the threshold is about.

## The pair that names the decision

Two rules in the registry sit either side of the same gap:

| rule                       | source                        | outcome                         |
| -------------------------- | ----------------------------- | ------------------------------- |
| `rotation.x_factor_top`    | a magazine article, 1992      | no measurement protocol         |
| `rotation.x_factor_top_3d` | three-dimensional measurement | no number this project can cite |

The quantity with a number attached has no protocol. The quantity with a protocol
has no number this project has read a figure for — that literature reports group
means and spreads, which is not a range that separates one swing from another,
and deriving one would be this engine's opinion wearing a citation.

Keeping both in the registry, refused for different reasons, is the honest
inventory. A coaching engine that silently omitted the eight numbers every golf
app displays would look like it had not thought of them.

## What survived, and what that cost

Five rules can reach a comparison, and on the reference footage the only ones
that ever produce a finding are the three temporal ones. That is not a
coincidence and it is the same argument the rest of this project runs on: a
duration depends on the clock and on nothing else, so it is the one family of
measurement a camera position cannot distort.

The other two compare a clip against itself and need no borrowed number at all.
They refuse on this footage for a different and more tractable reason: Phase 6
quantifies an uncertainty for the foreshortened rotations and for nothing else,
so a rule asking whether the spine angle changed between address and impact has
two measured values and no measurement of how well either is known.

The cost arrived immediately. `analyzer coach` on the 30 fps amateur reference
clip produces **no findings at all** — all three timing comparisons come back
`UNRESOLVED`. One frame of ambiguity at the top moves that clip's tempo ratio by
0.74, and the whole published band is 1.37 wide. The tempo number every golf app
puts on its front page is not resolvable by the camera it is measured with, and
`scripts/benchmark_coaching.py --sweep resolution` puts the crossover at
**56 fps** for that swing.

The source itself is more careful than its readers: Tour Tempo publishes frame
counts, 18/6 and 21/7 and 24/8, not a ratio. 3:1 exactly is a ratio of two
integers, and is not something a 30 fps measurement of a quarter-second downswing
can mean.

## The gate the build found: a supplied timebase

Absolute-duration comparisons are refused on any clip whose slow-motion factor is
not 1.0. Nothing in a conformed slow-motion file records its playback factor —
this project established that in Phase 6 and carries it as a caller-supplied
number — so every duration measured from such a clip is a measured number
multiplied by a guess, and comparing one against a band in seconds compares a
stopwatch against the guess.

The tempo **ratio** is spared, because a factor that stretches both durations by
the same amount divides out of their quotient. That is the same shape of argument
that makes a length in torso lengths survive an unknown camera, one dimension
over, and it is why the only rule that fires on the tour footage is the one that
is a ratio.

## Consequences

**A finding is checkable, and the first thing it checks is this engine.** The
tempo rule reports that a tour professional's swing is 1.83:1 against a tour
band. The finding cites the backswing, the downswing and the frames each came
from, and a reader following those frames finds the takeaway, which Phase 6 had
already flagged as suspect on slowed footage. A sentence of advice would have
been wrong in the same way and unfalsifiable. **Evidence is what makes a wrong
finding discoverable**, and that is the argument for citing it — not
transparency in the abstract.

**No score, ever.** There is no severity, no grade and no ranking between
findings, and a test asserts the absence of those field names. A single number
summarising a swing would be the most quoted output of this system and the one
with the least behind it: it would need a scale relating degrees of turn to
seconds of tempo, and nobody has measured one.

**The guard's first catch was our own sentence.** The same-clip template read
"more than the X degrees the two measurements leave open", and the guard read the
spelled-out "two" as a quantity absent from the evidence and rejected it. The fix
was to reword the template. Teaching the guard to ignore small counting words
would have widened the hole the whole check exists to close, and the template
now reads "those measurements".

**The phrasing layer has never run.** No language model is installed on the
machine this was built on, so the protocol, the request shape, the guard
integration and every failure path are exercised against fakes, and
`PhrasingReport` carries the counts that would turn a later run into evidence.
`attempted = 0` and "the model rephrased nothing" are different states and are
recorded differently.

**Eight refusals per clip is the normal output.** A reader who sees a short
findings list should not conclude the swing was clean, which is why every refused
rule is reported with its reason and why the report warns that seven of the
twelve were refused before the clip was looked at.

## Alternatives rejected

**Let conventions fire with low confidence.** Rejected on the measurement above:
a caveat does not stop a 53-degree reading being presented as a restricted turn.

**Invent bands from this project's own footage.** Four clips, two golfers. A
threshold derived from that would be a norm with a sample size of two presented
as a norm, which is the exact failure Phase 12 refused to commit with a model.

**Let the model repair its own output.** A repair pass makes the guard a
negotiation. The candidate is discarded whole and the engine's sentence ships.

**Have the model choose which findings to show.** Then a language model decides
what was measured. Findings are produced first and phrased second, and the order
is enforced by the phrasing layer only ever receiving findings that already
exist.

## References

- Phase 5's `MetricBasis`, which this gate is the consumer of:
  [ADR-0010](ADR-0010-projected-biomechanics.md)
- Phase 12's refusal to publish an unsupportable number:
  [ADR-0016](ADR-0016-labels-groups-and-the-noise-floor.md)
- The slow-motion factor as a supplied rather than measured quantity:
  the Phase 6 amendment in [ADR-0010](ADR-0010-projected-biomechanics.md)
