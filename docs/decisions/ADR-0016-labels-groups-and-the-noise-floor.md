# ADR-0016: The tool before the model, the player as the group, and the noise floor as the gate

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 12 — Temporal ML

## Context

Phase 4 locates the four swing events with rules that can be read and argued
with, and locates them well enough that nothing in Phase 12 is needed for the
product to work. What no phase so far can do is say **how well** — the rules have
never been scored against anything, because scoring needs somebody to have said
where the events actually were.

Phase 12 was specified as "temporal ML", and the obvious reading of that is
"train a TCN". The obvious reading is the one that produces the number every
golf-analysis project publishes and none of them can defend. Four decisions had
to be made before any model was trained, and each of them is a place where the
easy choice produces a plausible figure that means nothing.

## Decision

**The labelling tool is built before the model; the split holds whole players
together; a score is refused publication whenever the data cannot support it;
and nothing is ever labelled by the detector being evaluated.**

Concretely:

- `analyzer/ml/labeltool.py` and the `ClipLabel` schema exist and are tested
  before `analyzer/ml/tcn.py` has a training loop. `EventLabel.uncertainty_frames`
  is required and has no default.
- `player_id` and `session_id` are required fields with no defaults, and
  `analyzer/ml/splits.py` groups by **player**.
- `SplitReport.refused` is a normal outcome. `EvaluationReport.claims_permitted`
  is false with a stated reason whenever the set is synthetic, has too few
  held-out players or clips, or the error is smaller than what the labels resolve.
- `LabelProvenance` has two values, `HUMAN` and `SYNTHETIC`, and deliberately no
  third value for "produced by the rule-based detector".

## Why: the tool first

Building the model first is how a project ends up training on whatever labels
were convenient to produce. Building the tool first forces the question of what a
label _is_ to be answered while the answer is still cheap to change — and the
answer turned out to include three things that a "frame number per event" schema
would have lost:

**Who said it.** Two labellers disagree about the top by more than the frame rate
resolves. A set built from both without recording which is which cannot be
audited afterwards.

**How sure they were.** A takeaway on a sharp 240 fps clip is placeable to a
frame; impact on a 30 fps clip where the club crosses the ball inside one
exposure is not. A schema that accepts both as bare integers has two numbers that
look identical and mean different things, and every evaluation downstream then
credits the second with a precision nobody claimed.

**Whose swing it is.** This cannot be inferred from a file path, so it is not
inferred. It is entered, or there is no label.

The tool refuses to save a half-marked swing for the same class of reason: a file
with a takeaway and an impact and nothing between reads downstream as a clip
where the top was not visible, which is a different statement from a labeller
having been interrupted.

## Why: the player, not the session, not the clip

Swings from one session are near-duplicates. Same player, same club, same camera,
same light, same ball position, minutes apart, and a good golfer's swing repeats
to within less than the noise in the landmarks. One in training and its
neighbour in test is a model being asked a question it has already been given the
answer to.

Grouping by session is the tempting half-measure. It keeps near-duplicates
together and still lets one golfer appear on both sides of the split, which
measures how well the model recognises _that golfer on a different day_. Grouping
by player holds sessions together as a consequence — a session belongs to exactly
one player — so the stronger grouping is also the simpler one.

**The leak check is a measurement of the output, not an assertion inside the
splitter.** `LeakageCheck` counts players, sessions and content keys appearing in
more than one role, reading the assignment that was actually produced. An
assertion inside a splitter tests that the splitter agrees with itself. The
content-key count is the one that looks redundant and is not: it catches the same
footage labelled twice under two player ids, which is how duplicate clips
actually arrive — a file copied into two session folders.

`random_clip_split` is kept beside the real splitter as a measuring instrument,
and it fails the leak check by construction, so it can never be mistaken for the
honest one.

**We could not measure what the leak is worth, and that is reported rather than
estimated.** `scripts/benchmark_ml.py --sweep leakage` trains under both splits on
the same corpus with the same seeds and finds no difference larger than the
seed-to-seed scatter. The reason is a property of the fixture: its golfers differ
by six generator parameters, so a model that has seen seven of them has seen the
space and holding one out asks nothing. Real golfers differ in ways a generator
does not know how to vary. The grouping stands on the argument, not on that
measurement — a swing and its near-duplicate cannot be on opposite sides of a
question, and no measurement is needed to see it.

## Why: a noise floor, and a refusal that is part of the type

The project's standing rule is that no labelled dataset means no accuracy claim.
Phase 12 is the first phase where that rule has something to refuse, so it is
expressed where a caller cannot route around it by forgetting:
`EvaluationReport.claims_permitted` is computed, and `claim_refusal` names the
first reason in order of how disqualifying it is.

The reason worth explaining is the last one, because it is the failure that looks
like success. An error smaller than the labels' own uncertainty is not an
achievement; it is a measurement below the resolution of the instrument. The
floor has two parts and they add:

| part          | what it is                                       | value here     |
| ------------- | ------------------------------------------------ | -------------- |
| label bracket | the median of what labellers said they could see | set by the set |
| resampling    | half a sample of the 60 Hz feature grid          | 8.3 ms         |

A model whose mean error on the top is 6 ms against labels marked "give or take
two frames at 60 fps" has not been shown to be better than the labels. Reporting
that 6 ms as an accuracy is how a project talks itself into shipping.

The resampling half of the floor is worth its own sentence, because it sets a
ceiling on what this whole approach can be worth. At 240 fps a ball departure
brackets impact to 4.2 ms (ADR-0015); the feature grid alone costs 8.3 ms before a
model has run. **A learned detector is not a replacement for seeing the ball.**

## Why: never label with the detector being evaluated

The tempting shortcut, once Phase 11 can locate impact by observation, is to seed
labels from the engine's own output and hand-correct them. It is rejected, and
`LabelProvenance` has no value for it.

Training a model on a detector's output and then comparing the model against that
detector measures **imitation**. The model converges on the baseline's own biases,
including the ones nobody has found yet, and the comparison reports agreement as
accuracy. Phase 12.6 exists to answer whether the model beats the rules, and
seeding the labels from the rules answers it with a number that cannot mean
anything.

The weaker version — seeding only impact, from the ball departure, which _is_ an
observation — is not obviously wrong and is still not done here. A set in which
one of four events came from a different process than the other three has a
provenance that no single field can describe, and this schema would have to lie
about it.

## What this cannot claim

**No number produced in Phase 12 is a statement about golf.** Every figure in
`docs/ROADMAP.md` and in this repository's benchmarks for this phase was measured
on `tests/synthetic_labels.py`, whose events are inputs to the generator that drew
the motion. A detector scored against them is being asked to recover parameters
it was effectively told.

The comparison against the rule-based detector is, on that corpus, **rigged in the
model's favour**, and the rigging is worth understanding because a version of it
survives into real data. The model is trained on these labels, so it learns the
convention they were made with; the rules brought their own. The two conventions
genuinely differ at the takeaway: the generator's takeaway is where its easing
function leaves zero, and Phase 4's is where hand speed crosses 5% of its peak,
which on a sin-squared ramp is several frames later. The model is not more
accurate there. It has been told which definition is being marked. A human-labelled
set narrows this and does not close it — a person marks the takeaway where they
can see the club move, which is a third convention again.

## Consequences

- The engine gains a labelling tool, a versioned feature definition, a
  group-aware splitter, a trainable baseline, a comparison harness and a model
  registry, and gains **no accuracy claim whatsoever**.
- `analyzer labels` on this repository's own footage prints a refusal: four clips,
  one golfer, no held-out player. That refusal is Phase 12's result.
- `data/README.md` acquires a capture requirement it did not have: a labelled set
  needs _players_, not clips, and more clips of one golfer do not help.
- Phase 13's coaching rules may not cite a model's confidence as evidence, because
  no model in this registry is permitted to claim one.
- A feature-definition change now invalidates every checkpoint by digest rather
  than silently producing wrong predictions from the right-shaped arrays.
