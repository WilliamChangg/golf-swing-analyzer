# ADR-0021: The smoothing window is resolved against the clip, not asked of the caller

**Status:** Accepted · **Date:** 2026-09-19 · **Supersedes:** one consequence of [ADR-0009](ADR-0009-local-polynomial-filtering.md)

## Context

ADR-0009 established a frame-rate floor and chose to state it rather than work
around it:

> The alternative — silently widening the window to whatever the clip can
> support — was rejected. It produces numbers, and they would be worse in a way
> nothing reported.

That reasoning is sound and the decision it produced was wrong, for a reason the
ADR states two sentences earlier without drawing the conclusion: _"Both reference
clips in this repository are 24–30 fps, so this is the common case rather than an
edge one."_

What the floor did in practice was measured, on every clip in `data/`:

| Clip                                          | fps   | With the shipped defaults |
| --------------------------------------------- | ----- | ------------------------- |
| `amateur/face-on/PW_face-on.mp4`              | 30    | nothing fitted            |
| `amateur/dtl/iron_dtl.mp4`                    | 30    | nothing fitted            |
| `tommy_fleetwood/dtl/tommy_fleetwood_dtl.mp4` | 29.97 | nothing fitted            |
| `rory/face-on/rory_face_on.mp4`               | 30    | nothing fitted            |
| `rory/dtl/rory_dtl.mp4`                       | 30    | nothing fitted            |
| `rory/face-on/rory_faceon_3.mp4`              | 60    | filtered                  |
| `rory/dtl/rory_dtl_2.mp4`                     | 60    | filtered                  |
| `tommy_fleetwood/face-on/tommy_face-on.mp4`   | 60    | filtered                  |

Five of eight produced **no filtered value at any sample of any landmark**, so
`detect_phases` refused them with "the hands were never tracked" — which points
at pose estimation, the one thing that had worked. Nothing above the filter could
distinguish this from a clip with no subject in it.

Three further observations decided it:

1. **The floor was never actually accepted; it was routed around.** Every
   benchmark script that touched a 30 fps clip carried a hand-picked
   `window_s=0.17`, and the README told users to pass `--window 0.15`. A default
   that every caller in the repository overrides is not a policy.
2. **It masked better diagnoses.** `rory_face_on.mp4` is ~7x slow motion, and the
   engine has a precise warning that names the factor to supply. The filter died
   first, so the user never saw it.
3. **The rejected alternative was "silently".** The objection in ADR-0009 is to
   numbers that are worse in a way _nothing reported_ — not to the numbers.

## Decision

`filter_sequence` resolves the smoothing window against the clip's own measured
sampling interval. Where the configured window cannot hold the observations its
polynomial needs, it is widened to `narrowest_window_s` — the narrowest width
that can — and the widening is reported.

Four things make this the opposite of silent:

- `SequenceFilterReport.warnings` leads with the widening, naming both widths,
  the clip's frame rate, and what the wider window costs.
- `SequenceFilterReport.config` carries the window **as applied**, with
  `requested_window_s` beside it. Every other number in the report, and the
  resolution factor phase detection derives from it, describes the fit that ran.
- The existing resolution confidence factor already prices the cost: a window
  approaching the length of the event scores toward zero. On the 30 fps
  reference clips impact confidence lands near 0.5, which is the honest reading.
- `SmoothingConfig.auto_widen=False` restores the refusal for a caller who would
  rather recapture than measure wider.

**Narrowest, not comfortable.** Measured on `PW_face-on.mp4`, where 0.150 s is
the narrowest that fits:

| window at 30 fps | top / impact confidence |
| ---------------- | ----------------------- |
| 0.150            | **0.50 / 0.50**         |
| 0.183            | 0.38 / 0.37             |
| 0.200            | 0.00 / 0.00             |

Every extra sample averages over more of the swing, and velocity is what phase
detection keys on. Widening to a round number would have cost most of what the
widening was for.

**Resolved once for everything that will be compared.** Per-landmark resolution
would let two landmarks of one clip be smoothed differently, and their velocities
would not be comparable with nothing able to tell. The same argument reaches past
one clip: `sync_clips` resolves one window across the pair before filtering
either, so the coarser clip sets it for both. That was already the stated rule in
`SyncClipsParams.filter`; per-clip widening would have broken it silently, which
is exactly the failure ADR-0009 was guarding against.

## Consequences

**The frame-rate floor is still real and still reported.** It has moved from
"this clip yields nothing" to "this clip was measured at a width you did not ask
for, and here is what that costs". The quantitative backing for the ≥120 fps the
capture protocol asks for is unchanged — it is now visible as degraded
confidence on a result rather than as an absent one.

**ADR-0009's other decisions stand.** The fit on real timestamps, the derivatives
read off the polynomial, the bracketing rule, and the separation of gap policy
from the fit are all untouched. Only the consequence quoted above is superseded.

**A schema version.** `FILTER_SCHEMA_VERSION` is 2. A version-1 reader would take
`config.smoothing.window_s` for the requested width; it is now the applied one.

**Two reference clips still need a flag, and that is correct.** `rory_face_on`
(~7x) and `rory_dtl` (~5x) are slow motion, which nothing in a conformed file
records. The engine does not guess it; it now gets far enough to say so.
