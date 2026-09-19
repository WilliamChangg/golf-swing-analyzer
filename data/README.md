# Data

**Nothing in this directory is committed.** `.gitignore` excludes everything
except this file. Video is personal data and the project's core promise is that
it stays on your machine — keeping it out of git is part of that, not an
afterthought.

Put local footage anywhere under `data/`; a suggested layout:

```
data/
  raw/<session-date>/<player>/faceon.mov
  raw/<session-date>/<player>/dtl.mov
  calibration/<camera>/charuco_*.jpg
  fixtures/            small synthetic clips for tests
```

## Slow-motion footage

Nothing in a conformed slow-motion file records that it is slow motion. A clip
captured at 240 fps and written out at 30 fps has honest timestamps at 1/30 s
apart and a swing that takes eight times too long, and no metadata separates it
from a genuinely slow movement.

Every duration rule in this engine is stated in **real** seconds, so such a clip
is refused until the factor is supplied:

```bash
uv run --project python analyzer metrics clip.mp4 --slow-motion 8
```

The detector says so rather than leaving it at "no swing detected": when a
motion has a swing's shape but every phase is long in the same proportion, it
reports the smallest factor that would bring the durations inside the bounds.

**The factor is not a relabelling of the clock.** The smoothing window is in real
seconds too, so the factor decides how many frames it holds, and that changes
where events land. Getting it right matters; guessing it badly is visible in the
results. Phone slow motion is usually 4x or 8x, from a 120 or 240 fps capture.

Supplying it correctly is a gain rather than a concession: a slow-motion clip is
a high-speed capture, so the events resolve better than on any ordinary 30 fps
recording. On the reference footage every event scores a resolution factor of
1.00, which no 30 fps clip in this project manages.

## Reference footage

Not committed, and not ground truth — no one has motion-captured any of these.
They are a stress test, and they earn their keep by disagreeing with the engine.

| Clip                          | View          | Notes                                    |
| ----------------------------- | ------------- | ---------------------------------------- |
| `face-on/PW_face-on.mp4`      | face-on       | 68 frames, 30 fps, real time             |
| `dtl/iron_dtl.mp4`            | down the line | 96 frames, 30 fps; filmed from the front |
| `dtl/driver_swing_aug19_2026` | down the line | 24 fps; refused, motion blur             |
| `face-on/rory_face_on.mp4`    | face-on       | tour pro, ~7x slow motion, from behind   |
| `dtl/rory_dtl.mp4`            | down the line | tour pro, ~5x slow motion                |

The tour-pro clips found two defects the amateur footage could not, because they
contain a full turn and a slow-motion clock. Both are recorded in
[../docs/decisions/ADR-0010-projected-biomechanics.md](../docs/decisions/ADR-0010-projected-biomechanics.md).

Both tour clips have now been club-tracked, and they found the failure the
synthetic sweeps predicted. `rory_face_on.mp4` tracks 21% of frames clip-wide in
29 separate runs, and nine of the 139 tracked frames follow the vertical **edge
of the yardage sign** behind the player rather than the club — at a median
confidence of 0.98. Checked against the frames with `scripts/overlay_club.py`,
everything else is genuinely the shaft. The clip reports up to 217 px of
club-head smear at a 360-degree shutter, twenty times the ten-pixel bound, which
is why most of it is refused.

`dtl/rory_dtl.mp4` inverts the usual ordering: 8% clip-wide but 25% through the
downswing, its best-covered phase. Down the line the club points towards the
camera at address and is foreshortened past the minimum length, so "the club is
easiest to find where it is slowest" turns out to be a face-on observation.

`face-on/rory_face_on.mp4` also carries the only **observed** impact in the
project: the ball is on the tee at frame 360 and gone at frame 361. That was
first read off the frames by eye and is now measured — `analyzer ball` brackets
it between exactly those two frames, with no warnings — which makes it the only
event in this repository against which a kinematic estimate can be scored. It is
scored in
[../docs/decisions/ADR-0015-ball-departure-and-impact-precedence.md](../docs/decisions/ADR-0015-ball-departure-and-impact-precedence.md),
and the answer is that peak hand speed lands 20 frames away while the lowest
point of the hand arc lands within one.

The same clip found six defects in Phase 11 that the synthetic fixture passed
cleanly, including a search region anchored at the hands that identified a patch
of **sky** as the ball, and a fixed drift reference that read the camera's own
gentle motion as the ball departing 58 frames early. Every one of them produced a
confident wrong answer rather than an error.

## Capture protocol

Analysis quality is bounded by capture quality, and most of what limits it is
decided before you press record. Two cameras, both seeing the entire swing.

### Camera placement

**Face-on** — perpendicular to the target line, level with the hips, centred on
the player. The optical axis should be square to the target line; if the camera
is angled, measured rotations are biased by the projection and there is no way
to recover the true angle without calibration.

**Down-the-line (DTL)** — behind the ball, positioned _on_ the target line,
again at hip height. Standing off the line is the most common DTL error and it
systematically distorts apparent shaft plane and club path.

Keep both cameras fixed. Any pan or handheld drift is indistinguishable from
body motion to a single-camera pipeline.

**The angle between the two cameras is the single most important thing about the
placement, and it is measured rather than advised.** A 3D point is the
intersection of two rays, and how well it is determined depends on the angle they
meet at: depth error scales as `1 / sin(theta)`. Face-on plus down-the-line puts
that near 90°, which is the best it can be — and it is what the two placements
above already ask for, so following them costs nothing extra. What to avoid is
putting both cameras on the same side of the player:

| separation | reconstruction error | reprojection error |
| ---------- | -------------------- | ------------------ |
| 90°        | 5.5 mm               | 0.85 px            |
| 45°        | 6.5 mm               | 0.84 px            |
| 30°        | 8.2 mm               | 0.84 px            |
| 15°        | 14.3 mm              | 0.84 px            |
| 8°         | 26.0 mm              | 0.84 px            |

Note the right-hand column. Nothing in the reconstruction's own residual changes
across that range, which is why the system gates on the angle and refuses below
15° rather than trusting a fit that looks identical either way
(`scripts/benchmark_reconstruct.py --sweep convergence`).

### Settings

| Setting        | Recommendation               | Why                                                                                                                                          |
| -------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Frame rate     | **≥ 120 fps**, 240 preferred | Downswing lasts ~0.25 s; at 30 fps that is ~8 frames total. Measured: below ~56 fps no tempo comparison is possible at all (Phase 13, below) |
| Shutter speed  | **1/1000 s or faster**       | Measured: shaft detection dies above ~10 px of smear, and it is a cliff. See below                                                           |
| Resolution     | 1080p is sufficient          | Frame rate and shutter matter far more than resolution                                                                                       |
| Stabilisation  | **Off**                      | Digital stabilisation warps the frame non-rigidly, corrupting geometry                                                                       |
| Focus/exposure | Locked                       | Refocus mid-swing changes apparent scale                                                                                                     |

### Framing and environment

- Whole body plus the full club arc in frame for the entire swing, including
  follow-through. Clipping the top of the backswing loses the `top` event.
- Even, bright lighting. Avoid strong backlight, which silhouettes the player
  and destroys landmark confidence.
- Uncluttered background. Vertical lines behind the player (door frames, fence
  posts, window mullions) are the main false-positive source for Hough-based
  shaft detection, and that is now measured rather than asserted: a door frame
  passing through the hands scores **the same edge support as the club and more
  length**, so evidence alone ranks it first. What rescues it is that a door
  frame does not rotate — see below.
- Clothing that contrasts with the background; avoid very loose garments, which
  hide joint positions.

### Filming so the club can be tracked (Phase 10)

The club is the fastest thing in the frame and the only thing in it with no model
behind it — a shaft is found by its edges, and motion blur destroys edges. That
makes the shutter the single setting that decides whether club tracking works,
and the bound is measured rather than advised
(`scripts/benchmark_club.py --sweep blur`):

| smear at the club head | shaft found | angle error |
| ---------------------- | ----------- | ----------- |
| 0 px                   | **100%**    | 0.4°        |
| 5 px                   | **100%**    | 0.9°        |
| 10 px                  | **100%**    | 0.9°        |
| 14 px                  | **0%**      | —           |
| 29 px                  | **0%**      | —           |

**It is a cliff, not a decline.** A smeared shaft does not become a weaker line,
it stops being a line: an exposure draws a rotating club as a fan with no edge in
it. Where the club is found the angle is right to about half a degree, and where
it is not, no setting recovers it. There is deliberately no sensitivity option
that trades detection rate against accuracy, because there is nothing to trade.

Smear is `club-head speed in the image x exposure time`, so the arithmetic is
yours to do before you press record. A club head crossing 8,000 px/s on a 1080p
frame — an ordinary tour-speed swing at ordinary framing — needs an exposure
under about 1/800 s to stay inside the bound. That is where the 1/1000 s in the
table above comes from.

**Nothing in a video file records the exposure**, so the system cannot measure
your shutter. What it reports instead is `max_blur_px`: the club head's measured
image speed times the frame interval, which is the smear at a 360° shutter — the
worst any camera does — and therefore an upper bound. A shutter _n_ times faster
than the frame interval divides it by _n_.

**The frame rate is a second, weaker lever.** More frames make each exposure
shorter only if the camera is already shutter-limited by them, and they do not
help a camera that is exposing for the whole interval in poor light. Bright
light is what buys a fast shutter. Film outside.

**A clip's overall detection rate is not the number to read.** Detection is easy
at address, where the club is stationary, and address plus the follow-through are
most of a clip. The downswing is where every metric worth computing lives and it
is the part that fails first:

| shutter         | overall | address | **downswing** |
| --------------- | ------- | ------- | ------------- |
| 1/4 of interval | 93%     | 100%    | **83%**       |
| 180° (1/2)      | 86%     | 100%    | **65%**       |
| 360° (full)     | 64%     | 100%    | **25%**       |

`analyzer club <clip>` prints the per-phase table first for that reason. If the
downswing row is low, the shutter is the thing to change.

### Filming so the ball can be timed (Phase 11)

The ball is the only thing in the frame this system **observes** impact from.
Everything else infers it from the player's motion. What that costs you is four
things at capture time, and none of them is the shutter — a teed ball is not
moving, so no exposure smears it.

**Keep recording for two seconds after the strike.** Impact is read off the frame
the ball stops being visible, and a ball that has gone and a ball that something
moved in front of are the same picture at that instant. The only thing that tells
them apart is the absence lasting, so a clip that stops shortly after contact has
not made the check — `permanence` falls and the report says the instant is worth
less. Two seconds is plenty.

**One ball in the hitting area.** The ball is identified as the stationary,
ball-sized thing that vanishes, and a second one that also gets hit, a white tee
marker that gets picked up, or an alignment stick that rolls away is a rival the
picture cannot resolve. `analyzer ball` reports an **identification margin** and
drops it towards zero when more than one object departs.

**A surface the ball contrasts with, and enough of the frame spent on it.** The
one capture number that matters here is how many pixels across the ball is:

| ball radius on the sensor | what it means                                     |
| ------------------------- | ------------------------------------------------- |
| under 3 px                | refused — a disc that small has no shape left     |
| 3–5 px                    | found, but shape no longer separates it from turf |
| over 8 px                 | shape works as well as it is going to             |

That is measured rather than advised, and it is the main limit on the reference
footage in this repository: on `face-on/rory_face_on.mp4` the ball is **4.4 px**
in radius on a 576-wide phone clip, which is why the identification there rests
on the ball being in one place for three hundred frames rather than on it looking
round. Filming 1080p rather than 720p, or standing closer, buys this directly.

**Frame the ball, not just the player.** The search region is a disc of three
torso lengths around where the hands sit at address, which comfortably contains a
teed ball — but a clip cropped to the upper body does not contain one at all, and
`analyzer ball` reports that as `out_of_frame` rather than as a detection
failure, because it is a framing fix.

### Synchronising the two cameras

Nothing relates two cameras' clocks for you, and everything above two views
depends on that relation: triangulating a point is only meaningful for two views
of the same instant. `analyzer sync` recovers the offset from the footage itself
and reports how well it knows it, but two things about the capture bound what it
can do.

**Both cameras want the same frame rate, and both want to be fast.** An instant
located to the nearest frame carries an error of about a third of a frame
interval, and the two clips' errors add in quadrature, so the pair's floor is
set by the _coarser_ camera:

| Pair          | Best possible alignment |
| ------------- | ----------------------- |
| 30 + 30 fps   | 13.6 ms                 |
| 30 + 240 fps  | 9.7 ms                  |
| 240 + 240 fps | **1.7 ms**              |

Upgrading one camera of a 30 fps pair is worth a factor of sqrt(2) at most —
the slow one still contributes 9.6 ms on its own. Upgrading both is worth the
whole ratio.

A second consequence: both clips are smoothed with one window, because smoothing
them differently shifts the features the alignment reads. The coarser clip sets
it, and a 30 fps camera cannot support the default at all — so pairing a 240 fps
phone with a 30 fps one degrades the fast clip too.

**Get both slow-motion factors right, or discover which is wrong.** If both are
supplied and the fit returns a clock rate far from 1.0, one of them is wrong by
about that ratio: two real camera clocks do not differ by more than a fraction of
a percent. This is the one thing a second camera can measure that a single clip
cannot — nothing in one file records its own factor.

### Per-swing checklist

1. Both cameras recording before address.
2. Full swing to a held finish, then a pause before stopping.
3. Ball visible at address in both views.
4. Note club used and any range/course conditions.
5. **One swing per pair of clips.** Nothing in the system can tell that two
   recordings show the same swing — it aligns swing-shaped signals, and two
   different swings align perfectly happily. The only evidence is that their
   phase durations disagree, which shows up as a residual, and on two swings of
   similar tempo that evidence is weak. Do not pair a face-on clip of one swing
   with a down-the-line clip of another and expect to be told.

### Filming so a labelled set can exist (Phase 12)

**This is the one capture requirement no amount of filming by one person can
meet.** Everything else in this file asks for better footage. A labelled set asks
for more _golfers_, and the difference decides whether any accuracy figure can
ever be published here.

`analyzer labels` refuses to split a set with fewer than three players, and
refuses to permit a published number below eight. Those are not statistical
results; three is the fewest that fills train, validation and test with a
different person in each, and eight is the fewest that leaves more than one
person held out after the other two roles have taken theirs. Real work needs far
more. **Forty clips of one golfer produce no split at all.**

What to record, in order of what it buys:

1. **Different people.** Different builds, different tempos, different swing
   shapes, ideally different handicaps. A model that has seen six similar golfers
   has seen one golfer six times.
2. **More than one session per person**, on different days, at different places,
   in different light. A session is the unit that leaks: two swings from one
   sitting are near-duplicates, so they are held together and can never
   demonstrate that a model generalises.
3. **Clips with no swing in them.** A practice swing, a waggle, somebody walking
   into frame, a setup that is abandoned. A detector's most valuable behaviour is
   refusing to answer, and a set in which every clip contains a swing cannot
   measure it. Roughly one in six is enough.
4. **Consent, recorded.** Somebody else's swing is their likeness. Get permission
   in writing before a clip enters a set that will be trained on, and keep the
   `player_id` a pseudonym rather than a name — the schema treats it as an
   identifier and nothing in it needs to be a person's name.

When labelling, `--player` and `--session` are required and have no defaults. A
wrong value there is invisible and inflates every number measured afterwards, so
it is worth deciding the naming scheme before the first clip rather than during
it.

### Filming so a tempo can be compared (Phase 13)

The coaching engine will only say which side of a published range a swing's
tempo falls on if the measurement is finer than the distance to the range's
edge. That turns out to be a demanding requirement, and it is arithmetic rather
than opinion.

A tempo ratio is the backswing divided by the downswing, and the two share an
endpoint: moving the top by one frame lengthens one and shortens the other at
once. So the ratio's uncertainty is far larger than either duration's, and
largest exactly where the number is quoted — on a short downswing.

Measured on the reference amateur clip's own swing, varying nothing but the clock
(`scripts/benchmark_coaching.py --sweep resolution`):

| fps | uncertainty in the ratio | distance to the nearer band edge | outcome         |
| --- | ------------------------ | -------------------------------- | --------------- |
| 30  | 0.738                    | 0.371                            | **cannot tell** |
| 60  | 0.341                    | 0.371                            | reported        |
| 120 | 0.164                    | 0.371                            | reported        |
| 240 | 0.081                    | 0.371                            | reported        |

**A 30 fps recording supports no tempo comparison at all**; the crossover for
that swing is 56 fps. `analyzer coach` on the 30 fps face-on reference clip
produces zero findings, which is the correct answer and an unsatisfying one.

Two smaller things follow from the same arithmetic:

- **Declare the slow-motion factor, and know that it buys you the ratio and not
  the durations.** Absolute durations are refused on any clip whose factor was
  supplied rather than measured, because the number compared against a band in
  seconds would be a measurement multiplied by a guess. The ratio survives, since
  a factor that stretches both durations equally divides out of their quotient.
- **The takeaway is the weak event, and a fast capture is what moves it.** Tempo
  is measured from the takeaway, the top and impact, and the first of those is
  the hardest to place: on the tour footage this engine reads a tempo of 1.83:1
  where a tour swing is known for about 3:1, and the finding's own cited frames
  point at the takeaway rather than at the swing.

### Filming so two swings can be compared (Phase 16)

Comparing two recordings is the one thing here where a capture mistake does not
degrade the answer — it removes it. Three requirements, and each one is a gate
that refuses rather than a preference.

**Do not touch the tripod between the two swings.** This is the big one, and it
is much less forgiving than it sounds. `scripts/benchmark_compare.py --sweep
camera` films one unchanged synthetic swing from a series of positions:

| camera moved | address shoulder span | **shoulder turn reported** | what the engine does |
| ------------ | --------------------- | -------------------------- | -------------------- |
| 0°           | 1.05 torso            | **57.7°**                  | compares             |
| 10°          | 1.02                  | 55.1°                      | compares             |
| 20°          | 0.95                  | 47.7°                      | **refuses**          |
| 30°          | 0.83                  | **32.9°**                  | **refuses**          |

The body is identical in every row, and every row is still classified `face_on` —
the view detector's three labels decide whether a recording contains a
measurement at all, not whether two recordings contain the same one. A camera
moved thirty degrees round a player changes the most quoted number in golf
instruction by twenty-five degrees, with nothing in either clip to say so. The
comparison therefore refuses every projected quantity once the two clips' address
shoulder spans disagree by more than 10%.

**Start recording before the player addresses the ball.** The address phase is
where the camera view, the rotation baseline and the hand-path origin all come
from, and a clip without one has none of them — so every projected comparison
refuses, however good the rest of the footage is. This is not hypothetical: the
only same-player, same-position pair in this repository fails on exactly this,
because `data/rory/dtl/rory_dtl_2.mp4` begins at the takeaway.

**Shoot both clips at the same frame rate, above 60 fps, and slow neither or
both.** Each clip's own clock sets how finely it resolves a difference, and the
two brackets are added; a slow-motion factor is supplied rather than measured, so
a duration compared across clips whose factors were guessed is a difference of
two guesses. Ratios survive that; seconds do not.

What that buys, measured on two swings of genuinely different shape (`--sweep
resolution`), as the fraction of the swing whose difference clears the bracket:

| fps | of the swing that can be called different |
| --- | ----------------------------------------- |
| 30  | 39%                                       |
| 60  | 61%                                       |
| 120 | 80%                                       |
| 240 | 89%                                       |

The difference between the two swings is the same at every rate. What grows is
how much of it can be attributed to them rather than to the clock.

One thing no capture fixes: **nothing here can tell two swings by one player from
two different players.** Lengths are in each subject's own torso lengths and
durations on each clip's own clock, which makes the arithmetic meaningful across
subjects — it does not make the comparison meaningful. Whether the two clips are
comparable in that sense is the judgement of whoever chose them.

## Camera calibration (Phase 8)

Metric-scale 3D reconstruction requires calibration, and a calibration is only
as good as the capture it was measured from. The instructions below are not
style advice: each one is a thing the system checks and refuses on, and the
thresholds come from `scripts/benchmark_calibration.py`.

### The board

```bash
uv run --project python analyzer calibrate board board.png --squares 7x5 --square-mm 35
```

Print it at **100% scale** — no "fit to page" — mount it on something rigid and
flat, and then **measure one square with a ruler**. Pass what you measure, not
what you asked for:

```bash
uv run --project python analyzer calibrate camera board.mov --role face_on --square-mm 34.6
```

Every metric-scale claim this system ever makes descends from that one measured
length. A page silently scaled to 96% makes every future distance wrong by 4%,
and nothing anywhere will look amiss. It is the one step of the process no
software can check.

Generate the board here rather than downloading one. A pattern from a website is
a pattern whose dictionary, square count and marker layout are all guesses, and
OpenCV changed the Charuco layout in 4.6 — a board from an older generator
produces corners in the _wrong places_ rather than no corners at all, which is
much the worse failure.

### Capturing a camera

**Tilt the board. This is the one that is not obvious and the one that decides
whether the calibration is worth anything.** A board held square to the camera
cannot separate focal length from distance: a longer lens further away makes the
same picture. Such a capture fits beautifully — a fifth of a pixel of
reprojection error — and leaves the focal length wrong by up to 23%. Neither the
residual nor the uncertainty the fit reports about itself will tell you. Aim for
at least 20° of spread between your flattest and most oblique views; the system
refuses below that.

**Take the board to the corners of the frame.** Lens distortion is a function of
radius and is nearly nothing in the middle, so a centred capture fits its
distortion coefficients to almost no evidence and then applies them out at the
edge, where the swing is. Half the board leaving the frame is fine — that is
exactly what Charuco tolerates and a plain chessboard does not.

**Vary the distance.** The calibration otherwise describes the lens at one
working distance.

| what to vary    | why                                         | the bound      |
| --------------- | ------------------------------------------- | -------------- |
| tilt            | separates focal length from distance        | ≥ 20° spread   |
| position        | distortion is measurable only off-centre    | ≥ 35% of frame |
| distance        | one distance measures one working distance  | reported       |
| number of views | 8 well-spread views reach 0.08% focal error | ≥ 8            |

Twenty views buy 0.02% against eight views' 0.08%. Spread matters far more than
count, which is why the system measures spread directly.

**Calibrate at the setting you will film at.** A calibration belongs to a camera
_and_ a zoom, lens and capture resolution. Frame size is the only part of that a
file records, and the system refuses a calibration applied to footage of a
different size; zoom and lens changes leave no trace at all. Use `--notes` to
record what the file cannot.

### Capturing a stereo pair

Both cameras must see the board at the same instant, and two phones do not share
a clock. **Hold the board still at each position** and that stops mattering: the
pairing error is the clock uncertainty multiplied by how fast the board was
moving, so a still board makes it zero however badly the clocks are known.

Measured, the requirement is remarkably light — **three frames** of stillness,
which is a tenth of a second at 30 fps:

| board             | pairing error | baseline error | rotation error |
| ----------------- | ------------- | -------------- | -------------- |
| held 1 s          | 0.00 px       | 0.104%         | 0.031°         |
| held **3 frames** | 0.00 px       | 0.104%         | 0.031°         |
| held 2 frames     | —             | refused        | —              |
| never stops       | —             | refused        | —              |

So: move the board to a new position, pause for a moment, move again. Do not
wave it. A waved board produces images that are exactly as sharp and yields no
usable pairs, which is why the system measures this instead of trusting it.

Neither camera may move between the stereo capture and the swing — the
extrinsics describe where they stood, and nothing detects that one was nudged.

### Without a calibration

The system runs uncalibrated, and that is a supported state rather than a
failure: every measurement is reported as a statement about the image plane,
which is what it is, and no metric-scale 3D claim is made. What it costs is the
lens. An ordinary phone's main camera displaces a landmark near the frame edge
by about 170 px on a 1920x1080 frame, and every angle, distance and speed
measured above inherits that displacement with nothing in the numbers showing it.

Calibrating one camera removes it. It does **not** make one camera see depth: a
calibrated pixel is a direction, and how far along that direction anything sat is
exactly what the projection destroyed. That needs two calibrated views of the
same instant, which is what a stereo calibration plus an alignment buys.

### What a reconstruction still does not know: up, and the target line

Triangulating two calibrated views gives metres **centred on one of the
cameras**. That is enough for every quantity measured between two reconstructed
points — bone lengths, true joint angles, speeds in metres per second, and
rotations about the body's own spine axis — because none of those depends on the
frame they are expressed in.

What it does not give is a _scene_ frame, because nothing in the capture says
which way is up or which way the shot goes. The cost is specific: a **3D forward
spine tilt**, the posture angle a coach actually talks about, needs a vertical
and so is not produced.

**One extra shot would fix it, and the protocol does not currently ask for it:**
lay the calibration board flat on the ground in the hitting area, one edge along
the target line, and record a few seconds of it from both cameras before the
player steps in. The board's plane is then the ground, its normal is up, and its
own axes give the target line. If you are calibrating anyway, it costs ten
seconds — and it is worth capturing now even though this build does not yet read
it, because it cannot be recovered from footage afterwards.
