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

`face-on/rory_face_on.mp4` also carries the only **observed** impact in the
project: the ball is on the tee at frame 360 and gone at frame 361. Phase 4's
kinematic estimate is the only thing in the pipeline that can be checked against
a real event, and that check is recorded in the ADR.

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

### Settings

| Setting        | Recommendation               | Why                                                                                                          |
| -------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Frame rate     | **≥ 120 fps**, 240 preferred | Downswing lasts ~0.25 s; at 30 fps that is ~8 frames total, far too coarse to locate impact or measure tempo |
| Shutter speed  | **1/1000 s or faster**       | Limits motion blur on the club head, which classical shaft detection depends on                              |
| Resolution     | 1080p is sufficient          | Frame rate and shutter matter far more than resolution                                                       |
| Stabilisation  | **Off**                      | Digital stabilisation warps the frame non-rigidly, corrupting geometry                                       |
| Focus/exposure | Locked                       | Refocus mid-swing changes apparent scale                                                                     |

### Framing and environment

- Whole body plus the full club arc in frame for the entire swing, including
  follow-through. Clipping the top of the backswing loses the `top` event.
- Even, bright lighting. Avoid strong backlight, which silhouettes the player
  and destroys landmark confidence.
- Uncluttered background. Vertical lines behind the player (door frames, fence
  posts, window mullions) are the main false-positive source for Hough-based
  shaft detection.
- Clothing that contrasts with the background; avoid very loose garments, which
  hide joint positions.

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

## Camera calibration (Phases 8-9)

Metric-scale 3D reconstruction requires calibration. Print a **Charuco board**,
mount it rigidly flat, and capture 15-20 frames per camera with the board at
varied angles and distances filling different parts of the frame — corners
included, since that is where lens distortion is strongest.

For stereo extrinsics, both cameras must see the board **simultaneously** in a
set of frames.

Without this the system runs in `UNCALIBRATED` mode. That is a supported state,
not a failure: it simply means no metric-scale claims are made, and the system
will say so rather than producing a plausible-looking number.
