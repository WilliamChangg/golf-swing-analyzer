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

### Per-swing checklist

1. Both cameras recording before address.
2. Full swing to a held finish, then a pause before stopping.
3. Ball visible at address in both views.
4. Note club used and any range/course conditions.

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
