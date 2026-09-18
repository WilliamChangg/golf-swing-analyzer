# ADR-0018: A video element cannot be asked for a frame, so the ask is measured and the answer is checked

**Status:** Accepted
**Date:** 2026-09-18
**Phase:** 14 — Desktop UI

## Context

Everything this project has built since Phase 4 reports **frame indices**.
Impact at frame 46. The address baseline over frames 0 to 12. A finding citing
frames 13 to 38. Phase 14 puts a picture behind those numbers, and discovers
that the one thing a `<video>` element cannot be asked for is a frame. It is
asked for a **time**, and its decoder shows whichever frame is being displayed
then.

So something has to convert. The obvious conversion is `frame / fps`, and it is
the same mistake Phase 1 exists to prevent one layer down: on variable-rate
footage the interval between frames is not constant, so dividing by an average
accumulates error until the frame under the playhead is not the frame the panel
beside it is talking about. Nothing about the resulting picture looks wrong.

Measured over this repository's own clips, with `scripts/benchmark_seek.py`, the
naive map is worse than "imprecise on phone footage". It fails on clips nobody
would suspect, for three different reasons:

| clip                | vfr | declared fps | measured fps | `frame / declared fps` |
| ------------------- | --- | ------------ | ------------ | ---------------------- |
| cfr_30fps.mp4       | no  | 30.000       | 30.000       | 0 / 60                 |
| vfr_30_to_15fps.mp4 | yes | 23.684       | 22.759       | **39 / 45** (±6)       |
| PW_face-on.mp4      | no  | 27.470       | 30.000       | **56 / 68** (±5)       |
| iron_dtl.mp4        | yes | 30.063       | 30.063       | 0 / 96                 |
| rory_face_on.mp4    | no  | 30.006       | 30.000       | **651 / 652** (±1)     |

Three distinct failures, only one of which is the one everybody expects:

- **`vfr_30_to_15fps.mp4`** is genuinely variable, and drifts 1.7 frames by the
  end. This is the case the naive map is known to fail.
- **`PW_face-on.mp4` is constant-rate and still misses by five frames.** Its
  frames are evenly spaced at exactly 30 fps; its container **declares 27.470**.
  A rate that is merely stated wrong is indistinguishable from a rate that
  varies, and this clip is one of the two the whole project has been measured
  on since Phase 4.
- **`rory_face_on.mp4` misses on 651 of 652 frames by exactly one.** Its
  declared rate is right to four decimal places. `i / 30.006` lands a few
  hundred nanoseconds _below_ the presentation time of frame `i`, and a frame
  boundary has another frame on the other side of it.

That last row is the second half of the problem, and it survives getting the
rate exactly right: **a frame's own timestamp is a boundary.** Asking a decoder
to seek to it is asking it to resolve a tie, and which way it falls is decided
by rounding nobody controls — the container's rational time base, the double the
time is carried as, and the decoder's own comparison.

## Decision

**Seek to the midpoint of the interval a frame is displayed for, computed by the
engine from the presentation timestamps it read. Then look up the frame the
browser reports it actually painted, and show the difference.**

Three parts, and the third is the one that makes the other two honest.

### The times come from the engine

`SeekIndex` carries `timestamps_s` and `seek_targets_s` for every frame. Both
are computed in Python from the packet index `probe` already reads, and neither
is derived in the browser. The UI has no opinion about when a frame is shown;
it looks the answer up.

Not cached, measured: the probe is 72.6 ms against seconds of pose extraction
for the same clip, so a cache would add a schema to version for a saving nothing
can detect. The same conclusion Phase 3 reached about the filter.

### The target is a midpoint, not a timestamp

Frame _i_ is displayed from `timestamps_s[i]` until `timestamps_s[i + 1]`. The
midpoint is the furthest point from both boundaries, so it is the target that
survives every source of rounding at once. The final frame borrows the last
observed interval, because how long the last frame is displayed is not recorded
anywhere — the same assumption `VideoTiming.duration_s` already makes, repeated
rather than reinvented.

### The result is checked, not assumed

`requestVideoFrameCallback` reports the presentation time of the frame actually
painted. The player looks that up and displays the difference from the frame it
asked for. `currentTime` is **not** a substitute: browsers commonly leave it at
the value that was requested, so reading it back confirms the seek against
itself.

Where the browser has no such API, the player says **"seeks are unverified"**
rather than showing a residual of zero it never measured.

This is the part worth defending. A player that seeks by `frame / fps` and
displays the frame number it asked for looks correct on every clip in the table
above, including the two it is five frames wrong on. Displaying what actually
landed is what makes this one capable of being wrong out loud.

## Consequences

### What the check found, that the design did not predict

**Chromium reports media time in whole microseconds.** A frame whose
presentation time is 0.13333333… seconds comes back as `mediaTime = 0.133333` —
a third of a microsecond _below_ the timestamp the container holds. Looked up
strictly, that frame falls in the previous one's interval, and the player
reports a spurious one-frame miss on roughly half the frames of any clip whose
frame times are not whole microseconds, which is every clip at 30 fps.

`frameAt` therefore allows one microsecond, and that allowance is the resolution
of the number being looked up rather than a tolerance somebody chose: it is four
thousand times smaller than one frame at 240 fps, so it can never merge two
frames. Found by the browser test, not by reasoning.

**A media element will not seek at all in a resource that does not advertise
byte ranges.** Served without `Accept-Ranges`, Chromium reports `seekable` as
empty and silently clamps every `currentTime` assignment back to zero. The clip
loads, reports a duration and buffers end to end; it just never moves, which
looks exactly like a broken player rather than like a missing response header.
Tauri's asset protocol implements ranges, so this surfaced only in the test
harness — and the harness now reproduces the real protocol's behaviour rather
than working around it.

**The last frame of a clip can be unreachable.** Its seek target lies past the
end of the media, because how long it is displayed is not recorded; the browser
clamps to the declared duration, and on `vfr_30_to_15fps.mp4` the declared
duration _equals the last frame's own presentation time_. There is therefore no
time inside that media at which its final frame is displayed, by any map. Not a
defect this code can fix, and the player reports it as a one-frame residual
rather than hiding it.

### What could not be verified, and how

`scripts/benchmark_seek.py` also asks OpenCV to perform the seeks, and that
column **measures OpenCV**. `cv2.VideoCapture.set(CAP_PROP_POS_MSEC)` lands a
frame early on a sizeable minority of seeks once the decoder has been read from
— the same inaccuracy Phase 1 found and works around with `_scan_to` when
seeking by index. A reference implementation that is itself wrong cannot tell
two maps apart, and on three of five clips it does not.

So the only non-circular evidence is a real player, and it is
`e2e/player.spec.ts`: Chromium, the real fixture bytes, and every measured
target landing on the frame it asks for, with the naive map measured alongside
in the same browser on the same clip in the same run.

### The permission this costs

Phases 0 and 1 both recorded that the WebView never reads a file — the picker
returns a path, the WebView hands it to Rust, and the engine does the reading.
A `<video>` element **is** the WebView reading a file, and there is no version of
a frame-accurate player that avoids it.

So the line is crossed, as narrowly as the platform allows:

- `assetProtocol.scope` in `tauri.conf.json` is **empty**. At startup the
  WebView can read nothing.
- A path is added to the scope one file at a time, and only by
  `commands::choose_clip`.
- **There is deliberately no command that takes a path and grants access to
  it.** That is what carries the guarantee: the frontend cannot name a file it
  wants read. A grant has to be tied to something the caller could not have
  fabricated, so `choose_clip` opens its own dialog in Rust and admits exactly
  what came back.

`dialog:allow-open` is still granted, and that was reconsidered rather than
assumed. The WebView keeps it for the pickers that choose footage **the engine**
reads and the WebView does not — a second clip to align against, a directory of
board images — and none of those paths reach the asset scope, because no command
exists that would put one there. Removing it would stop the WebView popping up a
native dialog, which is a different and much smaller property than the one this
section is about.

`tauri-plugin-fs` is still absent, and the WebView still cannot spawn a process.

### Open

- The seek map is verified in Chromium. The app ships on **WKWebView**, and
  `tauri-driver` is not wired up, so the packaged binary's player is not covered
  by an automated test. The residual display is the mitigation and not a
  substitute: it means a WKWebView that behaves differently is visible to the
  person using it, rather than silent.
- `requestVideoFrameCallback` is assumed present in the shipping WebView and is
  not required. Where it is missing the player still seeks and says it cannot
  check.
