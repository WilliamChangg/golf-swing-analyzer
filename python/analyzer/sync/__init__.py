"""Relating two cameras' clocks to each other.

One swing, two recordings, two clocks that know nothing about each other. This
package produces the affine map between them, together with a measurement of how
well that map is determined -- which every phase above it inherits, because a
triangulated point is only as good as the claim that its two views are of the
same instant.

```
SwingSignals + SwingPhases (x2)
  ├─ anchors     events paired by name, or a person's picks; outliers dropped
  ├─ correlate   normalised cross-correlation of hand speed, masked by validity
  ├─ timemap     the affine fit, its refused rate, and its uncertainty
  └─ align       picks the map, checks it against what did not produce it
```

The mechanism is general -- an affine time map and a masked cross-correlation
would align any pair of recordings of any moving thing. What is golf-specific is
only the choice of signal: hand speed, and the four named swing events. So this
package sits with `phases` and `biomechanics` on the golf-aware side of the
codebase, while almost everything in it would survive the subject changing.
"""

from analyzer.sync.align import SyncInput, align
from analyzer.sync.anchors import AnchorError, ManualPick, event_anchors, manual_anchors
from analyzer.sync.correlate import SpeedTrack, cross_correlate
from analyzer.sync.timemap import TimeMapError, fit_time_map, quantisation_floor_s

__all__ = [
    "AnchorError",
    "ManualPick",
    "SpeedTrack",
    "SyncInput",
    "TimeMapError",
    "align",
    "cross_correlate",
    "event_anchors",
    "fit_time_map",
    "manual_anchors",
    "quantisation_floor_s",
]
