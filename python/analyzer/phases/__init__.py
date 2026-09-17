"""Swing phase detection: where a clip stops being motion and becomes a swing.

The first golf-specific package in the engine. `architecture.md` confines that
knowledge to `phases`, `biomechanics` and `coaching`; everything underneath is
general computer vision that would serve any moving body.

Detection is deterministic and rule-based. The events are defined by the shape
of the hand-speed signal -- one minimum between two maxima, with the minimum at
the highest point the hands reach -- and that shape is not subtle enough to need
a model. Phase 12 trains one anyway and compares it against this detector on the
same held-out clips, which is only a meaningful comparison because a readable
baseline exists to compare against.

Two things are refused rather than guessed. A clip whose hand speed never stands
out from the speed a still subject shows produces no events at all, not events at
frame zero. And an event the capture cannot resolve -- because the frame rate
forced a smoothing window as long as the downswing being measured -- keeps its
frame but loses its confidence, through a factor computed from those two
durations rather than assumed.
"""

from analyzer.phases.detect import detect_phases
from analyzer.phases.signals import (
    HandTrack,
    SignalError,
    SwingSignals,
    choose_hand,
    swing_signals,
)

__all__ = [
    "HandTrack",
    "SignalError",
    "SwingSignals",
    "choose_hand",
    "detect_phases",
    "swing_signals",
]
