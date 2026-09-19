"""Comparing two recordings of a swing.

The only package here whose input is a **pair**. Everything below it measures one
clip, and the difficulty of this layer is that two clips differ for reasons that
have nothing to do with the two swings: the camera moved, the lens was corrected
in one and not the other, a slow-motion factor was supplied, or the two events
that bound a phase were each located a frame out.

Three modules, in dependency order:

* `normalise` -- the phase-relative clock, and how to read a signal on it.
* `diff` -- the gates a metric difference passes before it may be called one.
* `compare` -- the entry point, `compare(reference, target, config)`.

It reads `filtering`, `phases` and `biomechanics`, and borrows
`coaching.bracket` deliberately rather than deriving its own: what one recording
can resolve is one question with one answer, and two layers computing it
separately would eventually disagree about a number a reader sees in both.
"""

from analyzer.comparison import normalise
from analyzer.comparison.compare import ComparisonError, SwingInput, compare
from analyzer.comparison.normalise import NormalisationError, build_clock

__all__ = [
    "ComparisonError",
    "NormalisationError",
    "SwingInput",
    "build_clock",
    "compare",
    "normalise",
]
