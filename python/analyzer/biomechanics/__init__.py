"""Biomechanics: turning a filtered swing into measured, qualified numbers.

The second golf-specific package, sitting on top of `phases`. Everything below
the two of them is general computer vision that would serve any moving body;
from here up, the code knows what a golf swing is and what is worth measuring
about one.

The output is a list of `Metric`s, and the shape of that type is the point of
the package. A number leaves here carrying a unit, the frames it came from, a
decomposed confidence, a methodology in words, and -- the field that does the
most work -- a **basis** saying what kind of claim it is. Nothing measured in
this phase sees three dimensions. A shoulder turn computed from how much the
shoulder line shortened is a real measurement of a real projection, and it is
not a measurement of the body's rotation; the difference is carried in the type
rather than in a footnote, so a consumer that wants to say "your shoulders
turned 85 degrees" has to go past something that says otherwise.

Three things here refuse rather than guess, and each refusal is reported with
its reason:

* a clip with no detected swing yields no metrics at all, because every metric
  is anchored to a swing event;
* rotation by foreshortening is refused when the body line was at its widest
  somewhere other than address, which is what a down-the-line recording looks
  like and means the baseline is wrong rather than slightly off;
* the lead and trail arm angles are refused when the recording does not show
  which arm leads, instead of assuming the player is right-handed.

Phase 6 adds explicit coordinate systems and view tagging, Phase 8 calibration,
and Phase 9 triangulation -- at which point the angles here can be replaced by
ones that are about the body.
"""

from analyzer.biomechanics.body import Body, BodyError, body_from
from analyzer.biomechanics.compute import compute_metrics, known_metrics
from analyzer.biomechanics.registry import REGISTRY, Anchor, MetricDefinition, definition

__all__ = [
    "REGISTRY",
    "Anchor",
    "Body",
    "BodyError",
    "MetricDefinition",
    "body_from",
    "compute_metrics",
    "definition",
    "known_metrics",
]
