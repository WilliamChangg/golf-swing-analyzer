"""Coaching: turning measured quantities into things that may be said out loud.

The top of the engine, and the first layer whose output is a **claim** rather
than a number. Everything below measures; this decides what those measurements
entitle anybody to conclude, which is a different and harder question.

Three properties hold everywhere in here, and each of them exists because the
obvious alternative produces a confident sentence nobody can check:

* **A finding cites its evidence.** Not "was derived from" -- cites, with the
  metric, the frames it was measured on, and the methodology, so the claim can be
  put back on the picture it came from.
* **A borrowed threshold carries its provenance, and mostly that disqualifies
  it.** A number measured on marker-based capture in three dimensions is not a
  threshold on a quantity inferred from how much a line shortened in one
  photograph. Eight of this registry's eleven rules are refused on every clip
  this system can record, and the refusals are reported by name.
* **No number may be said that the evidence does not contain.** Enforced by a
  deterministic guard that outranks the optional language layer and is applied to
  this engine's own sentences as well.

There is no score. A swing does not come out of here with a grade, and the
absence is deliberate -- a single number summarising it would need a scale
relating degrees of turn to seconds of tempo, and nobody has measured one.
"""

from analyzer.coaching.engine import coach
from analyzer.coaching.guard import Allowance, Offence, allowance_for, inspect, passes
from analyzer.coaching.registry import REGISTRY, Rule, RuleKind, known_rules, rule

__all__ = [
    "REGISTRY",
    "Allowance",
    "Offence",
    "Rule",
    "RuleKind",
    "allowance_for",
    "coach",
    "inspect",
    "known_rules",
    "passes",
    "rule",
]
