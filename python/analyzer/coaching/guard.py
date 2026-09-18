"""The guard: nothing may be said that the evidence does not contain.

A language model handed four findings and asked for a friendlier sentence will
produce a friendlier sentence, and somewhere in it will be a number nobody
measured. Not through malice and not often -- which is the problem, because a
system that is wrong once in fifty is a system whose output cannot be trusted
anywhere without checking it, and the checking is the expensive part.

So the checking is done here, deterministically, and it is the layer with the
final word. A candidate phrasing that fails is thrown away and the rule's own
sentence is used instead; there is no partial acceptance, no repair pass, and no
configuration that turns the guard off.

## It applies to this engine's own prose too

The rules' deterministic sentences are run through the same guard by the test
suite. This is not symmetry for its own sake: a template that could emit a number
absent from its own evidence is a bug of exactly the kind the guard exists to
catch, and discovering it in a rule's own words is cheaper than discovering it in
a model's. The guard is the contract for prose in this system, not a filter
bolted onto one optional component of it.

## What it checks

**Numbers.** Every numeral in the text must be present in the finding's evidence,
to the precision it was written at. "53 degrees" matches a measured 53.02;
"53.5 degrees" does not match anything and is rejected.

**Units.** A unit this engine does not produce is rejected wherever it appears --
miles per hour, yards, revolutions per minute. So is a unit it *can* produce but
did not produce here: metres are real in this system only when two calibrated
cameras triangulated something, and a sentence about a single-camera clip has no
business using them.

**Quantities.** Clubhead speed, ball speed, smash factor, launch angle, spin rate,
carry distance, and the words for where the ball went. None of these is measured
anywhere in this project, and a model that has read the internet knows what golf
analysis usually reports. This is the check that catches the confident sentence
containing no numbers at all.

**Causal claims.** "Causes", "leads to", "results in". A finding is a measurement
of a body at an instant; that one measured quantity produced another is a claim
about a mechanism nothing here tested.

## What it cannot check

It cannot tell whether the sentence is *about the right swing*. A model handed
two findings could describe the first using the second's numbers and pass every
check here, because every number would be in the evidence. That is why
`Finding.observation` is never replaced: the deterministic sentence survives in
the report, `phrased` sits beside it, and a reader who wants to know what was
measured reads the one this module did not have to approve.

The spelled-out-number table is finite, and starts at two. "One" is excluded
because in English it is usually a pronoun, which means a model writing "one
degree of tilt" gets through. Both holes are real; neither is closed by pretending
the guard is a proof.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from analyzer.contracts.coaching import Finding
from analyzer.contracts.metrics import MetricUnit


@dataclass(frozen=True)
class Offence:
    """One thing in a candidate that the evidence does not support."""

    offence: str
    token: str


@dataclass(frozen=True)
class Allowance:
    """Everything a phrasing of one finding is permitted to say.

    `numbers` is the closed set of quantities that may appear. It is built from
    the finding rather than from the report, so a model cannot borrow a number
    from a different finding and have it accepted here -- each candidate is
    checked against the evidence of the finding it is a phrasing of.

    `frames` is kept apart from it, and the separation matters more than it
    looks. A metric measured over the address phase cites a hundred and sixty
    consecutive frame indices; folding those into the allowance would permit
    every small integer in the language, and "ninety degrees of shoulder turn"
    would then be accepted on a clip that happens to have a frame 90. A frame
    index is quotable only as a frame, and the check requires the word.
    """

    numbers: tuple[float, ...]
    frames: tuple[float, ...]
    units: frozenset[str]

    def permits(self, value: float, tolerance: float) -> bool:
        # The epsilon is for the boundary, not for leniency. A value printed to
        # three decimals differs from the measured one by at most exactly half a
        # unit in the last place, which is exactly the tolerance -- and a binary
        # float lands either side of that equality by a few parts in 10^16. A
        # strict comparison would reject a correctly rounded number now and then,
        # non-reproducibly, which is the worst way for a guard to be wrong.
        return any(abs(value - allowed) <= tolerance + 1e-9 for allowed in self.numbers)

    def permits_frame(self, value: float) -> bool:
        return any(value == allowed for allowed in self.frames)


# Unit vocabulary, by the metric unit that licenses it. A word not licensed by
# anything in the evidence is an offence even when this engine can produce it
# elsewhere: metres exist in this system only downstream of a stereo
# triangulation, and a sentence about one camera must not reach for them.
_UNIT_WORDS: dict[MetricUnit, frozenset[str]] = {
    MetricUnit.DEGREES: frozenset({"degree", "degrees", "deg"}),
    MetricUnit.SECONDS: frozenset({"second", "seconds", "sec", "secs"}),
    MetricUnit.RATIO: frozenset(),
    MetricUnit.TORSO_LENGTHS: frozenset({"torso", "torsos"}),
    MetricUnit.TORSO_LENGTHS_PER_S: frozenset({"torso", "torsos"}),
    MetricUnit.METRES_PER_S: frozenset({"metre", "metres", "meter", "meters", "m/s"}),
}

# Always available: a frame is an index into the video rather than a measured
# quantity, and every piece of evidence carries frames.
_ALWAYS_ALLOWED_UNITS = frozenset({"frame", "frames"})

# Units no part of this system produces, in any configuration. Matched as whole
# words, so "ft" in "left" and "in" in "inside" cannot trigger them.
_IMPOSSIBLE_UNITS: frozenset[str] = frozenset(
    {
        "mph",
        "kph",
        "km/h",
        "kmh",
        "yard",
        "yards",
        "yd",
        "yds",
        "foot",
        "feet",
        "ft",
        "inch",
        "inches",
        "mm",
        "cm",
        "rpm",
        "percent",
        "%",
        "joule",
        "joules",
        "newton",
        "newtons",
        "watt",
        "watts",
        "kg",
        "lb",
        "lbs",
    }
)

# The whole unit vocabulary, so a licensed-elsewhere unit can be told apart from
# an unrecognised word. Anything here that the evidence does not license is an
# offence; anything not here at all is ordinary English and is left alone.
_KNOWN_UNITS: frozenset[str] = frozenset().union(*_UNIT_WORDS.values()) | _ALWAYS_ALLOWED_UNITS

# Quantities nothing in this project measures. A model asked about a golf swing
# reaches for these because they are what golf analysis reports; none of them is
# recoverable from what this system records, and several would need a launch
# monitor standing next to the camera.
_UNMEASURED_QUANTITIES: tuple[str, ...] = (
    "clubhead speed",
    "club head speed",
    "club-head speed",
    "swing speed",
    "ball speed",
    "smash factor",
    "launch angle",
    "spin rate",
    "backspin",
    "sidespin",
    "carry distance",
    "total distance",
    "yardage",
    "handicap",
    "attack angle",
    "angle of attack",
    "face angle",
    "swing plane angle",
    "strike quality",
    "centre of pressure",
    "center of pressure",
    "weight transfer",
    "ground force",
    "ground reaction",
)

# Where the ball went. This system sees a ball at rest and the frame it stops
# being there; it never sees flight, so every one of these is an outcome nothing
# observed. Restricted to words that are not ordinary English elsewhere in a
# swing description -- "draw", "push" and "pull" are left out deliberately,
# because they are verbs here far more often than ball flights.
_BALL_FLIGHT: tuple[str, ...] = (
    "slice",
    "slices",
    "sliced",
    "slicing",
    "hook",
    "hooks",
    "hooked",
    "hooking",
    "shank",
    "shanks",
    "shanked",
    "fade",
    "fades",
    "faded",
    "duck hook",
    "ball flight",
    "straighter",
    "further",
    "farther",
)

_CAUSAL_CLAIMS: tuple[str, ...] = (
    "causes",
    "caused by",
    "causing",
    "leads to",
    "results in",
    "responsible for",
    "which is why",
    "that is why",
    "due to",
    "so you",
)

# Spelled-out numbers, which a digit scanner would walk straight past. Starts at
# two: see the module docstring.
_WORD_NUMBERS: dict[str, float] = {
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "fifteen": 15.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
    "hundred": 100.0,
    # Multipliers, which are quantities written as adjectives. "Twice the tour
    # downswing" is a measurement claim with no digits in it.
    "twice": 2.0,
    "double": 2.0,
}

_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_UNITS_WORD = {
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
}

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")(?:[- ](\w+))?\b"
)
_WORD_RE = re.compile(r"[a-z%/]+")


def allowance_for(finding: Finding) -> Allowance:
    """Every number and unit word a phrasing of this finding may use."""
    numbers: list[float] = [finding.value, finding.bracket, finding.margin]
    if finding.band_low is not None:
        numbers.append(finding.band_low)
    if finding.band_high is not None:
        numbers.append(finding.band_high)
    if finding.source.year is not None:
        numbers.append(float(finding.source.year))
    if finding.source.sample_size is not None:
        numbers.append(float(finding.source.sample_size))

    frames: list[float] = []
    units = set(_ALWAYS_ALLOWED_UNITS)
    for item in finding.evidence:
        numbers.append(item.value)
        if item.uncertainty is not None:
            numbers.append(item.uncertainty)
        numbers.extend(item.timestamps_s)
        frames.extend(float(frame) for frame in item.frames)
        units |= _UNIT_WORDS.get(item.unit, frozenset())
    units |= _UNIT_WORDS.get(finding.unit, frozenset())

    return Allowance(
        numbers=tuple(value for value in numbers if math.isfinite(value)),
        frames=tuple(frames),
        units=frozenset(units),
    )


def _tolerance(written: str) -> float:
    """Half a unit in the last place the candidate wrote to.

    A model that writes "53" has claimed the value is nearer 53 than 52 or 54,
    and a measured 53.02 satisfies that. One that writes "53.02" has claimed two
    decimals, and must have them right. Rounding a number down to fewer digits is
    presentation; inventing digits is not.
    """
    if "." not in written:
        return 0.5
    return 0.5 * 10.0 ** -len(written.split(".", 1)[1])


def _word_numbers(text: str) -> list[tuple[str, float]]:
    """Spelled-out quantities, including hyphenated compounds like 'forty-five'."""
    found: list[tuple[str, float]] = []
    for match in _WORD_NUMBER_RE.finditer(text):
        head, tail = match.group(1), match.group(2)
        value = _WORD_NUMBERS[head]
        if head in _TENS and tail in _UNITS_WORD:
            value += _UNITS_WORD[tail]
        found.append((match.group(0), value))
    return found


def _reads_as_frame(text: str, span: tuple[int, int]) -> bool:
    """Whether a numeral is being used as a frame index rather than a quantity.

    Decided by looking for the word beside it, which is crude and is the point:
    a frame index is only quotable when the sentence says it is one, so "frame
    90" is checkable against the clip and a bare "90" is not.
    """
    start, end = span
    window = text[max(0, start - 14) : min(len(text), end + 14)]
    return "frame" in window


def inspect(text: str, allowance: Allowance) -> tuple[Offence, ...]:
    """Every way a candidate phrasing exceeds what the evidence supports.

    Empty means it may be used. The order is the order the checks run in, which
    is deliberate: a report showing which check fires most often is the only
    evidence anybody has about whether a phrasing model is worth having.
    """
    lowered = text.lower()
    offences: list[Offence] = []

    # Ratios first, and their digits are consumed here: "3:1" means the quotient,
    # and checking the 3 and the 1 separately would demand that a finding's
    # evidence contain the integer 1.
    consumed: set[tuple[int, int]] = set()
    for match in _RATIO_RE.finditer(lowered):
        consumed.add(match.span())
        left, right = float(match.group(1)), float(match.group(2))
        if right == 0.0:
            offences.append(Offence("number not in evidence", match.group(0)))
            continue
        tolerance = _tolerance(match.group(1)) / right
        if not allowance.permits(left / right, tolerance):
            offences.append(Offence("number not in evidence", match.group(0)))

    for match in _NUMBER_RE.finditer(lowered):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        written = match.group(1)
        value = float(written)
        if allowance.permits(value, _tolerance(written)):
            continue
        if _reads_as_frame(lowered, match.span()) and allowance.permits_frame(value):
            continue
        offences.append(Offence("number not in evidence", written))

    for token, value in _word_numbers(lowered):
        if not allowance.permits(value, 0.5):
            offences.append(Offence("number not in evidence", token))

    words = set(_WORD_RE.findall(lowered))
    for unit in sorted(words & _IMPOSSIBLE_UNITS):
        offences.append(Offence("unit this system cannot measure", unit))
    for unit in sorted((words & _KNOWN_UNITS) - allowance.units):
        offences.append(Offence("unit not in this finding's evidence", unit))

    for phrase in _UNMEASURED_QUANTITIES:
        if phrase in lowered:
            offences.append(Offence("quantity nothing here measures", phrase))
    for phrase in _BALL_FLIGHT:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            offences.append(Offence("claim about where the ball went", phrase))
    for phrase in _CAUSAL_CLAIMS:
        if phrase in lowered:
            offences.append(Offence("causal claim nothing here tested", phrase))

    return tuple(offences)


def passes(text: str, finding: Finding) -> bool:
    """Whether this text may stand in for a finding's own sentence."""
    return not inspect(text, allowance_for(finding))
