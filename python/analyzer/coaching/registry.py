"""The rule registry: every conclusion this engine knows how to reach.

A rule is a declaration, not a function. It names the measurement it is about,
the anchor it wants it at, the band it compares against, and -- the field that
decides whether it can ever fire -- the source that band came from. The engine
evaluates all of them the same way, so a rule cannot arrive with its own
private idea of when a number is close enough.

That matters more here than it did for the metric registry one layer down.
A metric that is computed wrongly produces an angle that is wrong; a rule that is
evaluated leniently produces a **sentence a person acts on**, and nothing about
the sentence records that the comparison was generous.

## The registry is mostly refusals, and that is the honest inventory

Twelve rules are declared here. Five can reach a comparison at all -- three
timing rules and the two that compare a clip against itself -- and seven are
refused before a clip is even looked at, each for a stated reason, on every
recording this system can make. They are not aspirational entries awaiting a
later phase: `rotation.shoulder_turn_top` is refused because the number it would
compare against has no measurement protocol behind it, and no phase of this
project can supply one.

On the reference footage only two of the five have ever produced a finding, and
both are timing rules. The same-clip rules are refused there for want of a
measured uncertainty on a projected angle, which is a gap in what Phase 6
quantifies rather than in what they compare.

They are here because omitting them would be the less honest choice. Every golf
app on a phone displays a shoulder turn against ninety degrees, and a system that
quietly did not would look like it had forgotten to, rather than like it had
looked at the number and declined.

## The two kinds of rule

`BAND` compares one measurement against a range somebody else published. It is
the kind that needs a citation, and the kind that mostly cannot fire.

`SAME_CLIP` compares one measurement against **itself** at a second anchor in the
same clip. It borrows nothing, so no population and no measurement protocol has
to match -- what it has to clear is the two measurements' own uncertainty, which
makes it the only kind of rule here whose threshold this project measured itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from analyzer.contracts.coaching import ThresholdMethod, ThresholdSource
from analyzer.contracts.metrics import CameraView, MetricBasis, MetricName
from analyzer.contracts.phases import SwingEvent, SwingPhase

# --------------------------------------------------------------------------
# Sources
#
# Declared once and shared, so that two rules resting on the same publication
# cannot drift into describing it differently. Ordered by how much standing the
# number has, which is also the order of how many rules each one lets fire.
# --------------------------------------------------------------------------

TOUR_TEMPO = ThresholdSource(
    citation="John Novosel, Tour Tempo: Golf's Last Secret Finally Revealed (Doubleday, 2004)",
    year=2004,
    population="Tour professionals, filmed for broadcast and analysed frame by frame",
    sample_size=None,
    method=ThresholdMethod.TWO_D_VIDEO,
    measures=(
        "Frames from the start of the takeaway to the top, and from the top to "
        "impact, counted on 30 fps video. The three patterns reported are 18/6, "
        "21/7 and 24/8 frames."
    ),
    permitted_bases=[MetricBasis.TEMPORAL],
    filmed_from=None,
    note=(
        "The one source in this registry whose measurement this engine can "
        "reproduce: a frame count is a frame count, and a duration does not "
        "depend on where the camera stood. Its bands here are derived from its "
        "own published frame counts by division, and each rule states the "
        "division it did, so a reader can check the band rather than trust it.\n\n"
        "It reports frame counts and not a ratio, which is the more careful of "
        "the two and worth preserving: 3:1 exactly is not something a 30 fps "
        "measurement of a half-second downswing can mean."
    ),
)

SAME_SWING = ThresholdSource(
    citation="This engine. No external threshold.",
    year=None,
    population=(
        "This swing, and nobody else's. A same-clip comparison says what changed "
        "between two instants of one recording; it carries no claim about how "
        "that compares with anybody."
    ),
    sample_size=None,
    method=ThresholdMethod.SAME_CLIP,
    measures="The same quantity, by the same method, at two anchors of one clip.",
    # Every basis, because nothing is being borrowed: two measurements made the
    # same way by the same camera are comparable with each other whatever they
    # are measurements *of*. What the projection costs is the size of the
    # difference, not its direction, and the rules using this say so.
    permitted_bases=[
        MetricBasis.TEMPORAL,
        MetricBasis.IMAGE_PLANE,
        MetricBasis.PROJECTED_ANGLE,
        MetricBasis.FORESHORTENED_ANGLE,
        MetricBasis.SPATIAL,
    ],
    filmed_from=None,
    note=(
        "The threshold is the two measurements' own combined uncertainty, so a "
        "change is reported only when it is larger than what the measurements "
        "could have produced by themselves."
    ),
)

TORSO_PELVIS_SEPARATION_3D = ThresholdSource(
    citation="Three-dimensional studies of torso and pelvis rotation in the golf swing",
    year=None,
    population="Varies by study; commonly grouped by clubhead speed or handicap",
    sample_size=None,
    method=ThresholdMethod.THREE_D_MOTION_CAPTURE,
    measures=(
        "The angle between the upper-torso and pelvis transverse axes, tracked in "
        "three dimensions through the swing."
    ),
    permitted_bases=[MetricBasis.SPATIAL],
    filmed_from=None,
    note=(
        "The mirror image of the convention sources, and the pair is the point. "
        "This literature has a measurement protocol and reports group means with "
        "spreads, which is not a range that separates one swing from another; "
        "this project has not read a figure it can cite and will not derive one. "
        "So the quantity this engine could honestly compare against has no number "
        "here, and the quantity that has a number has no protocol."
    ),
)

FULL_TURN_CONVENTION = ThresholdSource(
    citation="Golf instruction convention. No attributable measurement.",
    year=None,
    population="Unstated",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures=(
        "Nothing that is defined. 'A full shoulder turn' is taught as roughly a "
        "right angle without a stated axis, reference pose or instrument."
    ),
    permitted_bases=[],
    note=(
        "The most-quoted number in golf instruction and the least defensible. "
        "Ninety degrees of what, measured from where, against which reference "
        "frame, is not stated anywhere this project could find, so there is no "
        "quantity for it to be a threshold on."
    ),
)

X_FACTOR_CONVENTION = ThresholdSource(
    citation="Jim McLean, 'The X-Factor', Golf Magazine (1992), and the instruction following it",
    year=1992,
    population="Tour professionals, unstated in number",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures=(
        "Separation between the shoulder line and the hip line at the top, "
        "popularised as a figure near forty-five degrees. The article introduced "
        "the idea rather than a protocol for measuring it."
    ),
    permitted_bases=[],
    note=(
        "Later three-dimensional work measures a quantity of the same name in a "
        "defined way and gets different numbers, which is what happens when a "
        "measurement is given to a term rather than a term to a measurement. "
        "Neither number belongs against this engine's foreshortened estimate."
    ),
)

HEAD_STILL_CONVENTION = ThresholdSource(
    citation="Golf instruction convention. No attributable measurement.",
    year=None,
    population="Unstated",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures=(
        "Nothing measurable. 'Keep your head still' names a direction to move in "
        "and no quantity at all -- there is no published number, in any unit, for "
        "how far a head may travel."
    ),
    permitted_bases=[],
    note=(
        "Kept in the registry as the clearest case of the distinction this "
        "engine is built around: there is advice here, and there is no threshold "
        "here, and the two are not the same kind of thing."
    ),
)

POSTURE_CONVENTION = ThresholdSource(
    citation="Golf instruction convention. No attributable measurement.",
    year=None,
    population="Unstated",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures=(
        "'Keep your posture through impact' and 'stay in your tilt', taught "
        "without a reference angle, an axis or an instrument."
    ),
    permitted_bases=[],
    note=(
        "A down-the-line spine angle is genuinely visible on video and genuinely "
        "worth looking at, which is what makes this one tempting. What does not "
        "exist is a published number it can be scored against."
    ),
)

STRAIGHT_ARM_CONVENTION = ThresholdSource(
    citation="Golf instruction convention. No attributable measurement.",
    year=None,
    population="Unstated",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures="'Keep the lead arm straight at the top', taught without a tolerance.",
    permitted_bases=[],
    note=(
        "The engine's lead arm angle is a projected shoulder-elbow-wrist angle, "
        "which is at most the real one and equals it only when the whole arm lies "
        "in the image plane. At the top it rarely does, from either camera."
    ),
)

EARLY_EXTENSION_CONVENTION = ThresholdSource(
    citation="Golf instruction convention. No attributable measurement.",
    year=None,
    population="Unstated",
    sample_size=None,
    method=ThresholdMethod.CONVENTION,
    measures=(
        "'Early extension' -- the hips moving towards the ball through the "
        "downswing -- named as a fault without a distance that makes it one."
    ),
    permitted_bases=[],
    note=(
        "Measurable in principle from a down-the-line camera, in this engine's "
        "torso lengths. No source publishes a threshold in any length unit, let "
        "alone that one."
    ),
)


class RuleKind(StrEnum):
    """What sort of comparison a rule makes. See the module docstring."""

    BAND = "band"
    SAME_CLIP = "same_clip"


@dataclass(frozen=True)
class Rule:
    """One declared conclusion, independent of any particular swing."""

    rule_id: str
    title: str
    kind: RuleKind
    metric: MetricName
    source: ThresholdSource
    subject: str
    """The thing being talked about, as a noun phrase: 'the backswing'.

    Used to build the observation sentence. Written here rather than in the
    engine so that a rule's words and its arithmetic are declared together and
    cannot describe different quantities.
    """

    event: SwingEvent | None = None
    phase: SwingPhase | None = None
    band_low: float | None = None
    band_high: float | None = None
    band_derivation: str = ""
    """How the band was obtained from the source's published figures, in words.

    Never omitted where a band exists. A range that appeared without a derivation
    is a range somebody chose, and the reader cannot tell the difference
    afterwards.
    """

    no_threshold: str = ""
    """Why this rule has no band, when it has none. Empty when it has one."""

    compare_event: SwingEvent | None = None
    """The second anchor, for a `SAME_CLIP` rule."""

    compare_phase: SwingPhase | None = None
    requires_view: CameraView | None = None
    """A camera position this rule needs, where it needs one.

    Distinct from the metric layer's own view gate, which decides whether a
    quantity is *measurable* from a view. This decides whether a comparison is
    *meaningful* in it -- hip travel towards the ball is early extension seen
    down the line and a slide along the target line seen face-on, and only one of
    those is what the rule is about.
    """


# Every rule the engine knows, in the order a report reads best: what the clock
# says first, because it is the only family that can currently answer; then the
# body, in the order the metric registry lists it.
_RULES: tuple[Rule, ...] = (
    # --- timing ----------------------------------------------------------
    Rule(
        rule_id="tempo.ratio",
        title="Tempo against the tour band",
        kind=RuleKind.BAND,
        metric=MetricName.TEMPO_RATIO,
        source=TOUR_TEMPO,
        subject="the ratio of backswing to downswing",
        band_low=2.43,
        band_high=3.80,
        band_derivation=(
            "The source's three patterns are 18/6, 21/7 and 24/8 frames, each of "
            "which is exactly 3:1. A ratio is not what it measured, though: it "
            "counted frames, and the top is one frame wide. Moving the top by a "
            "single frame turns 18/6 into 17/7 or 19/5 -- 2.43 or 3.80 -- and the "
            "band here is the widest that its own published counts permit under "
            "that one frame of ambiguity. The exact 3:1 the number is quoted as "
            "is a ratio of two integers, not a measurement with a resolution."
        ),
    ),
    Rule(
        rule_id="tempo.backswing",
        title="Backswing duration against the tour band",
        kind=RuleKind.BAND,
        metric=MetricName.BACKSWING_DURATION,
        phase=SwingPhase.BACKSWING,
        source=TOUR_TEMPO,
        subject="the backswing",
        band_low=18.0 / 30.0,
        band_high=24.0 / 30.0,
        band_derivation=(
            "18 and 24 frames, the source's shortest and longest published "
            "backswings, divided by the 30 fps it counted them at."
        ),
    ),
    Rule(
        rule_id="tempo.downswing",
        title="Downswing duration against the tour band",
        kind=RuleKind.BAND,
        metric=MetricName.DOWNSWING_DURATION,
        phase=SwingPhase.DOWNSWING,
        source=TOUR_TEMPO,
        subject="the downswing",
        band_low=6.0 / 30.0,
        band_high=8.0 / 30.0,
        band_derivation=(
            "6 and 8 frames, the source's shortest and longest published "
            "downswings, divided by the 30 fps it counted them at."
        ),
    ),
    # --- rotation --------------------------------------------------------
    Rule(
        rule_id="rotation.shoulder_turn_top",
        title="Shoulder turn at the top against a full turn",
        kind=RuleKind.BAND,
        metric=MetricName.SHOULDER_TURN,
        event=SwingEvent.TOP,
        source=FULL_TURN_CONVENTION,
        subject="the shoulder turn at the top",
        band_low=80.0,
        band_high=100.0,
        band_derivation=(
            "'About ninety degrees', give or take the ten either side that the "
            "phrase is used with. Recorded so the refusal is about the source "
            "rather than about a missing number."
        ),
    ),
    Rule(
        rule_id="rotation.pelvis_turn_top",
        title="Pelvis turn at the top against a half turn",
        kind=RuleKind.BAND,
        metric=MetricName.PELVIS_TURN,
        event=SwingEvent.TOP,
        source=FULL_TURN_CONVENTION,
        subject="the pelvis turn at the top",
        band_low=40.0,
        band_high=50.0,
        band_derivation="'About forty-five degrees', used the same way as the shoulder figure.",
    ),
    Rule(
        rule_id="rotation.x_factor_top",
        title="X-factor at the top against the popularised figure",
        kind=RuleKind.BAND,
        metric=MetricName.X_FACTOR,
        event=SwingEvent.TOP,
        source=X_FACTOR_CONVENTION,
        subject="the separation between shoulders and hips at the top",
        band_low=40.0,
        band_high=50.0,
        band_derivation="The forty-five degrees the term was popularised with, plus or minus five.",
    ),
    Rule(
        rule_id="rotation.x_factor_top_3d",
        title="X-factor at the top against three-dimensional measurement",
        kind=RuleKind.BAND,
        metric=MetricName.X_FACTOR_3D,
        event=SwingEvent.TOP,
        source=TORSO_PELVIS_SEPARATION_3D,
        subject="the separation between shoulders and hips at the top",
        no_threshold=(
            "The three-dimensional literature measures this quantity in a defined "
            "way, which is what the engine's own triangulated version would need "
            "to be compared against -- and it reports group means and spreads "
            "rather than a range separating one swing from another. This project "
            "has no figure it can cite, so this rule has no band. It is declared "
            "so that the gap is visible: the quantity with a protocol is the one "
            "with no number, and the quantity with a number has no protocol."
        ),
    ),
    Rule(
        rule_id="rotation.pelvis_turn_through_impact",
        title="Whether the pelvis kept turning into impact",
        kind=RuleKind.SAME_CLIP,
        metric=MetricName.PELVIS_TURN,
        event=SwingEvent.TOP,
        compare_event=SwingEvent.IMPACT,
        source=SAME_SWING,
        subject="the pelvis turn",
        requires_view=CameraView.FACE_ON,
        band_derivation=(
            "No band. The change has to exceed the two measurements' own combined "
            "uncertainty, which is the only threshold in this registry that this "
            "project measured rather than borrowed."
        ),
    ),
    # --- posture and arms ------------------------------------------------
    Rule(
        rule_id="posture.head_sway_top",
        title="Head movement at the top",
        kind=RuleKind.BAND,
        metric=MetricName.HEAD_SWAY,
        event=SwingEvent.TOP,
        source=HEAD_STILL_CONVENTION,
        subject="the head's sideways travel from address",
        no_threshold=(
            "There is no number. 'Keep your head still' is advice about a "
            "direction, and no source publishes a distance -- in torso lengths or "
            "in anything else -- past which a head has moved too far. A rule that "
            "invented one would be this engine's own opinion wearing a citation."
        ),
    ),
    Rule(
        rule_id="posture.spine_tilt_impact",
        title="Spine angle retained to impact",
        kind=RuleKind.SAME_CLIP,
        metric=MetricName.SPINE_TILT,
        phase=SwingPhase.ADDRESS,
        compare_event=SwingEvent.IMPACT,
        source=SAME_SWING,
        subject="the spine angle",
        requires_view=CameraView.DOWN_THE_LINE,
        band_derivation=(
            "No band. The change from address to impact has to exceed what the "
            "two measurements could have produced by themselves."
        ),
    ),
    Rule(
        rule_id="posture.hip_sway_impact",
        title="Hip travel towards the ball at impact",
        kind=RuleKind.BAND,
        metric=MetricName.HIP_SWAY,
        event=SwingEvent.IMPACT,
        source=EARLY_EXTENSION_CONVENTION,
        subject="the hips' travel from where they sat at address",
        requires_view=CameraView.DOWN_THE_LINE,
        no_threshold=(
            "Early extension is named as a fault and never quantified. No source "
            "publishes a distance the hips may travel towards the ball, in this "
            "engine's torso lengths or in any other unit."
        ),
    ),
    Rule(
        rule_id="arms.lead_arm_top",
        title="Lead arm straightness at the top",
        kind=RuleKind.BAND,
        metric=MetricName.LEAD_ARM_ANGLE,
        event=SwingEvent.TOP,
        source=STRAIGHT_ARM_CONVENTION,
        subject="the lead arm angle at the top",
        band_low=160.0,
        band_high=180.0,
        band_derivation=(
            "180 degrees is a straight arm; twenty degrees is the bend usually "
            "described as 'soft' rather than 'broken'. Both ends are this "
            "project's reading of the phrase, which is exactly the problem."
        ),
    ),
)

REGISTRY: dict[str, Rule] = {rule.rule_id: rule for rule in _RULES}


def rule(rule_id: str) -> Rule:
    """The named rule, or KeyError. Every caller wants this to be total."""
    return REGISTRY[rule_id]


def known_rules() -> tuple[Rule, ...]:
    """Every rule the engine can evaluate, in registry order."""
    return _RULES
