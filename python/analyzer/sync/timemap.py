"""Fitting the affine map between two camera clocks.

Two parameters, one of which is usually refused. The arithmetic is a weighted
least-squares line and is not the interesting part; deciding when the second
parameter may be fitted at all, and what the fit is allowed to claim about its
own accuracy, is.

## Why the fit is unweighted

Every anchor carries a confidence, and using it as a least-squares weight is the
obvious thing to do and is wrong. A least-squares weight asserts an inverse
variance -- that an anchor scoring 0.25 is located four times less precisely than
one scoring 1.0 -- and nothing in this system has measured that relationship.
Phase 4's confidence is a product of signal margin, landmark visibility and
frame-rate resolution; only the last of those is a statement about *when*, and
even it is a resolution bound rather than a standard deviation. So every anchor
counts once, its confidence is reported rather than applied, and a badly located
anchor shows up where it should: in the residual.

## Why the quantisation floor is a floor on the reported uncertainty

An instant located to the nearest frame carries a uniform error one frame
interval wide, whose standard deviation is that interval over sqrt(12). Both
clips contribute one, so the per-anchor standard error cannot be smaller than
their quadrature sum, however tidily the anchors happen to line up. Four anchors
that agree to a millisecond on a pair of 30 fps clips have got lucky with the
rounding; taking the scatter at face value there would report a sync ten times
better than the footage can support. The reported uncertainty is therefore the
larger of the scatter-based estimate and the quantisation-based one.
"""

from __future__ import annotations

import numpy as np

from analyzer.contracts.sync import (
    AnchorResidual,
    SyncAnchor,
    SyncConfig,
    TimeMap,
)


class TimeMapError(ValueError):
    """A map could not be fitted from the anchors supplied."""


def quantisation_floor_s(reference_interval_s: float, target_interval_s: float) -> float:
    """Best per-anchor standard error the two clips' frame rates permit.

    Each clip contributes a uniform quantisation of one frame interval, standard
    deviation interval/sqrt(12); the two are independent and add in quadrature.

    The coarser clip sets the scale, which is worth knowing before buying a
    camera. Two at 30 fps give 13.6 ms. Replacing one of them with a 240 fps
    camera gives 9.7 ms -- a 29% gain, and it can never beat 9.6 ms however fast
    that camera gets, because the 30 fps clip contributes that much on its own.
    Replacing *both* gives 1.7 ms. Upgrading one camera of a pair is therefore
    worth a factor of sqrt(2) at most; upgrading both is worth the full ratio.
    """
    twelfth = 1.0 / 12.0
    return float(np.sqrt(twelfth * (reference_interval_s**2 + target_interval_s**2)))


def residuals_against(anchors: list[SyncAnchor], time_map: TimeMap) -> list[AnchorResidual]:
    return [
        AnchorResidual(
            label=anchor.label,
            reference_s=anchor.reference_s,
            observed_target_s=anchor.target_s,
            predicted_target_s=time_map.to_target(anchor.reference_s),
            residual_ms=(anchor.target_s - time_map.to_target(anchor.reference_s)) * 1000.0,
        )
        for anchor in anchors
    ]


def fit_time_map(
    anchors: list[SyncAnchor],
    *,
    floor_s: float,
    config: SyncConfig,
) -> tuple[TimeMap, list[AnchorResidual], list[str]]:
    """Fit the map, refusing the rate parameter unless the anchors support it.

    Returns the map, the per-anchor residuals against it, and any notes the
    fitting itself generated -- a refused rate, or one so far from 1.0 that the
    supplied slow-motion factors are the likelier explanation.
    """
    if not anchors:
        raise TimeMapError("A time map needs at least one anchor; none were supplied.")

    reference = np.array([anchor.reference_s for anchor in anchors], dtype=np.float64)
    target = np.array([anchor.target_s for anchor in anchors], dtype=np.float64)
    count = reference.size

    # The centroid, which makes the two fitted parameters uncorrelated. Every
    # uncertainty below is only separable because of this choice.
    pivot = float(np.mean(reference))
    centred = reference - pivot
    spread = float(np.sum(centred**2))
    span = float(np.ptp(reference)) if count > 1 else 0.0

    notes: list[str] = []
    rate = 1.0
    rate_estimated = False

    if count < 2:
        notes.append(
            "One anchor determines an offset and nothing else, so the two clocks are "
            "assumed to run at the same speed."
        )
    elif span < config.min_rate_span_s:
        notes.append(
            f"The anchors span {span:.3f} s, under the {config.min_rate_span_s:g} s needed to "
            "fit a clock rate; a rate estimated over a baseline that short does more harm "
            "away from the anchors than assuming the clocks agree. Offset only."
        )
    elif spread <= 0.0:
        notes.append(
            "Every anchor sits at the same reference instant, so no rate is determined. "
            "Offset only."
        )
    else:
        candidate = float(np.sum(centred * (target - np.mean(target))) / spread)
        if abs(candidate - 1.0) > config.max_rate_deviation:
            notes.append(
                f"A clock rate of {candidate:.3f} was fitted and refused: it is further than "
                f"{config.max_rate_deviation:g} from 1.0, which no pair of camera clocks "
                "reaches. One anchor pair is probably wrong. Offset only."
            )
        else:
            rate, rate_estimated = candidate, True

    def solve(with_rate: float, parameters: int) -> tuple[float, float]:
        """The offset at a given rate, and the per-anchor standard error.

        The offset is the mean discrepancy at the pivot, and the expression is
        the same whether or not a rate was fitted because the model is centred
        there. The quantisation floor is the bound that always applies; the
        observed scatter is allowed to widen it and never to beat it, so a set
        of anchors that agree better than their frame rates allow is reported at
        the frame rates' limit rather than at its own lucky rounding.
        """
        found = float(np.mean(target - reference - (with_rate - 1.0) * centred))
        predicted = reference + found + (with_rate - 1.0) * centred
        spare = count - parameters
        if spare <= 0:
            return found, floor_s
        observed = float(np.sqrt(np.sum((target - predicted) ** 2) / spare))
        return found, max(observed, floor_s)

    offset, per_anchor = solve(rate, 2 if rate_estimated else 1)

    # A rate has to earn its degree of freedom. Fitted over a short baseline from
    # instants located to about a frame, a rate of 1.002 is what two identical
    # clocks produce; keeping it spends evidence on noise and buys an uncertainty
    # that grows with distance from the anchors, where a constant offset has
    # none. So it is kept only when it sits clear of its own standard error.
    if rate_estimated and spread > 0.0:
        uncertainty = per_anchor / float(np.sqrt(spread))
        if abs(rate - 1.0) < config.rate_significance * uncertainty:
            notes.append(
                f"A clock rate of {rate:.5f} was fitted and dropped: it sits within "
                f"{config.rate_significance:g} standard errors (+/-{uncertainty:.5f}) of 1.0, so "
                "these anchors are consistent with two clocks running in step. Keeping it "
                "would have made the map less certain away from the anchors and no better "
                "between them. Offset only."
            )
            rate, rate_estimated = 1.0, False
            offset, per_anchor = solve(rate, 1)

    time_map = TimeMap(
        offset_s=offset,
        rate=rate,
        rate_estimated=rate_estimated,
        pivot_s=pivot,
        offset_uncertainty_s=per_anchor / float(np.sqrt(count)),
        rate_uncertainty=(
            per_anchor / float(np.sqrt(spread)) if rate_estimated and spread > 0.0 else None
        ),
        support_start_s=float(np.min(reference)),
        support_end_s=float(np.max(reference)),
    )
    residuals = residuals_against(anchors, time_map)

    if rate_estimated and abs(rate - 1.0) > config.clock_drift_tolerance:
        notes.append(
            f"The clocks were fitted at a ratio of {rate:.4f}, which is {abs(rate - 1.0):.1%} "
            f"apart -- far beyond the {config.clock_drift_tolerance:.0%} two camera clocks "
            "differ by. The likelier explanation is that one of the supplied slow-motion "
            "factors is wrong by about that ratio; a second camera can measure the ratio of "
            "the two even though neither clip can recover its own."
        )

    return time_map, residuals, notes
