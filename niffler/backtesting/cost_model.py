"""
Transaction cost models.

A backtest that fills every order at the exact reference price, in unlimited
size, is a statement about a market that does not exist. This module models the
part of the gap that is cheap to model honestly:

* the **half spread** paid for crossing the book,
* **slippage / market impact**, which grows with the size of the order relative
  to the liquidity the bar actually traded,
* a **participation limit**, because a bar cannot absorb an unlimited order.

Direction discipline
--------------------
Every cost is *adverse*. A buy fills at or above the reference price and a sell
at or below it, for every model and every parameterisation. This is enforced
structurally rather than by convention: :meth:`CostModel.fill_price` is concrete
and applies the sign itself, and the extension point subclasses implement
(:meth:`CostModel.adverse_fraction`) can only return a non-negative fraction.
A subclass returning a negative number gets a ``ValueError``, not a free lunch.

Costs are not commission. Commission stays on the engine
(``BacktestEngine(commission=...)``) and is charged on the *filled* notional;
these models move the fill price itself, so the two compose without
double-counting.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

#: One basis point expressed as a fraction of price.
BPS = 1e-4

#: An adverse move of 100% would take a sell fill to zero, so no model may be
#: configured (or may return) a fraction at or above this.
MAX_ADVERSE_FRACTION = 1.0


@dataclass(frozen=True)
class FillRequest:
    """
    One order presented to a cost model for pricing.

    Attributes:
        side: ``+1`` for a buy (entry), ``-1`` for a sell (exit)
        reference_price: The untouched price the fill is benchmarked against -
            the bar's open for a normal order, the stop fill price for a stop
            exit. Must be finite and strictly positive.
        quantity: Units the caller wants to trade. May be 0.0 when a model is
            only being asked for its size-independent cost.
        bar_volume: Volume the execution bar traded, or None when it is unknown.
            Only the volume-aware models look at it.
        timestamp: Bar timestamp of the fill; carried for logging and for
            models that want a time-of-day term. Not used by the models here.
    """

    side: int
    reference_price: float
    quantity: float
    bar_volume: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        """
        Validate the request.

        Raises:
            ValueError: If the side is not +1/-1, the reference price is not a
                finite positive number, or the quantity is negative or not finite
        """
        if self.side not in (1, -1):
            raise ValueError(f"side must be +1 (buy) or -1 (sell), got {self.side}")
        if not math.isfinite(self.reference_price) or self.reference_price <= 0:
            raise ValueError(
                f"reference_price must be finite and positive, got {self.reference_price}"
            )
        if not math.isfinite(self.quantity) or self.quantity < 0:
            raise ValueError(
                f"quantity must be finite and non-negative, got {self.quantity}"
            )

    @property
    def is_buy(self) -> bool:
        """True when this request is a buy."""
        return self.side == 1

    @property
    def usable_volume(self) -> Optional[float]:
        """
        The bar's volume when it can absorb anything, None otherwise.

        A missing, NaN or non-positive volume all mean the same thing to a
        liquidity model: this bar traded nothing, so it cannot fill an order.
        They are deliberately not conflated with "unlimited liquidity".
        """
        volume = self.bar_volume
        if volume is None:
            return None
        volume = float(volume)
        if not math.isfinite(volume) or volume <= 0:
            return None
        return volume


class CostModel(ABC):
    """
    Base class for transaction cost models.

    Subclasses implement :meth:`adverse_fraction` - how much of the reference
    price the order gives up - and optionally :meth:`max_fillable_quantity`.
    They do **not** implement :meth:`fill_price`: the base class owns the sign,
    so no parameterisation of any subclass can produce a favourable fill.
    """

    @abstractmethod
    def adverse_fraction(self, request: FillRequest) -> float:
        """
        Fraction of the reference price given up to trading frictions.

        Args:
            request: The order being priced

        Returns:
            A non-negative fraction (0.01 = 1% = 100 bps). It is applied against
            the trade's direction by :meth:`fill_price`, so implementations must
            never try to signal a direction themselves.
        """

    def fill_price(self, request: FillRequest) -> float:
        """
        Price the order actually fills at.

        Args:
            request: The order being priced

        Returns:
            ``reference_price * (1 + fraction)`` for a buy and
            ``reference_price * (1 - fraction)`` for a sell, which is always at
            least as bad for the trader as the reference price.

        Raises:
            ValueError: If the model returned a negative, non-finite or
                impossibly large (>= 100%) fraction, or the resulting price is
                not strictly positive
        """
        fraction = float(self.adverse_fraction(request))

        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError(
                f"{type(self).__name__}.adverse_fraction returned {fraction}; slippage "
                f"must be a finite, non-negative fraction (costs are never favourable)"
            )
        if fraction >= MAX_ADVERSE_FRACTION:
            raise ValueError(
                f"{type(self).__name__}.adverse_fraction returned {fraction}: an adverse "
                f"move of 100% or more would take a sell fill to zero or below"
            )

        price = request.reference_price * (1.0 + request.side * fraction)

        if not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"{type(self).__name__} produced a non-positive fill price ({price}) "
                f"from reference price {request.reference_price}"
            )
        return price

    def slippage_cost(self, request: FillRequest, quantity: Optional[float] = None) -> float:
        """
        Cash cost of the fill's deviation from the reference price.

        Args:
            request: The order being priced
            quantity: Units actually filled, when that differs from the request's
                quantity (a truncated order). Defaults to the request's quantity.

        Returns:
            ``|fill_price - reference_price| * quantity``, always >= 0
        """
        filled = request.quantity if quantity is None else float(quantity)
        return abs(self.fill_price(request) - request.reference_price) * filled

    def max_fillable_quantity(self, request: FillRequest) -> float:
        """
        Largest quantity this model is willing to fill on this bar.

        Args:
            request: The order being priced

        Returns:
            ``math.inf`` by default - unlimited liquidity. Models that cap
            participation return a finite number, and the engine truncates the
            order to it rather than dropping it.
        """
        return math.inf

    @property
    def is_frictionless(self) -> bool:
        """
        True when this configuration charges nothing and caps nothing.

        Used by the CLI to label a run as assuming a market that does not exist.
        """
        return False

    @property
    def description(self) -> str:
        """One-line human-readable description, reported next to the results."""
        return type(self).__name__


class ZeroCostModel(CostModel):
    """
    No slippage, no spread, no size limit.

    The default, so that every number produced before transaction costs existed
    stays reproducible. It is also a lie about the market, which is why the
    scripts label a run using it explicitly.
    """

    def adverse_fraction(self, request: FillRequest) -> float:
        """
        Always 0.0.

        Args:
            request: The order being priced (ignored)

        Returns:
            0.0
        """
        return 0.0

    @property
    def is_frictionless(self) -> bool:
        """Always True: this model is the frictionless assumption."""
        return True

    @property
    def description(self) -> str:
        """One-line human-readable description."""
        return "none (frictionless fills: no spread, no slippage, no size limit)"


class FixedSlippageModel(CostModel):
    """
    Constant cost per fill, independent of order size.

    The simplest honest model: every buy pays ``slippage_bps + half_spread_bps``
    above the reference price and every sell gives up the same below it. The two
    terms are separate only so they can be reported separately - a spread is a
    property of the instrument, slippage a property of your execution.

    Order size is ignored, so this model understates the cost of a large order
    in a thin market; use :class:`VolumeShareSlippageModel` for that.
    """

    def __init__(self, slippage_bps: float = 0.0, half_spread_bps: float = 0.0):
        """
        Initialize the model.

        Args:
            slippage_bps: Execution slippage in basis points (5.0 = 0.05%)
            half_spread_bps: Half the bid/ask spread in basis points, i.e. the
                cost of crossing from the mid to the touch

        Raises:
            ValueError: If either figure is negative or not finite, or if they
                sum to 10000 bps (100%) or more, which would take a sell fill
                to zero
        """
        self.slippage_bps = _validate_bps(slippage_bps, 'slippage_bps')
        self.half_spread_bps = _validate_bps(half_spread_bps, 'half_spread_bps')

        total = (self.slippage_bps + self.half_spread_bps) * BPS
        if total >= MAX_ADVERSE_FRACTION:
            raise ValueError(
                f"slippage_bps + half_spread_bps must stay below "
                f"{MAX_ADVERSE_FRACTION / BPS:.0f} bps (100%), got {total / BPS:.0f} bps"
            )

    def adverse_fraction(self, request: FillRequest) -> float:
        """
        Constant cost fraction.

        Args:
            request: The order being priced (only its existence matters)

        Returns:
            ``(slippage_bps + half_spread_bps) / 10000``
        """
        return (self.slippage_bps + self.half_spread_bps) * BPS

    @property
    def is_frictionless(self) -> bool:
        """True only when both terms were configured to zero."""
        return self.slippage_bps == 0.0 and self.half_spread_bps == 0.0

    @property
    def description(self) -> str:
        """One-line human-readable description."""
        return (f"fixed (slippage {self.slippage_bps:g} bps, "
                f"half-spread {self.half_spread_bps:g} bps)")


class VolumeShareSlippageModel(CostModel):
    """
    Size-dependent cost driven by the order's share of the bar's volume.

    The cost of an order is::

        half_spread + impact_coefficient * sqrt(participation)

    where ``participation = quantity / bar_volume``. The **square-root** law is
    used rather than a linear one because it is what the empirical market-impact
    literature reports (impact grows fast for the first slice of the book and
    then flattens); a linear term would make small orders look free and huge
    orders implausibly expensive. ``impact_coefficient`` is dimensionless and
    expressed as a fraction of price: ``0.1`` means an order equal to the whole
    bar's volume would pay 10% (1000 bps) of impact on top of the half spread,
    and one taking 1% of the bar pays ``0.1 * sqrt(0.01)`` = 1% (100 bps).

    Liquidity is finite: :meth:`max_fillable_quantity` caps an order at
    ``max_participation * bar_volume`` and the engine truncates to it, recording
    the reduced quantity as a partial fill.

    Bars with **no usable volume** (missing, NaN or non-positive) are treated as
    **unfillable**, not as free: a bar that traded nothing cannot absorb an
    order. :meth:`max_fillable_quantity` returns 0.0 for them and
    :meth:`adverse_fraction` raises, because pricing a fill that cannot happen
    would be inventing liquidity.
    """

    def __init__(self, half_spread_bps: float = 0.0, impact_coefficient: float = 0.1,
                 max_participation: float = 0.1):
        """
        Initialize the model.

        Args:
            half_spread_bps: Half the bid/ask spread in basis points, charged on
                every fill regardless of size
            impact_coefficient: Dimensionless coefficient on
                ``sqrt(participation)``, expressed as a fraction of price
            max_participation: Largest share of a bar's volume a single order may
                take, in (0, 1]

        Raises:
            ValueError: If any figure is negative or not finite, if
                ``max_participation`` is outside (0, 1], or if the worst cost the
                model can charge reaches 100%
        """
        self.half_spread_bps = _validate_bps(half_spread_bps, 'half_spread_bps')

        impact_coefficient = float(impact_coefficient)
        if not math.isfinite(impact_coefficient) or impact_coefficient < 0:
            raise ValueError(
                f"impact_coefficient must be finite and non-negative, got {impact_coefficient}"
            )
        self.impact_coefficient = impact_coefficient

        max_participation = float(max_participation)
        if not math.isfinite(max_participation) or max_participation <= 0:
            raise ValueError(
                f"max_participation must be positive, got {max_participation}"
            )
        if max_participation > 1.0:
            raise ValueError(
                f"max_participation cannot exceed 1.0 (a whole bar's volume), "
                f"got {max_participation}"
            )
        self.max_participation = max_participation

        worst = self._fraction_for(max_participation)
        if worst >= MAX_ADVERSE_FRACTION:
            raise ValueError(
                f"half_spread_bps and impact_coefficient combine to a worst-case cost of "
                f"{worst * 100:.1f}% at the participation cap; it must stay below 100%"
            )

    def _fraction_for(self, participation: float) -> float:
        """
        Cost fraction for a given participation rate.

        Args:
            participation: Order quantity divided by the bar's volume

        Returns:
            ``half_spread + impact_coefficient * sqrt(participation)``
        """
        return self.half_spread_bps * BPS + self.impact_coefficient * math.sqrt(participation)

    def adverse_fraction(self, request: FillRequest) -> float:
        """
        Cost fraction for this order's share of the bar.

        The participation used is capped at ``max_participation``, because the
        model never intends to fill more than that: the engine truncates the
        order first, so pricing the untruncated size would charge for units that
        were never traded.

        Args:
            request: The order being priced

        Returns:
            The cost fraction

        Raises:
            ValueError: If the bar has no usable volume - such a bar cannot
                absorb an order at any price
        """
        volume = request.usable_volume
        if volume is None:
            raise ValueError(
                f"{type(self).__name__} cannot price a fill on a bar with no usable "
                f"volume (bar_volume={request.bar_volume!r}); such a bar is unfillable, "
                f"see max_fillable_quantity"
            )

        participation = min(request.quantity / volume, self.max_participation)
        return self._fraction_for(participation)

    def max_fillable_quantity(self, request: FillRequest) -> float:
        """
        Units this bar's volume can absorb.

        Args:
            request: The order being priced

        Returns:
            ``max_participation * bar_volume``, or 0.0 when the bar traded
            nothing usable
        """
        volume = request.usable_volume
        if volume is None:
            return 0.0
        return self.max_participation * volume

    @property
    def is_frictionless(self) -> bool:
        """True only when nothing is charged and nothing is capped."""
        return (self.half_spread_bps == 0.0
                and self.impact_coefficient == 0.0
                and self.max_participation >= 1.0)

    @property
    def description(self) -> str:
        """One-line human-readable description."""
        return (f"volume-share (half-spread {self.half_spread_bps:g} bps, "
                f"impact {self.impact_coefficient:g} * sqrt(participation), "
                f"max participation {self.max_participation:g})")


def _validate_bps(value: float, name: str) -> float:
    """
    Validate a basis-point figure.

    Args:
        value: The figure to validate
        name: Parameter name, used in the error message

    Returns:
        The value as a float

    Raises:
        ValueError: If the value is negative or not finite. Negative basis
            points would mean the market pays you to trade.
    """
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value
