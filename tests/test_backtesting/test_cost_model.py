"""
Unit tests for the transaction cost models.

The invariants asserted here are the ones a "helpful" refactor is most likely to
break: costs are always adverse, fill prices stay strictly positive, an unfunded
parameterisation is rejected at construction, and a bar that traded nothing
cannot fill an order.
"""

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.cost_model import (
    BPS,
    CostModel,
    FillRequest,
    FixedSlippageModel,
    VolumeShareSlippageModel,
    ZeroCostModel,
)

TIMESTAMP = pd.Timestamp('2024-01-02')


def buy(reference_price=100.0, quantity=10.0, bar_volume=None):
    """Build a buy FillRequest with sensible defaults."""
    return FillRequest(side=1, reference_price=reference_price, quantity=quantity,
                       bar_volume=bar_volume, timestamp=TIMESTAMP)


def sell(reference_price=100.0, quantity=10.0, bar_volume=None):
    """Build a sell FillRequest with sensible defaults."""
    return FillRequest(side=-1, reference_price=reference_price, quantity=quantity,
                       bar_volume=bar_volume, timestamp=TIMESTAMP)


class TestFillRequest(unittest.TestCase):
    """A request the models can trust is a request they validated."""

    def test_rejects_an_unknown_side(self):
        with self.assertRaises(ValueError):
            FillRequest(side=0, reference_price=100.0, quantity=1.0)

    def test_rejects_a_non_positive_reference_price(self):
        for price in (0.0, -1.0):
            with self.subTest(price=price), self.assertRaises(ValueError):
                FillRequest(side=1, reference_price=price, quantity=1.0)

    def test_rejects_a_non_finite_reference_price(self):
        for price in (float('nan'), float('inf')):
            with self.subTest(price=price), self.assertRaises(ValueError):
                FillRequest(side=1, reference_price=price, quantity=1.0)

    def test_rejects_a_negative_quantity(self):
        with self.assertRaises(ValueError):
            FillRequest(side=1, reference_price=100.0, quantity=-1.0)

    def test_zero_quantity_is_allowed(self):
        request = FillRequest(side=1, reference_price=100.0, quantity=0.0)
        self.assertEqual(request.quantity, 0.0)

    def test_usable_volume_rejects_missing_zero_and_nan(self):
        for volume in (None, 0.0, -5.0, float('nan')):
            with self.subTest(volume=volume):
                self.assertIsNone(buy(bar_volume=volume).usable_volume)

    def test_usable_volume_returns_a_positive_volume(self):
        self.assertEqual(buy(bar_volume=1000.0).usable_volume, 1000.0)


class TestZeroCostModel(unittest.TestCase):
    """The frictionless default, kept so old numbers stay reproducible."""

    def setUp(self):
        self.model = ZeroCostModel()

    def test_fills_at_the_reference_price(self):
        self.assertEqual(self.model.fill_price(buy()), 100.0)
        self.assertEqual(self.model.fill_price(sell()), 100.0)

    def test_costs_nothing(self):
        self.assertEqual(self.model.slippage_cost(buy()), 0.0)

    def test_fills_any_size(self):
        self.assertEqual(self.model.max_fillable_quantity(buy(quantity=1e12)), math.inf)

    def test_is_flagged_frictionless(self):
        self.assertTrue(self.model.is_frictionless)


class TestFixedSlippageModel(unittest.TestCase):
    """Constant cost per fill."""

    def test_buy_pays_up_and_sell_gives_up(self):
        model = FixedSlippageModel(slippage_bps=10.0, half_spread_bps=5.0)

        self.assertAlmostEqual(model.fill_price(buy()), 100.0 * 1.0015)
        self.assertAlmostEqual(model.fill_price(sell()), 100.0 * 0.9985)

    def test_cost_is_independent_of_size(self):
        model = FixedSlippageModel(slippage_bps=10.0)

        small = model.fill_price(buy(quantity=1.0))
        large = model.fill_price(buy(quantity=1_000_000.0))

        self.assertEqual(small, large)

    def test_slippage_cost_scales_with_quantity(self):
        model = FixedSlippageModel(slippage_bps=100.0)

        self.assertAlmostEqual(model.slippage_cost(buy(quantity=10.0)), 10.0)

    def test_rejects_negative_basis_points(self):
        with self.assertRaises(ValueError):
            FixedSlippageModel(slippage_bps=-1.0)
        with self.assertRaises(ValueError):
            FixedSlippageModel(half_spread_bps=-0.5)

    def test_rejects_non_finite_basis_points(self):
        with self.assertRaises(ValueError):
            FixedSlippageModel(slippage_bps=float('nan'))
        with self.assertRaises(ValueError):
            FixedSlippageModel(slippage_bps=float('inf'))

    def test_rejects_a_cost_that_would_zero_a_sell_fill(self):
        with self.assertRaises(ValueError):
            FixedSlippageModel(slippage_bps=9_000.0, half_spread_bps=1_000.0)

    def test_zero_configuration_is_flagged_frictionless(self):
        self.assertTrue(FixedSlippageModel().is_frictionless)
        self.assertFalse(FixedSlippageModel(slippage_bps=0.1).is_frictionless)

    def test_description_names_both_terms(self):
        description = FixedSlippageModel(slippage_bps=5.0, half_spread_bps=1.0).description
        self.assertIn('5', description)
        self.assertIn('1', description)


class TestVolumeShareSlippageModel(unittest.TestCase):
    """Cost and fillable size both driven by the bar's traded volume."""

    def test_cost_grows_with_participation(self):
        model = VolumeShareSlippageModel(impact_coefficient=0.1, max_participation=1.0)

        small = model.adverse_fraction(buy(quantity=1.0, bar_volume=10_000.0))
        large = model.adverse_fraction(buy(quantity=100.0, bar_volume=10_000.0))

        self.assertLess(small, large)

    def test_square_root_law(self):
        """An order four times bigger pays twice the impact, not four times."""
        model = VolumeShareSlippageModel(impact_coefficient=0.2, max_participation=1.0)

        single = model.adverse_fraction(buy(quantity=100.0, bar_volume=10_000.0))
        quadruple = model.adverse_fraction(buy(quantity=400.0, bar_volume=10_000.0))

        self.assertAlmostEqual(quadruple, 2 * single)

    def test_half_spread_is_charged_even_on_a_vanishing_order(self):
        model = VolumeShareSlippageModel(half_spread_bps=3.0, impact_coefficient=0.1)

        fraction = model.adverse_fraction(buy(quantity=0.0, bar_volume=10_000.0))

        self.assertAlmostEqual(fraction, 3.0 * BPS)

    def test_participation_is_capped_at_the_models_own_limit(self):
        """Pricing beyond the cap would charge for units the model refuses to fill."""
        model = VolumeShareSlippageModel(impact_coefficient=0.1, max_participation=0.1)

        at_cap = model.adverse_fraction(buy(quantity=1_000.0, bar_volume=10_000.0))
        way_over = model.adverse_fraction(buy(quantity=100_000.0, bar_volume=10_000.0))

        self.assertEqual(at_cap, way_over)

    def test_max_fillable_quantity_is_a_share_of_the_bar(self):
        model = VolumeShareSlippageModel(max_participation=0.25)

        self.assertEqual(model.max_fillable_quantity(buy(bar_volume=800.0)), 200.0)

    def test_a_bar_without_volume_is_unfillable(self):
        model = VolumeShareSlippageModel()

        for volume in (None, 0.0, float('nan')):
            with self.subTest(volume=volume):
                request = buy(bar_volume=volume)
                self.assertEqual(model.max_fillable_quantity(request), 0.0)
                with self.assertRaises(ValueError):
                    model.fill_price(request)

    def test_rejects_negative_parameters(self):
        with self.assertRaises(ValueError):
            VolumeShareSlippageModel(half_spread_bps=-1.0)
        with self.assertRaises(ValueError):
            VolumeShareSlippageModel(impact_coefficient=-0.01)

    def test_rejects_a_non_positive_participation_cap(self):
        for cap in (0.0, -0.1):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                VolumeShareSlippageModel(max_participation=cap)

    def test_rejects_a_participation_cap_above_one_whole_bar(self):
        with self.assertRaises(ValueError):
            VolumeShareSlippageModel(max_participation=1.5)

    def test_rejects_a_worst_case_cost_of_a_hundred_percent(self):
        with self.assertRaises(ValueError):
            VolumeShareSlippageModel(impact_coefficient=2.0, max_participation=1.0)

    def test_frictionless_only_when_nothing_is_charged_or_capped(self):
        self.assertTrue(
            VolumeShareSlippageModel(impact_coefficient=0.0, max_participation=1.0)
            .is_frictionless
        )
        # A participation cap is a friction even when the price is untouched.
        self.assertFalse(
            VolumeShareSlippageModel(impact_coefficient=0.0, max_participation=0.5)
            .is_frictionless
        )


class TestCostsAreAlwaysAdverse(unittest.TestCase):
    """
    The invariant the whole module exists to guarantee.

    No model and no parameterisation may fill a buy below, or a sell above, the
    reference price - that would be the market paying you to trade.
    """

    def _models(self):
        for slippage in (0.0, 0.5, 5.0, 250.0, 4_000.0):
            for half_spread in (0.0, 1.0, 500.0):
                if (slippage + half_spread) * BPS >= 1.0:
                    continue
                yield FixedSlippageModel(slippage_bps=slippage,
                                         half_spread_bps=half_spread)
        for impact in (0.0, 0.01, 0.5, 0.9):
            for participation in (0.01, 0.5, 1.0):
                yield VolumeShareSlippageModel(half_spread_bps=2.0,
                                               impact_coefficient=impact,
                                               max_participation=participation)
        yield ZeroCostModel()

    def test_buys_never_fill_below_and_sells_never_above_the_reference(self):
        for model in self._models():
            for quantity in (0.0, 1.0, 500.0, 1_000_000.0):
                with self.subTest(model=model.description, quantity=quantity):
                    buy_request = buy(quantity=quantity, bar_volume=10_000.0)
                    sell_request = sell(quantity=quantity, bar_volume=10_000.0)

                    self.assertGreaterEqual(model.fill_price(buy_request), 100.0)
                    self.assertLessEqual(model.fill_price(sell_request), 100.0)

    def test_fill_prices_are_strictly_positive(self):
        for model in self._models():
            for price in (0.0001, 1.0, 100.0, 1e6):
                with self.subTest(model=model.description, price=price):
                    request = sell(reference_price=price, quantity=1_000_000.0,
                                   bar_volume=10_000.0)
                    self.assertGreater(model.fill_price(request), 0.0)

    def test_slippage_cost_is_never_negative(self):
        for model in self._models():
            with self.subTest(model=model.description):
                self.assertGreaterEqual(
                    model.slippage_cost(sell(quantity=25.0, bar_volume=10_000.0)), 0.0
                )


class FavourableModel(CostModel):
    """A hostile subclass that tries to hand the trader a better price."""

    def adverse_fraction(self, request):
        return -0.01


class NonFiniteModel(CostModel):
    """A broken subclass returning a non-finite cost."""

    def adverse_fraction(self, request):
        return float('nan')


class RuinousModel(CostModel):
    """A subclass whose cost would take a sell fill to zero."""

    def adverse_fraction(self, request):
        return 1.5


class TestSubclassesCannotProduceFavourableFills(unittest.TestCase):
    """The sign lives in the base class, so a subclass cannot invert it."""

    def test_a_negative_fraction_raises(self):
        with self.assertRaises(ValueError):
            FavourableModel().fill_price(buy())

    def test_a_non_finite_fraction_raises(self):
        with self.assertRaises(ValueError):
            NonFiniteModel().fill_price(buy())

    def test_a_fraction_that_zeroes_the_fill_raises(self):
        with self.assertRaises(ValueError):
            RuinousModel().fill_price(sell())

    def test_unlimited_liquidity_is_the_default(self):
        self.assertEqual(FavourableModel().max_fillable_quantity(buy()), math.inf)


if __name__ == '__main__':
    unittest.main()
