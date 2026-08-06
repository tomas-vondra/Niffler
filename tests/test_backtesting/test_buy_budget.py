"""
Unit tests for the buy-order cash budget.

A full-deployment buy (``position_size = 1.0``, the default) spends the whole
cash balance. ``budget / (1 + commission)``, recomposed as
``trade_value + trade_value * commission``, lands one or two ULP above the
budget for most balances; the engine's affordability check then rejected the
order with no trade and no log line. These tests pin the arithmetic down so that
a budget-exact order is always affordable - and that it still never exceeds the
budget, which is the failure mode the naive "just add an epsilon" fix creates.
"""

import random
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.cost_model import FixedSlippageModel
from niffler.backtesting.trade import TradeSide
from niffler.strategies.base_strategy import BaseStrategy

COMMISSIONS = (0.0, 0.0001, 0.001, 0.0025, 0.01, 0.1)


class AlternatingStrategy(BaseStrategy):
    """Buys and sells on alternating bars, always deploying the full balance."""

    def __init__(self, first_signal_index=1, stride=2):
        super().__init__("AlternatingStrategy", {}, None)
        self.first_signal_index = first_signal_index
        self.stride = stride

    def generate_signals(self, data):
        df = data.copy()
        signals = [0] * len(df)
        side = 1
        for i in range(self.first_signal_index, len(df), self.stride):
            signals[i] = side
            side = -side
        df['signal'] = signals
        df['position_size'] = 1.0
        return df

    def validate_data(self, data):
        return True

    def get_description(self):
        return "Alternating full-deployment strategy"


def make_awkward_data(n_bars=60):
    """
    Prices chosen so cash balances land on awkward binary fractions.

    Round prices leave round balances, which happen to survive the round trip;
    the bug only shows up on the balances real data produces.
    """
    index = pd.date_range('2024-01-01', periods=n_bars, freq='D')
    closes = [100.0 + 7.31 * ((i * 37) % 11) / 3.0 for i in range(n_bars)]
    return pd.DataFrame({
        'open': closes,
        'high': [c * 1.013 for c in closes],
        'low': [c * 0.987 for c in closes],
        'close': closes,
        'volume': [1_000_000.0] * n_bars,
    }, index=index)


class TestAffordableTradeValue(unittest.TestCase):
    """The arithmetic itself, across many balances and commission rates."""

    def _budgets(self, count=2000):
        rng = random.Random(20240801)
        budgets = [10_000.0, 9842.10123456, 1.0, 0.5, 123456.789, 1e-3, 1e9]
        budgets += [rng.uniform(1e-2, 1e7) for _ in range(count)]
        return budgets

    def test_the_recomposed_cost_never_exceeds_the_budget(self):
        for commission in COMMISSIONS:
            engine = BacktestEngine(commission=commission)
            for budget in self._budgets():
                trade_value = engine._affordable_trade_value(budget)
                total_cost = trade_value + trade_value * commission
                if total_cost > budget:
                    self.fail(
                        f"commission={commission} budget={budget!r}: "
                        f"total_cost={total_cost!r} exceeds the budget"
                    )

    def test_the_result_is_not_needlessly_small(self):
        """Stepping down must cost a few ULP, not a meaningful amount of money."""
        for commission in COMMISSIONS:
            engine = BacktestEngine(commission=commission)
            for budget in self._budgets(200):
                ideal = budget / (1.0 + commission)
                trade_value = engine._affordable_trade_value(budget)

                self.assertLessEqual(trade_value, ideal)
                self.assertGreater(trade_value, ideal * (1 - 1e-12))

    def test_a_non_positive_budget_buys_nothing(self):
        engine = BacktestEngine(commission=0.001)

        self.assertEqual(engine._affordable_trade_value(0.0), 0.0)
        self.assertEqual(engine._affordable_trade_value(-5.0), 0.0)


class TestFullDeploymentBuysAreNotDropped(unittest.TestCase):
    """
    The regression this fixes.

    A buy that deploys the entire balance must execute. It used to be rejected
    for roughly 70% of balances, silently.
    """

    def test_a_full_capital_buy_executes_for_any_balance(self):
        rng = random.Random(20240802)
        balances = [9842.10123456, 10_000.0, 1_000.0] + [
            rng.uniform(100.0, 500_000.0) for _ in range(500)
        ]

        for commission in (0.0, 0.001, 0.0025, 0.01):
            engine = BacktestEngine(commission=commission)
            for balance in balances:
                trade = engine._execute_buy_trade(
                    timestamp=pd.Timestamp('2024-01-02'), symbol='TEST',
                    price=100.24, position_size=1.0, available_cash=balance,
                    bar_volume=1e9
                )
                if trade is None:
                    self.fail(
                        f"commission={commission} balance={balance!r}: "
                        f"a full-capital buy was rejected"
                    )

    def test_the_order_never_spends_more_than_the_balance(self):
        rng = random.Random(20240803)

        for commission in (0.0, 0.001, 0.01):
            engine = BacktestEngine(commission=commission)
            for _ in range(300):
                balance = rng.uniform(100.0, 500_000.0)
                trade = engine._execute_buy_trade(
                    timestamp=pd.Timestamp('2024-01-02'), symbol='TEST',
                    price=100.24, position_size=1.0, available_cash=balance,
                    bar_volume=1e9
                )

                self.assertIsNotNone(trade)
                self.assertLessEqual(trade.value + trade.commission, balance)

    def test_every_buy_signal_becomes_a_trade(self):
        data = make_awkward_data()
        strategy = AlternatingStrategy()
        signals = strategy.generate_signals(data)['signal']

        # A signal on the final bar never fills (next_bar_open), by design.
        expected_buys = int((signals.iloc[:-1] == 1).sum())
        self.assertGreater(expected_buys, 3)

        result = BacktestEngine(commission=0.001).run_backtest(strategy, data, 'TEST')
        executed_buys = sum(1 for t in result.trades if t.side == TradeSide.BUY)

        self.assertEqual(executed_buys, expected_buys)

    def test_every_buy_signal_becomes_a_trade_with_costs_too(self):
        data = make_awkward_data()
        strategy = AlternatingStrategy()
        signals = strategy.generate_signals(data)['signal']
        expected_buys = int((signals.iloc[:-1] == 1).sum())

        engine = BacktestEngine(
            commission=0.001,
            cost_model=FixedSlippageModel(slippage_bps=50.0, half_spread_bps=5.0)
        )
        result = engine.run_backtest(strategy, data, 'TEST')
        executed_buys = sum(1 for t in result.trades if t.side == TradeSide.BUY)

        self.assertEqual(executed_buys, expected_buys)

    def test_the_trade_count_is_stable_across_cost_levels(self):
        """
        The comparison this whole feature exists for.

        Costs must change the returns, not which signals happen to execute.
        """
        data = make_awkward_data()
        strategy = AlternatingStrategy()

        counts, returns = [], []
        for bps in (0.0, 25.0, 50.0, 100.0):
            engine = BacktestEngine(
                commission=0.001,
                cost_model=FixedSlippageModel(slippage_bps=bps)
            )
            result = engine.run_backtest(AlternatingStrategy(), data, 'TEST')
            counts.append(result.total_trades)
            returns.append(result.total_return_pct)

        self.assertEqual(len(set(counts)), 1, f"trade count varied with costs: {counts}")
        self.assertEqual(sorted(returns, reverse=True), returns,
                         f"returns did not degrade monotonically: {returns}")

    def test_cash_stays_non_negative_over_a_full_run(self):
        data = make_awkward_data()

        for bps in (0.0, 100.0, 1_000.0):
            engine = BacktestEngine(
                commission=0.001,
                cost_model=FixedSlippageModel(slippage_bps=bps)
            )
            result = engine.run_backtest(AlternatingStrategy(), data, 'TEST')

            self.assertGreaterEqual(result.portfolio_values.min(), 0.0)
            self.assertGreater(result.total_trades, 0)


if __name__ == '__main__':
    unittest.main()
