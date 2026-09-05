"""
Risk management module for Niffler trading system.

This module provides risk management capabilities including position sizing,
stop-loss management, and portfolio-level risk controls.

Risk managers hold **no position state**: the portfolio state they need is
handed to them per call as a :class:`~niffler.risk.contract.PortfolioSnapshot`.
Nothing in this package may import from ``niffler.backtesting`` or
``niffler.strategies`` - the dependency runs the other way, and
``tests/test_risk/test_registry.py`` pins the direction.
"""

from .base_risk_manager import BaseRiskManager, RiskDecision
from .contract import PortfolioSnapshot, RiskManager
from .fixed_risk_manager import FixedRiskManager
from .kelly_risk_manager import KellyRiskManager
from .registry import (
    NO_RISK_MANAGER,
    RISK_MANAGER_CLASSES,
    create_risk_manager,
    describe_risk_manager,
    get_available_risk_managers,
    get_risk_manager_class,
    get_risk_manager_name,
    get_risk_manager_parameter_names,
)

__all__ = [
    'BaseRiskManager',
    'FixedRiskManager',
    'KellyRiskManager',
    'NO_RISK_MANAGER',
    'PortfolioSnapshot',
    'RISK_MANAGER_CLASSES',
    'RiskDecision',
    'RiskManager',
    'create_risk_manager',
    'describe_risk_manager',
    'get_available_risk_managers',
    'get_risk_manager_class',
    'get_risk_manager_name',
    'get_risk_manager_parameter_names',
]
