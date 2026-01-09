"""Tests for Risk Metrics"""

import numpy as np
import sys

sys.path.append("../code")


def test_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    from risk_management.risk_metrics import RiskMetricsCalculator

    calc = RiskMetricsCalculator()
    returns = np.random.randn(252) * 0.01
    sharpe = calc.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_cvar():
    """Test CVaR calculation"""
    from risk_management.risk_metrics import RiskMetricsCalculator

    calc = RiskMetricsCalculator(cvar_alpha=0.95)
    returns = np.random.randn(1000) * 0.01
    cvar = calc.calculate_cvar(returns)
    assert isinstance(cvar, float)
    assert cvar <= 0  # CVaR should be negative (loss)
