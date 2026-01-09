"""Tests for ESG Integration"""

import sys

sys.path.append("../code")


def test_esg_provider():
    """Test ESG provider"""
    from features.esg_provider import ESGProvider

    provider = ESGProvider()
    score = provider.get_esg_score("AAPL")
    assert 0 <= score <= 100


def test_esg_filtering():
    """Test ESG filtering"""
    from features.esg_provider import ESGProvider

    provider = ESGProvider(min_score=70.0)
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD"]
    filtered = provider.filter_by_esg(tickers, min_score=70.0)
    assert len(filtered) <= len(tickers)
