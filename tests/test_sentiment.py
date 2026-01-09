"""Tests for Sentiment Analysis"""

import sys

sys.path.append("../code")


def test_sentiment_analyzer():
    """Test sentiment analyzer"""
    from features.sentiment_analyzer import FinBERTSentimentAnalyzer

    analyzer = FinBERTSentimentAnalyzer()
    sentiment = analyzer.analyze_ticker_sentiment("AAPL")
    assert "positive" in sentiment
    assert "negative" in sentiment
    assert "compound" in sentiment
    assert -1 <= sentiment["compound"] <= 1
