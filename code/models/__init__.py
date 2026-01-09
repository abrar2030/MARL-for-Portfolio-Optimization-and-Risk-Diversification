"""
Models module for enhanced MADDPG
Includes Transformer-based architectures and attention mechanisms
"""

from .transformer_actor import TransformerActor
from .transformer_critic import TransformerCritic
from .attention_module import MultiHeadAttention, CrossAssetAttention
from .regime_detector import MarketRegimeDetector

__all__ = [
    "TransformerActor",
    "TransformerCritic",
    "MultiHeadAttention",
    "CrossAssetAttention",
    "MarketRegimeDetector",
]
