"""
Pinescript to Python Trading Library

A modular trading library that converts Pine Script strategies to Python,
with clean separation of concerns and SOLID principles.
"""

from models import StrategyParams, TradeResult
from strategy import MomentumStrategy, create_default_strategy, create_custom_strategy

__version__ = "1.0.0"
__all__ = [
    'StrategyParams',
    'TradeResult', 
    'MomentumStrategy',
    'create_default_strategy',
    'create_custom_strategy'
]
