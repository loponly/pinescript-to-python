"""Trade result model."""

from dataclasses import dataclass


@dataclass
class TradeResult:
    """Data class for trade results."""
    
    entry_price: float
    exit_price: float
    position_type: str  # 'long' or 'short'
    entry_index: int
    exit_index: int
    
    @property
    def pnl(self) -> float:
        """Calculate profit/loss for the trade."""
        if self.position_type == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # short
            return (self.entry_price - self.exit_price) / self.entry_price
