import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy


class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30, 
                 position_size: float = 1.0):
        """
        Initialize the strategy.
        
        Args:
            short_window: Period for short moving average
            long_window: Period for long moving average  
            position_size: Fraction of portfolio to use for each trade (0.0 to 1.0)
        """
        parameters = {
            'short_window': short_window,
            'long_window': long_window,
            'position_size': position_size
        }
        super().__init__("Simple MA Crossover", parameters)
        
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal and position_size columns added
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
            
        df = data.copy()
        
        # Calculate moving averages
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Initialize signal column
        df['signal'] = 0
        df['position_size'] = self.position_size
        
        # Generate signals
        # Buy when short MA crosses above long MA
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
        
        # Sell when short MA crosses below long MA  
        df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1
        
        # Only generate signals when we have enough data for both MAs
        min_periods = max(self.short_window, self.long_window)
        df.iloc[:min_periods, df.columns.get_loc('signal')] = 0
        
        # Only signal on actual crossovers, not just when one is above the other
        df['ma_short_prev'] = df['ma_short'].shift(1)
        df['ma_long_prev'] = df['ma_long'].shift(1)
        
        # Buy signal: short MA crosses above long MA
        buy_condition = (
            (df['ma_short'] > df['ma_long']) & 
            (df['ma_short_prev'] <= df['ma_long_prev'])
        )
        
        # Sell signal: short MA crosses below long MA
        sell_condition = (
            (df['ma_short'] < df['ma_long']) & 
            (df['ma_short_prev'] >= df['ma_long_prev'])
        )
        
        # Reset signals to only crossovers
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Clean up temporary columns
        df = df.drop(['ma_short_prev', 'ma_long_prev'], axis=1)
        
        return df
        
    def get_description(self) -> str:
        """Return strategy description."""
        return (f"Simple Moving Average Crossover Strategy with "
                f"{self.short_window}-period short MA and {self.long_window}-period long MA. "
                f"Position size: {self.position_size * 100}%")