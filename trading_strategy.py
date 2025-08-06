# trading_strategy.py - RSI + Moving Average Trading Strategy
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
import talib

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Trading strategy implementation with RSI and Moving Average crossover
    
    Strategy Rules:
    - BUY: RSI < 30 (oversold) AND 20-day MA crosses above 50-day MA
    - SELL: RSI > 70 (overbought) OR 20-day MA crosses below 50-day MA
    """
    
    def __init__(self, rsi_period: int = 14, ma_short: int = 20, ma_long: int = 50):
        """
        Initialize strategy parameters
        
        Args:
            rsi_period: RSI calculation period
            ma_short: Short moving average period
            ma_long: Long moving average period
        """
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        
        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        logger.info(f"Strategy initialized - RSI({rsi_period}), MA({ma_short}, {ma_long})")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series (typically closing prices)
            period: RSI calculation period
            
        Returns:
            RSI values as pandas Series
        """
        try:
            # Use TA-Lib if available, otherwise manual calculation
            return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
        except:
            # Manual RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_moving_average(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Price series
            period: Moving average period
            
        Returns:
            Moving average values as pandas Series
        """
        return prices.rolling(window=period).mean()
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        try:
            # Use TA-Lib if available
            macd_line, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd_line, index=prices.index),
                'signal': pd.Series(signal_line, index=prices.index),
                'histogram': pd.Series(histogram, index=prices.index)
            }
        except:
            # Manual MACD calculation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Calculate RSI
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        
        # Calculate Moving Averages
        df['MA_20'] = self.calculate_moving_average(df['Close'], self.ma_short)
        df['MA_50'] = self.calculate_moving_average(df['Close'], self.ma_long)
        
        # Calculate MACD
        macd_data = self.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        # Calculate additional indicators
        df['Volume_MA'] = self.calculate_moving_average(df['Volume'], 20)
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # MA crossover signals
        df['MA_Crossover'] = np.where(df['MA_20'] > df['MA_50'], 1, 0)
        df['MA_Crossover_Signal'] = df['MA_Crossover'].diff()
        
        logger.info("Technical indicators calculated successfully")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate buy/sell signals based on strategy rules
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Skip if we don't have enough data
            if pd.isna(current['RSI']) or pd.isna(current['MA_20']) or pd.isna(current['MA_50']):
                continue
            
            signal = None
            
            # BUY Signal: RSI < 30 (oversold) AND 20-day MA crosses above 50-day MA
            if (current['RSI'] < self.rsi_oversold and 
                current['MA_20'] > current['MA_50'] and 
                previous['MA_20'] <= previous['MA_50']):
                
                signal = {
                    'date': current['Date'],
                    'symbol': 'SYMBOL',  # Will be set by calling function
                    'action': 'BUY',
                    'price': current['Close'],
                    'rsi': current['RSI'],
                    'ma_20': current['MA_20'],
                    'ma_50': current['MA_50'],
                    'volume': current['Volume'],
                    'reason': f"RSI oversold ({current['RSI']:.2f}) + MA crossover"
                }
            
            # SELL Signal: RSI > 70 (overbought) OR 20-day MA crosses below 50-day MA
            elif (current['RSI'] > self.rsi_overbought or 
                  (current['MA_20'] < current['MA_50'] and previous['MA_20'] >= previous['MA_50'])):
                
                reason = "RSI overbought" if current['RSI'] > self.rsi_overbought else "MA crossover down"
                
                signal = {
                    'date': current['Date'],
                    'symbol': 'SYMBOL',  # Will be set by calling function
                    'action': 'SELL',
                    'price': current['Close'],
                    'rsi': current['RSI'],
                    'ma_20': current['MA_20'],
                    'ma_50': current['MA_50'],
                    'volume': current['Volume'],
                    'reason': f"{reason} (RSI: {current['RSI']:.2f})"
                }
            
            if signal:
                signals.append(signal)
        
        # Set symbol for all signals
        for signal in signals:
            signal['symbol'] = df.get('Symbol', 'UNKNOWN').iloc[0] if 'Symbol' in df.columns else 'UNKNOWN'
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def backtest_strategy(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Backtest the trading strategy
        
        Args:
            df: DataFrame with OHLCV data and indicators
            initial_capital: Initial capital for backtesting
            
        Returns:
            Backtest results dictionary
        """
        signals = self.generate_signals(df)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0  # Number of shares held
        trades = []
        portfolio_values = []
        
        for signal in signals:
            if signal['action'] == 'BUY' and capital > signal['price']:
                # Buy as many shares as possible
                shares_to_buy = int(capital // signal['price'])
                if shares_to_buy > 0:
                    position += shares_to_buy
                    capital -= shares_to_buy * signal['price']
                    
                    trade = {
                        'date': signal['date'],
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': signal['price'],
                        'total': shares_to_buy * signal['price'],
                        'capital_remaining': capital,
                        'position': position
                    }
                    trades.append(trade)
            
            elif signal['action'] == 'SELL' and position > 0:
                # Sell all shares
                capital += position * signal['price']
                
                trade = {
                    'date': signal['date'],
                    'action': 'SELL',
                    'shares': position,
                    'price': signal['price'],
                    'total': position * signal['price'],
                    'capital_remaining': capital,
                    'position': 0
                }
                trades.append(trade)
                position = 0
        
        # Calculate final portfolio value
        final_price = df['Close'].iloc[-1]
        final_portfolio_value = capital + (position * final_price)
        
        # Calculate performance metrics
        total_return = final_portfolio_value - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        # Calculate win/loss ratio
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        profitable_trades = 0
        total_trade_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_trade_pairs):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                profitable_trades += 1
        
        win_ratio = (profitable_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': len(trades),
            'win_ratio': win_ratio,
            'trades': trades,
            'signals': signals
        }
        
        logger.info(f"Backtest completed: {return_percentage:.2f}% return, {win_ratio:.2f}% win ratio")
        return results
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """
        Get current trading signal based on latest data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Current signal dictionary or None
        """
        if len(df) < 2:
            return None
        
        signals = self.generate_signals(df)
        
        # Return the most recent signal
        if signals:
            return signals[-1]
        
        return None
    
    def analyze_market_condition(self, df: pd.DataFrame) -> Dict:
        """
        Analyze current market conditions
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Market analysis dictionary
        """
        if df.empty or len(df) < self.ma_long:
            return {'condition': 'insufficient_data'}
        
        latest = df.iloc[-1]
        
        # Determine trend
        if latest['MA_20'] > latest['MA_50']:
            trend = 'uptrend'
        elif latest['MA_20'] < latest['MA_50']:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Determine momentum
        if latest['RSI'] > 70:
            momentum = 'overbought'
        elif latest['RSI'] < 30:
            momentum = 'oversold'
        elif latest['RSI'] > 50:
            momentum = 'bullish'
        else:
            momentum = 'bearish'
        
        # Volatility assessment
        volatility = df['Volatility'].iloc[-1]
        if volatility > df['Volatility'].quantile(0.8):
            vol_condition = 'high'
        elif volatility < df['Volatility'].quantile(0.2):
            vol_condition = 'low'
        else:
            vol_condition = 'normal'
        
        return {
            'condition': 'analyzed',
            'trend': trend,
            'momentum': momentum,
            'volatility': vol_condition,
            'rsi': latest['RSI'],
            'ma_20': latest['MA_20'],
            'ma_50': latest['MA_50'],
            'price': latest['Close'],
            'volume_vs_avg': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1
        }

# Test the strategy
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample price data
    prices = []
    base_price = 1000
    for i in range(len(dates)):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + change)
        prices.append(base_price)
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(len(dates))]
    })
    
    # Test the strategy
    strategy = TradingStrategy()
    
    print("Calculating indicators...")
    df_with_indicators = strategy.calculate_indicators(sample_data)
    
    print("Running backtest...")
    results = strategy.backtest_strategy(df_with_indicators)
    
    print(f"\n{'='*50}")
    print("STRATEGY BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ₹{results['final_portfolio_value']:,.2f}")
    print(f"Total Return: ₹{results['total_return']:,.2f}")
    print(f"Return Percentage: {results['return_percentage']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Ratio: {results['win_ratio']:.2f}%")
    
    # Show market analysis
    print(f"\n{'='*30}")
    print("CURRENT MARKET ANALYSIS")
    print(f"{'='*30}")
    analysis = strategy.analyze_market_condition(df_with_indicators)
    for key, value in analysis.items():
        if key != 'condition':
            print(f"{key.replace('_', ' ').title()}: {value}")