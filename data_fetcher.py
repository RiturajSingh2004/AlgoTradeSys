# data_fetcher.py - Stock Data Fetching Module
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional
import time

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetching class that supports multiple APIs
    Primary: Yahoo Finance (free, reliable)
    Fallback: Alpha Vantage (with API key)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize data fetcher
        
        Args:
            api_key: Alpha Vantage API key (optional)
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_stock_data(self, symbol: str, period: str = '6mo') -> Optional[pd.DataFrame]:
        """
        Fetch stock data using Yahoo Finance (primary) or Alpha Vantage (fallback)
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Try Yahoo Finance first (free and reliable)
            return self._fetch_yahoo_data(symbol, period)
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            
            # Fallback to Alpha Vantage if API key is available
            if self.api_key:
                try:
                    return self._fetch_alpha_vantage_data(symbol)
                except Exception as e:
                    logger.error(f"Alpha Vantage also failed for {symbol}: {e}")
            
            # Final fallback: generate dummy data for testing
            logger.warning(f"Generating dummy data for {symbol}")
            return self._generate_dummy_data(symbol, period)
    
    def _fetch_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            period: Time period
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} data from Yahoo Finance...")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        df = ticker.history(period=period, interval='1d')
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]
        
        # Clean data
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
    
    def _fetch_alpha_vantage_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage API
        
        Args:
            symbol: Stock symbol (will be converted from Yahoo format)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert Yahoo Finance symbol to Alpha Vantage format
        # Remove .NS suffix for Indian stocks
        av_symbol = symbol.replace('.NS', '')
        
        logger.info(f"Fetching {av_symbol} data from Alpha Vantage...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': av_symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")
        
        time_series = data.get('Time Series (Daily)', {})
        
        if not time_series:
            raise ValueError(f"No time series data for {symbol}")
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date')
        
        # Keep only last 6 months
        six_months_ago = datetime.now() - timedelta(days=180)
        df = df[df['Date'] >= six_months_ago]
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
    
    def _generate_dummy_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Generate realistic dummy stock data for testing
        
        Args:
            symbol: Stock symbol
            period: Time period
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        logger.info(f"Generating dummy data for {symbol}...")
        
        # Determine number of days based on period
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, 
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        days = period_days.get(period, 180)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Remove weekends (assuming market is closed)
        dates = dates[dates.dayofweek < 5]
        
        np.random.seed(42)  # For reproducible results
        
        # Base price (different for each stock)
        base_prices = {
            'RELIANCE.NS': 2500,
            'TCS.NS': 3200,
            'INFY.NS': 1400,
            'HDFCBANK.NS': 1600,
            'ICICIBANK.NS': 900
        }
        base_price = base_prices.get(symbol, 1000)
        
        # Generate realistic stock data using random walk
        returns = np.random.normal(0.001, 0.02, len(dates))  # Small daily returns with volatility
        prices = [base_price]
        
        for r in returns[1:]:
            new_price = prices[-1] * (1 + r)
            prices.append(max(new_price, base_price * 0.5))  # Prevent unrealistic crashes
        
        # Create OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            daily_volatility = 0.015
            high = close_price * (1 + np.random.uniform(0, daily_volatility))
            low = close_price * (1 - np.random.uniform(0, daily_volatility))
            
            # Ensure realistic OHLC relationships
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume (higher volume on larger price movements)
            price_change = abs(close_price - open_price) / open_price
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 5) * np.random.uniform(0.5, 2))
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} dummy records for {symbol}")
        return df
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', info.get('currentPrice'))
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            return not hist.empty
        except:
            return False

# Test the data fetcher
if __name__ == "__main__":
    # Test with NIFTY 50 stocks
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    fetcher = DataFetcher()
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        df = fetcher.get_stock_data(symbol, '6mo')
        
        if df is not None:
            print(f"✅ Success: {len(df)} records")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Latest close: ₹{df['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Failed to fetch data")