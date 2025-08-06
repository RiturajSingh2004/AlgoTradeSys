# main.py - Algo Trading System Main Application
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import schedule
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_fetcher import DataFetcher
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from sheets_manager import SheetsManager
from telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlgoTradingSystem:
    """
    Main Algo Trading System that orchestrates all components
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trading system with configuration
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.symbols = config['symbols']
        self.initial_capital = config['initial_capital']
        self.current_capital = self.initial_capital
        
        # Initialize components
        self.data_fetcher = DataFetcher(config['api_key'])
        self.strategy = TradingStrategy()
        self.ml_predictor = MLPredictor()
        self.sheets_manager = SheetsManager(config['sheets_credentials'])
        self.telegram_bot = TelegramBot(config['telegram_token'], config['telegram_chat_id'])
        
        # Trading state
        self.positions = {symbol: {'quantity': 0, 'avg_price': 0} for symbol in self.symbols}
        self.trade_history = []
        
        logger.info("Algo Trading System initialized successfully")
    
    def fetch_data_for_symbols(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols
        
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        for symbol in self.symbols:
            try:
                df = self.data_fetcher.get_stock_data(symbol, period='6mo')
                if df is not None and not df.empty:
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                self.telegram_bot.send_message(f"❌ Error fetching data for {symbol}: {e}")
        
        return data
    
    def execute_trading_strategy(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Execute trading strategy on a symbol's data
        
        Args:
            symbol: Stock symbol
            df: Historical price data
            
        Returns:
            List of trade signals
        """
        try:
            # Calculate technical indicators
            df_with_indicators = self.strategy.calculate_indicators(df)
            
            # Generate signals
            signals = self.strategy.generate_signals(df_with_indicators)
            
            # Filter signals for the symbol
            symbol_signals = [s for s in signals if s['symbol'] == symbol]
            
            logger.info(f"Generated {len(symbol_signals)} signals for {symbol}")
            return symbol_signals
            
        except Exception as e:
            logger.error(f"Error executing strategy for {symbol}: {e}")
            return []
    
    def get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Get ML prediction for next day movement
        
        Args:
            symbol: Stock symbol
            df: Historical data with indicators
            
        Returns:
            Prediction dictionary or None
        """
        try:
            prediction = self.ml_predictor.predict_next_day(df)
            if prediction:
                prediction['symbol'] = symbol
                logger.info(f"ML prediction for {symbol}: {prediction['direction']} (confidence: {prediction['confidence']:.2f})")
            return prediction
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return None
    
    def process_trade_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Process a trade signal and execute if conditions are met
        
        Args:
            signal: Trade signal dictionary
            
        Returns:
            Trade execution result or None
        """
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        quantity = self.calculate_position_size(price)
        
        # Check if we have sufficient capital for buy orders
        if action == 'BUY' and price * quantity > self.current_capital:
            logger.warning(f"Insufficient capital for {symbol} buy order")
            return None
        
        # Check if we have sufficient shares for sell orders
        if action == 'SELL' and self.positions[symbol]['quantity'] < quantity:
            quantity = self.positions[symbol]['quantity']
            if quantity <= 0:
                logger.warning(f"No shares to sell for {symbol}")
                return None
        
        # Execute trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'total_value': price * quantity,
            'rsi': signal.get('rsi', 0),
            'ma_20': signal.get('ma_20', 0),
            'ma_50': signal.get('ma_50', 0)
        }
        
        # Update positions and capital
        if action == 'BUY':
            old_quantity = self.positions[symbol]['quantity']
            old_avg_price = self.positions[symbol]['avg_price']
            
            new_quantity = old_quantity + quantity
            new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
            
            self.positions[symbol]['quantity'] = new_quantity
            self.positions[symbol]['avg_price'] = new_avg_price
            self.current_capital -= trade['total_value']
            
        elif action == 'SELL':
            self.positions[symbol]['quantity'] -= quantity
            self.current_capital += trade['total_value']
            
            # Calculate P&L
            avg_price = self.positions[symbol]['avg_price']
            trade['pnl'] = (price - avg_price) * quantity
        
        self.trade_history.append(trade)
        logger.info(f"Executed {action} order: {quantity} shares of {symbol} at ₹{price:.2f}")
        
        return trade
    
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate position size based on available capital and risk management
        
        Args:
            price: Stock price
            
        Returns:
            Number of shares to trade
        """
        # Use 10% of available capital per trade (simple position sizing)
        max_investment = self.current_capital * 0.1
        quantity = int(max_investment / price)
        return max(1, quantity)  # At least 1 share
    
    def run_backtest(self) -> Dict:
        """
        Run backtest on historical data
        
        Returns:
            Backtest results
        """
        logger.info("Starting backtest...")
        
        # Reset trading state
        self.current_capital = self.initial_capital
        self.positions = {symbol: {'quantity': 0, 'avg_price': 0} for symbol in self.symbols}
        self.trade_history = []
        
        # Fetch data
        data = self.fetch_data_for_symbols()
        
        all_signals = []
        ml_predictions = []
        
        # Process each symbol
        for symbol, df in data.items():
            # Execute strategy
            signals = self.execute_trading_strategy(symbol, df)
            all_signals.extend(signals)
            
            # Get ML predictions
            df_with_indicators = self.strategy.calculate_indicators(df)
            prediction = self.get_ml_prediction(symbol, df_with_indicators)
            if prediction:
                ml_predictions.append(prediction)
        
        # Sort signals by date
        all_signals.sort(key=lambda x: x['date'])
        
        # Process signals chronologically
        executed_trades = []
        for signal in all_signals:
            trade = self.process_trade_signal(signal)
            if trade:
                executed_trades.append(trade)
        
        # Calculate final portfolio value
        final_portfolio_value = self.current_capital
        for symbol, position in self.positions.items():
            if position['quantity'] > 0 and symbol in data:
                latest_price = data[symbol]['Close'].iloc[-1]
                final_portfolio_value += position['quantity'] * latest_price
        
        # Calculate performance metrics
        total_return = final_portfolio_value - self.initial_capital
        return_percentage = (total_return / self.initial_capital) * 100
        
        # Calculate win ratio
        profitable_trades = len([t for t in executed_trades if t.get('pnl', 0) > 0])
        total_trades = len([t for t in executed_trades if 'pnl' in t])
        win_ratio = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': len(executed_trades),
            'win_ratio': win_ratio,
            'executed_trades': executed_trades,
            'ml_predictions': ml_predictions,
            'current_positions': self.positions
        }
        
        logger.info(f"Backtest completed. Return: {return_percentage:.2f}%, Win Ratio: {win_ratio:.2f}%")
        return results
    
    def update_google_sheets(self, results: Dict):
        """
        Update Google Sheets with trading results
        
        Args:
            results: Backtest/trading results
        """
        try:
            # Update trade log
            if results['executed_trades']:
                self.sheets_manager.update_trade_log(results['executed_trades'])
            
            # Update summary
            summary_data = {
                'Initial Capital': results['initial_capital'],
                'Final Portfolio Value': results['final_portfolio_value'],
                'Total Return': results['total_return'],
                'Return %': results['return_percentage'],
                'Total Trades': results['total_trades'],
                'Win Ratio %': results['win_ratio'],
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.sheets_manager.update_summary(summary_data)
            
            # Update ML predictions if available
            if results['ml_predictions']:
                self.sheets_manager.update_ml_predictions(results['ml_predictions'])
            
            logger.info("Google Sheets updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating Google Sheets: {e}")
            self.telegram_bot.send_message(f"❌ Error updating Google Sheets: {e}")
    
    def send_telegram_alerts(self, results: Dict):
        """
        Send Telegram alerts with trading results
        
        Args:
            results: Trading results
        """
        try:
            # Create summary message
            message = f"""
📊 *Algo Trading Update*

💰 *Performance*
• Initial Capital: ₹{results['initial_capital']:,.2f}
• Final Value: ₹{results['final_portfolio_value']:,.2f}
• Total Return: ₹{results['total_return']:,.2f}
• Return %: {results['return_percentage']:.2f}%

📈 *Trading Stats*
• Total Trades: {results['total_trades']}
• Win Ratio: {results['win_ratio']:.2f}%

🤖 *ML Predictions*
• Predictions Generated: {len(results['ml_predictions'])}
            """
            
            # Add recent trades
            if results['executed_trades']:
                recent_trades = results['executed_trades'][-3:]  # Last 3 trades
                message += "\n📋 *Recent Trades*\n"
                for trade in recent_trades:
                    pnl_text = f" (P&L: ₹{trade.get('pnl', 0):.2f})" if 'pnl' in trade else ""
                    message += f"• {trade['action']} {trade['quantity']} {trade['symbol']} @ ₹{trade['price']:.2f}{pnl_text}\n"
            
            self.telegram_bot.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def run_daily_scan(self):
        """
        Run daily scan and trading logic
        """
        logger.info("Starting daily scan...")
        
        try:
            # Run analysis
            results = self.run_backtest()
            
            # Update Google Sheets
            self.update_google_sheets(results)
            
            # Send Telegram alerts
            self.send_telegram_alerts(results)
            
            logger.info("Daily scan completed successfully")
            
        except Exception as e:
            logger.error(f"Error in daily scan: {e}")
            self.telegram_bot.send_message(f"❌ Error in daily scan: {e}")
    
    def start_automated_trading(self):
        """
        Start automated trading with scheduled runs
        """
        logger.info("Starting automated trading system...")
        
        # Schedule daily runs (you can adjust timing)
        schedule.every().day.at("09:30").do(self.run_daily_scan)  # Market opening
        schedule.every().day.at("15:30").do(self.run_daily_scan)  # Market closing
        
        # Send startup notification
        self.telegram_bot.send_message("🚀 Algo Trading System started successfully!")
        
        # Run initial scan
        self.run_daily_scan()
        
        # Keep the system running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user")
                self.telegram_bot.send_message("⏹️ Algo Trading System stopped")
                break
            except Exception as e:
                logger.error(f"Error in automated trading loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """
    Main function to run the trading system
    """
    # Configuration
    config = {
        'symbols': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],  # NIFTY 50 stocks
        'initial_capital': 100000,  # ₹1 Lakh
        'api_key': 'your_alpha_vantage_api_key',  # Replace with your API key
        'sheets_credentials': 'path/to/your/sheets_credentials.json',  # Replace with your path
        'telegram_token': 'your_telegram_bot_token',  # Replace with your token
        'telegram_chat_id': 'your_telegram_chat_id'  # Replace with your chat ID
    }
    
    try:
        # Initialize and run the system
        trading_system = AlgoTradingSystem(config)
        
        # For testing, run a single backtest
        print("Running backtest...")
        results = trading_system.run_backtest()
        
        print(f"\n{'='*50}")
        print("BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ₹{results['final_portfolio_value']:,.2f}")
        print(f"Total Return: ₹{results['total_return']:,.2f}")
        print(f"Return Percentage: {results['return_percentage']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Ratio: {results['win_ratio']:.2f}%")
        
        # Update sheets and send alerts
        trading_system.update_google_sheets(results)
        trading_system.send_telegram_alerts(results)
        
        # Uncomment below to start automated trading
        # trading_system.start_automated_trading()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()