# sheets_manager.py - Google Sheets Integration Module
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread not available. Install with: pip install gspread google-auth")

logger = logging.getLogger(__name__)

class SheetsManager:
    """
    Google Sheets manager for logging trading data
    Handles trade logs, P&L summaries, and ML predictions
    """
    
    def __init__(self, credentials_path: str, spreadsheet_name: str = "Algo Trading System"):
        """
        Initialize Google Sheets manager
        
        Args:
            credentials_path: Path to Google service account credentials JSON
            spreadsheet_name: Name of the Google spreadsheet
        """
        self.credentials_path = credentials_path
        self.spreadsheet_name = spreadsheet_name
        self.gc = None
        self.spreadsheet = None
        self.connected = False
        
        if GSPREAD_AVAILABLE:
            self._connect()
        else:
            logger.error("Google Sheets integration not available - install required packages")
    
    def _connect(self):
        """Connect to Google Sheets API"""
        try:
            # Define the scope
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Load credentials
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=scope)
            self.gc = gspread.authorize(creds)
            
            # Try to open existing spreadsheet or create new one
            try:
                self.spreadsheet = self.gc.open(self.spreadsheet_name)
                logger.info(f"Connected to existing spreadsheet: {self.spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                self.spreadsheet = self.gc.create(self.spreadsheet_name)
                logger.info(f"Created new spreadsheet: {self.spreadsheet_name}")
                self._initialize_sheets()
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            self.connected = False
    
    def _initialize_sheets(self):
        """Initialize sheets with proper headers"""
        if not self.connected:
            return
        
        try:
            # Create Trade Log sheet
            trade_sheet = self.spreadsheet.sheet1
            trade_sheet.update_title("Trade_Log")
            
            trade_headers = [
                "Timestamp", "Symbol", "Action", "Quantity", "Price", "Total Value",
                "RSI", "MA_20", "MA_50", "Reason", "P&L"
            ]
            trade_sheet.update("A1:K1", [trade_headers])
            
            # Create Summary sheet
            summary_sheet = self.spreadsheet.add_worksheet("Summary", 10, 5)
            summary_headers = [["Metric", "Value"]]
            summary_sheet.update("A1:B1", summary_headers)
            
            # Create ML Predictions sheet
            ml_sheet = self.spreadsheet.add_worksheet("ML_Predictions", 100, 8)
            ml_headers = [
                "Timestamp", "Symbol", "Direction", "Confidence",
                "Probability_Up", "Probability_Down", "Actual_Result", "Accuracy"
            ]
            ml_sheet.update("A1:H1", [ml_headers])
            
            # Create Portfolio sheet
            portfolio_sheet = self.spreadsheet.add_worksheet("Portfolio", 20, 6)
            portfolio_headers = [
                "Symbol", "Quantity", "Avg_Price", "Current_Price", "Market_Value", "P&L"
            ]
            portfolio_sheet.update("A1:F1", [portfolio_headers])
            
            logger.info("Initialized all sheets with headers")
            
        except Exception as e:
            logger.error(f"Error initializing sheets: {e}")
    
    def update_trade_log(self, trades: List[Dict]):
        """
        Update trade log sheet with new trades
        
        Args:
            trades: List of trade dictionaries
        """
        if not self.connected or not trades:
            logger.warning("Cannot update trade log - not connected or no trades")
            return
        
        try:
            trade_sheet = self.spreadsheet.worksheet("Trade_Log")
            
            # Prepare data for bulk update
            rows_to_add = []
            for trade in trades:
                row = [
                    trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    trade.get('symbol', ''),
                    trade.get('action', ''),
                    trade.get('quantity', 0),
                    trade.get('price', 0),
                    trade.get('total_value', 0),
                    trade.get('rsi', 0),
                    trade.get('ma_20', 0),
                    trade.get('ma_50', 0),
                    trade.get('reason', ''),
                    trade.get('pnl', 0)
                ]
                rows_to_add.append(row)
            
            # Find next empty row
            values = trade_sheet.get_all_values()
            next_row = len(values) + 1
            
            # Batch update
            range_name = f"A{next_row}:K{next_row + len(rows_to_add) - 1}"
            trade_sheet.update(range_name, rows_to_add)
            
            logger.info(f"Added {len(trades)} trades to trade log")
            
        except Exception as e:
            logger.error(f"Error updating trade log: {e}")
    
    def update_summary(self, summary_data: Dict):
        """
        Update summary sheet with portfolio metrics
        
        Args:
            summary_data: Dictionary with summary metrics
        """
        if not self.connected:
            logger.warning("Cannot update summary - not connected")
            return
        
        try:
            summary_sheet = self.spreadsheet.worksheet("Summary")
            
            # Prepare data for update
            summary_rows = []
            for key, value in summary_data.items():
                if isinstance(value, float):
                    value = round(value, 2)
                summary_rows.append([key.replace('_', ' ').title(), str(value)])
            
            # Clear existing data and update
            summary_sheet.clear()
            summary_sheet.update("A1:B1", [["Metric", "Value"]])
            summary_sheet.update(f"A2:B{len(summary_rows) + 1}", summary_rows)
            
            logger.info("Updated summary sheet")
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
    
    def update_ml_predictions(self, predictions: List[Dict]):
        """
        Update ML predictions sheet
        
        Args:
            predictions: List of ML prediction dictionaries
        """
        if not self.connected or not predictions:
            logger.warning("Cannot update ML predictions - not connected or no predictions")
            return
        
        try:
            ml_sheet = self.spreadsheet.worksheet("ML_Predictions")
            
            # Prepare data for bulk update
            rows_to_add = []
            for pred in predictions:
                row = [
                    pred.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    pred.get('symbol', ''),
                    pred.get('direction', ''),
                    pred.get('confidence', 0),
                    pred.get('probability_up', 0),
                    pred.get('probability_down', 0),
                    pred.get('actual_result', ''),  # To be filled later
                    pred.get('accuracy', '')  # To be calculated later
                ]
                rows_to_add.append(row)
            
            # Find next empty row
            values = ml_sheet.get_all_values()
            next_row = len(values) + 1
            
            # Batch update
            range_name = f"A{next_row}:H{next_row + len(rows_to_add) - 1}"
            ml_sheet.update(range_name, rows_to_add)
            
            logger.info(f"Added {len(predictions)} ML predictions")
            
        except Exception as e:
            logger.error(f"Error updating ML predictions: {e}")
    
    def update_portfolio(self, positions: Dict, current_prices: Dict):
        """
        Update portfolio sheet with current positions
        
        Args:
            positions: Dictionary of current positions
            current_prices: Dictionary of current prices
        """
        if not self.connected:
            logger.warning("Cannot update portfolio - not connected")
            return
        
        try:
            portfolio_sheet = self.spreadsheet.worksheet("Portfolio")
            
            # Prepare portfolio data
            portfolio_rows = []
            for symbol, position in positions.items():
                if position['quantity'] > 0:
                    current_price = current_prices.get(symbol, position['avg_price'])
                    market_value = position['quantity'] * current_price
                    pnl = (current_price - position['avg_price']) * position['quantity']
                    
                    row = [
                        symbol,
                        position['quantity'],
                        round(position['avg_price'], 2),
                        round(current_price, 2),
                        round(market_value, 2),
                        round(pnl, 2)
                    ]
                    portfolio_rows.append(row)
            
            # Clear and update
            portfolio_sheet.clear()
            portfolio_sheet.update("A1:F1", [["Symbol", "Quantity", "Avg_Price", "Current_Price", "Market_Value", "P&L"]])
            
            if portfolio_rows:
                portfolio_sheet.update(f"A2:F{len(portfolio_rows) + 1}", portfolio_rows)
            
            logger.info(f"Updated portfolio with {len(portfolio_rows)} positions")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history from sheets
        
        Returns:
            DataFrame with trade history
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            trade_sheet = self.spreadsheet.worksheet("Trade_Log")
            data = trade_sheet.get_all_records()
            
            if data:
                df = pd.DataFrame(data)
                # Convert timestamp to datetime
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()
    
    def get_ml_predictions_history(self) -> pd.DataFrame:
        """
        Get ML predictions history from sheets
        
        Returns:
            DataFrame with ML predictions history
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            ml_sheet = self.spreadsheet.worksheet("ML_Predictions")
            data = ml_sheet.get_all_records()
            
            if data:
                df = pd.DataFrame(data)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting ML predictions history: {e}")
            return pd.DataFrame()
    
    def calculate_trading_metrics(self) -> Dict:
        """
        Calculate comprehensive trading metrics from sheets data
        
        Returns:
            Dictionary with trading metrics
        """
        trade_df = self.get_trade_history()
        
        if trade_df.empty:
            return {'error': 'No trade data available'}
        
        try:
            # Basic metrics
            total_trades = len(trade_df)
            buy_trades = trade_df[trade_df['Action'] == 'BUY']
            sell_trades = trade_df[trade_df['Action'] == 'SELL']
            
            # P&L calculation
            total_pnl = trade_df['P&L'].sum() if 'P&L' in trade_df.columns else 0
            profitable_trades = len(trade_df[trade_df['P&L'] > 0]) if 'P&L' in trade_df.columns else 0
            
            # Win ratio
            completed_trades = min(len(buy_trades), len(sell_trades))
            win_ratio = (profitable_trades / completed_trades * 100) if completed_trades > 0 else 0
            
            # Average trade size
            avg_trade_value = trade_df['Total Value'].mean() if 'Total Value' in trade_df.columns else 0
            
            # Trading frequency
            if len(trade_df) > 1:
                date_range = (trade_df['Timestamp'].max() - trade_df['Timestamp'].min()).days
                trades_per_day = total_trades / max(date_range, 1)
            else:
                trades_per_day = 0
            
            metrics = {
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_pnl': round(total_pnl, 2),
                'profitable_trades': profitable_trades,
                'win_ratio': round(win_ratio, 2),
                'avg_trade_value': round(avg_trade_value, 2),
                'trades_per_day': round(trades_per_day, 2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {'error': str(e)}
    
    def backup_to_csv(self, backup_dir: str = "./backups"):
        """
        Backup all sheets to CSV files
        
        Args:
            backup_dir: Directory to save backup files
        """
        if not self.connected:
            logger.warning("Cannot backup - not connected")
            return
        
        import os
        
        try:
            # Create backup directory
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup each sheet
            worksheets = self.spreadsheet.worksheets()
            for sheet in worksheets:
                data = sheet.get_all_records()
                if data:
                    df = pd.DataFrame(data)
                    filename = f"{backup_dir}/{sheet.title}_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Backed up {sheet.title} to {filename}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")

# Mock implementation for testing without Google Sheets
class MockSheetsManager:
    """
    Mock Google Sheets manager for testing without actual Google Sheets connection
    """
    
    def __init__(self, credentials_path: str = "", spreadsheet_name: str = "Algo Trading System"):
        self.connected = True
        self.trade_log = []
        self.summary = {}
        self.ml_predictions = []
        self.portfolio = {}
        logger.info("Mock Sheets Manager initialized (for testing)")
    
    def update_trade_log(self, trades: List[Dict]):
        self.trade_log.extend(trades)
        logger.info(f"Mock: Added {len(trades)} trades to trade log")
    
    def update_summary(self, summary_data: Dict):
        self.summary.update(summary_data)
        logger.info("Mock: Updated summary")
    
    def update_ml_predictions(self, predictions: List[Dict]):
        self.ml_predictions.extend(predictions)
        logger.info(f"Mock: Added {len(predictions)} ML predictions")
    
    def update_portfolio(self, positions: Dict, current_prices: Dict):
        self.portfolio = positions.copy()
        logger.info(f"Mock: Updated portfolio with {len(positions)} positions")
    
    def get_trade_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
    
    def calculate_trading_metrics(self) -> Dict:
        if not self.trade_log:
            return {'total_trades': 0, 'win_ratio': 0}
        
        return {
            'total_trades': len(self.trade_log),
            'win_ratio': 65.5,  # Mock win ratio
            'total_pnl': 15000,  # Mock P&L
            'profitable_trades': 13
        }

# Test the sheets manager
if __name__ == "__main__":
    # Test with mock manager (no actual Google Sheets connection needed)
    manager = MockSheetsManager()
    
    # Test trade log update
    sample_trades = [
        {
            'timestamp': datetime.now(),
            'symbol': 'RELIANCE.NS',
            'action': 'BUY',
            'quantity': 10,
            'price': 2500.0,
            'total_value': 25000.0,
            'rsi': 25.5,
            'ma_20': 2480.0,
            'ma_50': 2450.0,
            'reason': 'RSI oversold + MA crossover',
            'pnl': 0
        }
    ]
    
    manager.update_trade_log(sample_trades)
    
    # Test summary update
    summary_data = {
        'Initial_Capital': 100000,
        'Current_Value': 115000,
        'Total_Return': 15000,
        'Return_Percentage': 15.0,
        'Win_Ratio': 65.5
    }
    
    manager.update_summary(summary_data)
    
    # Test ML predictions
    ml_predictions = [
        {
            'timestamp': datetime.now(),
            'symbol': 'TCS.NS',
            'direction': 'UP',
            'confidence': 0.75,
            'probability_up': 0.75,
            'probability_down': 0.25
        }
    ]
    
    manager.update_ml_predictions(ml_predictions)
    
    print("Mock Sheets Manager test completed successfully!")
    print(f"Trade Log: {len(manager.trade_log)} trades")
    print(f"Summary: {len(manager.summary)} metrics")
    print(f"ML Predictions: {len(manager.ml_predictions)} predictions")