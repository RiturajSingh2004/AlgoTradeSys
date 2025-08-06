# telegram_bot.py - Telegram Bot Integration Module
import logging
import requests
import json
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for sending trading alerts and notifications
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram bot
        
        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bool(bot_token and chat_id and bot_token != 'your_telegram_bot_token')
        
        if self.enabled:
            # Test connection
            if self._test_connection():
                logger.info("Telegram bot connected successfully")
            else:
                logger.warning("Telegram bot connection failed")
                self.enabled = False
        else:
            logger.info("Telegram bot disabled (no valid token/chat_id provided)")
    
    def _test_connection(self) -> bool:
        """
        Test Telegram bot connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    logger.info(f"Connected to bot: {bot_info['result']['username']}")
                    return True
            
            logger.error(f"Bot connection failed: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error testing bot connection: {e}")
            return False
    
    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        Send a message to Telegram chat
        
        Args:
            message: Message text to send
            parse_mode: Message formatting ('Markdown' or 'HTML')
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled - would send: {message}")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {result.get('description', 'Unknown error')}")
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_alert(self, trade: Dict) -> bool:
        """
        Send trading alert message
        
        Args:
            trade: Trade dictionary with details
            
        Returns:
            True if alert sent successfully
        """
        try:
            # Format trade alert message
            action_emoji = "🟢" if trade.get('action') == 'BUY' else "🔴"
            
            message = f"""
{action_emoji} *TRADE ALERT*

📊 *{trade.get('symbol', 'Unknown')}*
🎯 Action: *{trade.get('action', 'Unknown')}*
💰 Price: ₹{trade.get('price', 0):,.2f}
📈 Quantity: {trade.get('quantity', 0):,}
💵 Value: ₹{trade.get('total_value', 0):,.2f}

📋 *Technical Indicators*
• RSI: {trade.get('rsi', 0):.2f}
• 20-MA: ₹{trade.get('ma_20', 0):,.2f}
• 50-MA: ₹{trade.get('ma_50', 0):,.2f}

🔍 Reason: {trade.get('reason', 'Strategy signal')}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
            return False
    
    def send_portfolio_update(self, portfolio_data: Dict) -> bool:
        """
        Send portfolio performance update
        
        Args:
            portfolio_data: Portfolio performance data
            
        Returns:
            True if update sent successfully
        """
        try:
            # Determine performance emoji
            return_pct = portfolio_data.get('return_percentage', 0)
            perf_emoji = "📈" if return_pct > 0 else "📉" if return_pct < 0 else "➡️"
            
            message = f"""
{perf_emoji} *PORTFOLIO UPDATE*

💰 *Performance*
• Initial: ₹{portfolio_data.get('initial_capital', 0):,.2f}
• Current: ₹{portfolio_data.get('final_portfolio_value', 0):,.2f}
• P&L: ₹{portfolio_data.get('total_return', 0):,.2f}
• Return: {return_pct:.2f}%

📊 *Trading Stats*
• Total Trades: {portfolio_data.get('total_trades', 0)}
• Win Ratio: {portfolio_data.get('win_ratio', 0):.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending portfolio update: {e}")
            return False
    
    def send_ml_prediction_alert(self, prediction: Dict) -> bool:
        """
        Send ML prediction alert
        
        Args:
            prediction: ML prediction data
            
        Returns:
            True if alert sent successfully
        """
        try:
            direction_emoji = "🚀" if prediction.get('direction') == 'UP' else "📉"
            confidence = prediction.get('confidence', 0)
            confidence_emoji = "🔥" if confidence > 0.8 else "✅" if confidence > 0.6 else "⚠️"
            
            message = f"""
🤖 *ML PREDICTION*

{direction_emoji} *{prediction.get('symbol', 'Unknown')}*
📊 Direction: *{prediction.get('direction', 'Unknown')}*

{confidence_emoji} *Confidence Analysis*
• Overall: {confidence:.2%}
• UP Probability: {prediction.get('probability_up', 0):.2%}
• DOWN Probability: {prediction.get('probability_down', 0):.2%}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending ML prediction alert: {e}")