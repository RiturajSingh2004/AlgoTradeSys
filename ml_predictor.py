# ml_predictor.py - Machine Learning Prediction Module
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Machine Learning predictor for stock price movements
    Uses technical indicators to predict next-day price direction
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML predictor
        
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'decision_tree')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"ML Predictor initialized with {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with prepared features
        """
        df = df.copy()
        
        # Ensure we have required indicators
        required_indicators = ['RSI', 'MA_20', 'MA_50', 'MACD', 'MACD_Signal']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
            return df
        
        # Create additional features
        df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
        df['Price_MA50_Ratio'] = df['Close'] / df['MA_50']
        df['MA_Ratio'] = df['MA_20'] / df['MA_50']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Price momentum features
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_3d'] = df['Close'].pct_change(3)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        
        # Volatility features
        df['Volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        df['Volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        
        # RSI features
        df['RSI_Change'] = df['RSI'].diff()
        df['RSI_MA'] = df['RSI'].rolling(5).mean()
        
        # MACD features
        df['MACD_Signal_Diff'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Change'] = df['MACD'].diff()
        
        # Bollinger Bands features (if available)
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            df['BB_Squeeze'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Target variable: next day price direction
        df['Next_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)  # 1 for up, 0 for down
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for ML model
        
        Args:
            df: DataFrame with prepared features
            
        Returns:
            List of selected feature column names
        """
        # Core technical indicators
        features = [
            'RSI', 'MA_20', 'MA_50', 'MACD', 'MACD_Signal',
            'Price_MA20_Ratio', 'Price_MA50_Ratio', 'MA_Ratio',
            'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_Change_1d', 'Price_Change_3d', 'Price_Change_5d',
            'Volatility_5d', 'Volatility_10d',
            'RSI_Change', 'RSI_MA', 'MACD_Signal_Diff', 'MACD_Change'
        ]
        
        # Add Bollinger Bands features if available
        if 'BB_Position' in df.columns:
            features.extend(['BB_Position', 'BB_Squeeze'])
        
        # Filter features that actually exist in the DataFrame
        available_features = [f for f in features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} features for ML model")
        return available_features
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """
        Train the ML model
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting ML model training...")
        
        # Prepare features
        df_features = self.prepare_features(df)
        self.feature_columns = self.select_features(df_features)
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        if len(df_clean) < 50:
            logger.error("Insufficient data for training (need at least 50 samples)")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Prepare X and y
        X = df_clean[self.feature_columns]
        y = df_clean['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        cv_accuracy = cv_scores.mean()
        
        self.is_trained = True
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance = importance_df.to_dict('records')
        
        results = {
            'success': True,
            'model_type': self.model_type,
            'accuracy': accuracy,
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_scores.std(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_columns),
            'feature_importance': feature_importance
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}, CV Accuracy: {cv_accuracy:.4f}")
        return results
    
    def predict_next_day(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Predict next day price movement
        
        Args:
            df: DataFrame with latest OHLCV data and indicators
            
        Returns:
            Prediction dictionary or None
        """
        if not self.is_trained:
            # Auto-train if not trained
            logger.info("Model not trained, training now...")
            train_result = self.train_model(df)
            if not train_result['success']:
                return None
        
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            
            # Get latest data point
            latest_data = df_features.iloc[-1:][self.feature_columns]
            
            # Handle missing values
            if latest_data.isnull().any().any():
                logger.warning("Missing values in prediction data, filling with median")
                latest_data = latest_data.fillna(latest_data.median())
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_data)
            
            # Make prediction
            prediction = self.model.predict(latest_scaled)[0]
            prediction_proba = self.model.predict_proba(latest_scaled)[0]
            
            # Get confidence (max probability)
            confidence = max(prediction_proba)
            
            result = {
                'prediction': int(prediction),
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': confidence,
                'probability_up': prediction_proba[1],
                'probability_down': prediction_proba[0],
                'timestamp': pd.Timestamp.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
    
    def get_model_performance(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on recent data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Performance metrics dictionary
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            df_clean = df_features.dropna()
            
            # Use recent data for evaluation (last 30% of data)
            recent_cutoff = int(len(df_clean) * 0.7)
            recent_data = df_clean.iloc[recent_cutoff:]
            
            if len(recent_data) < 10:
                return {'error': 'Insufficient recent data'}
            
            X_recent = recent_data[self.feature_columns]
            y_recent = recent_data['Target']
            
            # Scale and predict
            X_recent_scaled = self.scaler.transform(X_recent)
            predictions = self.model.predict(X_recent_scaled)
            probabilities = self.model.predict_proba(X_recent_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_recent, predictions)
            
            # Calculate directional accuracy (more relevant for trading)
            correct_predictions = (predictions == y_recent).sum()
            total_predictions = len(predictions)
            
            # Calculate average confidence
            avg_confidence = np.mean([max(p) for p in probabilities])
            
            return {
                'recent_accuracy': accuracy,
                'correct_predictions': int(correct_predictions),
                'total_predictions': int(total_predictions),
                'average_confidence': avg_confidence,
                'evaluation_period': f"Last {len(recent_data)} days"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {'error': str(e)}
    
    def get_feature_analysis(self) -> Optional[Dict]:
        """
        Get analysis of feature importance
        
        Returns:
            Feature analysis dictionary or None
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Group features by category
        categories = {
            'Price_Momentum': ['Price_Change_1d', 'Price_Change_3d', 'Price_Change_5d'],
            'Technical_Indicators': ['RSI', 'MACD', 'MACD_Signal', 'RSI_Change', 'RSI_MA'],
            'Moving_Averages': ['MA_20', 'MA_50', 'Price_MA20_Ratio', 'Price_MA50_Ratio', 'MA_Ratio'],
            'Volatility': ['Volatility_5d', 'Volatility_10d'],
            'Volume': ['Volume_Ratio'],
            'Price_Ratios': ['High_Low_Ratio', 'Open_Close_Ratio'],
            'Bollinger_Bands': ['BB_Position', 'BB_Squeeze']
        }
        
        category_importance = {}
        for category, features in categories.items():
            category_features = [f for f in features if f in importance_df['feature'].values]
            if category_features:
                total_importance = importance_df[importance_df['feature'].isin(category_features)]['importance'].sum()
                category_importance[category] = total_importance
        
        return {
            'top_features': importance_df.head(10).to_dict('records'),
            'category_importance': category_importance,
            'total_features': len(self.feature_columns)
        }

# Test the ML predictor
if __name__ == "__main__":
    # Create sample data with technical indicators
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample price data
    prices = []
    base_price = 1000
    for i in range(len(dates)):
        change = np.random.normal(0, 0.02)
        base_price *= (1 + change)
        prices.append(base_price)
    
    # Create sample DataFrame with technical indicators
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(len(dates))]
    })
    
    # Add technical indicators (simplified for testing)
    sample_data['RSI'] = 50 + np.random.normal(0, 15, len(sample_data))
    sample_data['MA_20'] = sample_data['Close'].rolling(20).mean()
    sample_data['MA_50'] = sample_data['Close'].rolling(50).mean()
    sample_data['MACD'] = sample_data['Close'].ewm(12).mean() - sample_data['Close'].ewm(26).mean()
    sample_data['MACD_Signal'] = sample_data['MACD'].ewm(9).mean()
    sample_data['Volume_MA'] = sample_data['Volume'].rolling(20).mean()
    
    # Test the ML predictor
    predictor = MLPredictor(model_type='random_forest')
    
    print("Training ML model...")
    training_results = predictor.train_model(sample_data)
    
    if training_results['success']:
        print(f"\n{'='*50}")
        print("ML MODEL TRAINING RESULTS")
        print(f"{'='*50}")
        print(f"Model Type: {training_results['model_type']}")
        print(f"Accuracy: {training_results['accuracy']:.4f}")
        print(f"Cross-Validation Accuracy: {training_results['cv_accuracy']:.4f} ± {training_results['cv_std']:.4f}")
        print(f"Training Samples: {training_results['train_samples']}")
        print(f"Test Samples: {training_results['test_samples']}")
        print(f"Feature Count: {training_results['feature_count']}")
        
        # Show top features
        if training_results['feature_importance']:
            print(f"\nTop 5 Important Features:")
            for i, feat in enumerate(training_results['feature_importance'][:5]):
                print(f"{i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        # Make a prediction
        print(f"\n{'='*30}")
        print("MAKING PREDICTION")
        print(f"{'='*30}")
        prediction = predictor.predict_next_day(sample_data)
        
        if prediction:
            print(f"Direction: {prediction['direction']}")
            print(f"Confidence: {prediction['confidence']:.4f}")
            print(f"Probability UP: {prediction['probability_up']:.4f}")
            print(f"Probability DOWN: {prediction['probability_down']:.4f}")
        
        # Model performance
        print(f"\n{'='*30}")
        print("MODEL PERFORMANCE")
        print(f"{'='*30}")
        performance = predictor.get_model_performance(sample_data)
        if 'error' not in performance:
            print(f"Recent Accuracy: {performance['recent_accuracy']:.4f}")
            print(f"Correct Predictions: {performance['correct_predictions']}/{performance['total_predictions']}")
            print(f"Average Confidence: {performance['average_confidence']:.4f}")
    else:
        print(f"Training failed: {training_results.get('error', 'Unknown error')}")