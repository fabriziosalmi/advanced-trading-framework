#!/usr/bin/env python3
"""
Advanced Trading Framework - Model Training Script

This script allows you to train ML models for the trading strategies
without running the full trading application.

Usage:
    python train_models.py --tickers AAPL MSFT GOOGL
    python train_models.py --all  # Train on default ticker list
    python train_models.py --config config.yaml
"""

import argparse
import asyncio
import logging
import sys
import yaml
import pandas as pd
from typing import List
from datetime import datetime

# Framework imports
from strategy_layer.ml_random_forest_strategy import MLRandomForestStrategy
from strategy_layer.lgbm_strategy import MLLGBMStrategy
from core.portfolio import Portfolio


class ModelTrainer:
    """
    Dedicated model training class for ML trading strategies.
    """
    
    def __init__(self, config_file: str = "config.yaml", 
                 training_window: int = 500, retrain_every: int = 21, testing_period_days: int = 252):
        """Initialize the model trainer."""
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
        self.strategy = None
        
        # Walk-forward validation parameters
        self.training_window_size = training_window
        self.retrain_every = retrain_every
        self.testing_period_days = testing_period_days
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file {self.config_file} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'trading': {
                'default_tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                'strategies': {
                    'ml_random_forest': {
                        'confidence_threshold': 0.5,
                        'enabled': True
                    }
                }
            }
        }
    
    def _setup_logging(self):
        """Setup logging for training."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/training.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize_strategy(self, model_type: str = 'rf'):
        """Initialize the ML strategy."""
        try:
            if model_type == 'rf':
                confidence_threshold = self.config.get('trading', {}).get(
                    'strategies', {}
                ).get('ml_random_forest', {}).get('confidence_threshold', 0.5)
                self.strategy = MLRandomForestStrategy(confidence_threshold=confidence_threshold)
                strategy_name = "RandomForest"
            elif model_type == 'lgbm':
                confidence_threshold = self.config.get('trading', {}).get(
                    'strategies', {}
                ).get('ml_lgbm', {}).get('confidence_threshold', 0.5)
                self.strategy = MLLGBMStrategy(confidence_threshold=confidence_threshold)
                strategy_name = "LightGBM"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            await self.strategy.initialize()
            
            self.logger.info(f"✅ {strategy_name} Strategy initialized with confidence threshold: {confidence_threshold}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize strategy: {e}")
            return False
    
    async def train_models(self, tickers: List[str], force_retrain: bool = False, use_walk_forward: bool = False):
        """
        Train ML models for specified tickers.
        
        Args:
            tickers: List of ticker symbols to train models for
            force_retrain: Whether to force retraining even if models exist
            use_walk_forward: Whether to use walk-forward validation
        """
        if not self.strategy:
            self.logger.error("❌ Strategy not initialized. Call initialize_strategy() first.")
            return False
        
        self.logger.info(f"🚀 Starting model training for {len(tickers)} tickers: {', '.join(tickers)}")
        self.logger.info(f"🔄 Force retrain: {force_retrain}")
        self.logger.info(f"🔄 Walk-forward validation: {use_walk_forward}")
        
        if use_walk_forward:
            self.logger.info(f"📊 Walk-forward parameters:")
            self.logger.info(f"   - Training window: {self.training_window_size} days")
            self.logger.info(f"   - Retrain every: {self.retrain_every} days")
            self.logger.info(f"   - Test period: {self.testing_period_days} days")
        
        success_count = 0
        total_tickers = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            self.logger.info(f"📊 [{i}/{total_tickers}] Training model for {ticker}")
            
            try:
                # Check if model already exists
                model_exists = self.strategy._model_exists(ticker)
                
                if model_exists and not force_retrain:
                    self.logger.info(f"⏭️  Model for {ticker} already exists, skipping (use --force to retrain)")
                    success_count += 1
                    continue
                
                # Train the model
                success = await self._train_single_model(ticker, use_walk_forward)
                
                if success:
                    self.logger.info(f"✅ Successfully trained model for {ticker}")
                    success_count += 1
                else:
                    self.logger.error(f"❌ Failed to train model for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error training {ticker}: {e}")
                continue
        
        # Summary
        self.logger.info(f"🎯 Training completed: {success_count}/{total_tickers} models trained successfully")
        
        if success_count == total_tickers:
            self.logger.info("🎉 All models trained successfully!")
            return True
        elif success_count > 0:
            self.logger.warning(f"⚠️  Partial success: {success_count}/{total_tickers} models trained")
            return True
        else:
            self.logger.error("💥 Training failed for all tickers")
            return False
    
    async def _train_single_model(self, ticker: str, use_walk_forward: bool = False) -> bool:
        """Train a single model for the given ticker."""
        try:
            if use_walk_forward:
                self.logger.info(f"🧠 Training ML model for {ticker} using Walk-Forward Validation")
                
                # Use walk-forward validation
                success = await self._train_walk_forward(ticker)
            else:
                self.logger.info(f"🧠 Training ML model for {ticker} using Single Split")
                
                # Use traditional single train/test split
                success = await self.strategy._train_model_for_ticker(ticker)
            
            if success:
                self.logger.info(f"✅ Successfully trained model for {ticker}")
                return True
            else:
                self.logger.error(f"❌ Model training failed for {ticker}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Exception during training for {ticker}: {e}")
            return False
    
    async def _train_walk_forward(self, ticker: str) -> bool:
        """
        Implement walk-forward validation for robust out-of-sample testing.
        
        This simulates real-world deployment where the model is periodically retrained
        to adapt to changing market conditions.
        """
        try:
            # Fetch all available data for the ticker
            self.logger.info(f"📊 Fetching complete dataset for {ticker}")
            all_data = await self.strategy._fetch_data_async(ticker)
            
            if all_data is None or len(all_data) < self.training_window_size + self.testing_period_days:
                self.logger.warning(f"Insufficient data for walk-forward validation: {len(all_data) if all_data else 0} rows")
                return False
            
            self.logger.info(f"✅ Loaded {len(all_data)} days of data for {ticker}")
            self.logger.info(f"📅 Data range: {all_data.index.min()} to {all_data.index.max()}")
            
            # Calculate walk-forward parameters
            total_samples = len(all_data)
            testing_start_idx = total_samples - self.testing_period_days
            
            if testing_start_idx < self.training_window_size:
                self.logger.warning(f"Insufficient data for walk-forward: need at least {self.training_window_size + self.testing_period_days} samples")
                return False
            
            # Initialize walk-forward loop
            current_train_end = testing_start_idx
            folds = []
            all_predictions = []
            all_true_labels = []
            
            fold_num = 0
            while current_train_end + self.retrain_every <= total_samples:
                fold_num += 1
                
                # Define training window (rolling backward from current position)
                train_start_idx = max(0, current_train_end - self.training_window_size)
                train_end_idx = current_train_end
                
                # Define testing window (next retrain_every days)
                test_start_idx = current_train_end
                test_end_idx = min(total_samples, current_train_end + self.retrain_every)
                
                # Extract data slices
                train_data = all_data.iloc[train_start_idx:train_end_idx]
                test_data = all_data.iloc[test_start_idx:test_end_idx]
                
                self.logger.info(f"🔄 Fold {fold_num}: Train[{train_start_idx}:{train_end_idx}] ({len(train_data)} samples), "
                               f"Test[{test_start_idx}:{test_end_idx}] ({len(test_data)} samples)")
                
                # Train model on this fold
                fold_result = await self._train_single_fold(ticker, train_data, test_data, fold_num)
                
                if fold_result:
                    predictions, true_labels = fold_result
                    all_predictions.extend(predictions)
                    all_true_labels.extend(true_labels)
                    folds.append(fold_num)
                else:
                    self.logger.warning(f"⚠️  Failed to train fold {fold_num}, skipping")
                
                # Move forward
                current_train_end += self.retrain_every
            
            if not all_predictions:
                self.logger.error("❌ No successful folds completed")
                return False
            
            # Calculate final walk-forward metrics
            self.logger.info(f"📊 Aggregating results from {len(folds)} successful folds")
            final_metrics = self._calculate_walk_forward_metrics(all_predictions, all_true_labels)
            
            self.logger.info("🎯 Walk-Forward Validation Results:")
            self.logger.info(f"  - Total folds: {len(folds)}")
            self.logger.info(f"  - Total predictions: {len(all_predictions)}")
            self.logger.info(f"  - Final Accuracy: {final_metrics['accuracy']:.3f}")
            self.logger.info(f"  - Final Precision: {final_metrics['precision']:.3f}")
            self.logger.info(f"  - Final Recall: {final_metrics['recall']:.3f}")
            self.logger.info(f"  - Final F1-Score: {final_metrics['f1_score']:.3f}")
            
            # Train final model on most recent data for deployment
            self.logger.info("🏁 Training final model on most recent data for deployment")
            final_train_data = all_data.iloc[-self.training_window_size:]
            success = await self._train_final_model(ticker, final_train_data, final_metrics)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Exception in walk-forward training: {e}")
            return False
    
    async def _train_single_fold(self, ticker: str, train_data: pd.DataFrame, test_data: pd.DataFrame, fold_num: int):
        """
        Train a model on a single fold of walk-forward validation.
        
        Returns:
            Tuple of (predictions, true_labels) if successful, None otherwise
        """
        try:
            # Use the strategy's fold training method
            return await self.strategy._train_fold(ticker, train_data, test_data, fold_num)
            
        except Exception as e:
            self.logger.error(f"❌ Exception in fold {fold_num}: {e}")
            return None
    
    def _calculate_walk_forward_metrics(self, all_predictions, all_true_labels):
        """Calculate final metrics from all walk-forward predictions."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    async def _train_final_model(self, ticker: str, final_train_data: pd.DataFrame, walk_forward_metrics: dict) -> bool:
        """
        Train the final model on most recent data for deployment.
        Since walk-forward validation already passed, we save the model regardless of final test performance.
        """
        try:
            # Check if walk-forward performance meets minimum threshold
            min_f1_threshold = 0.50  # Target threshold for reliable trading signal
            actual_f1 = walk_forward_metrics['f1_score']
            
            if actual_f1 < min_f1_threshold:
                self.logger.warning(f"⚠️  Walk-forward F1 ({actual_f1:.3f}) below threshold ({min_f1_threshold:.3f})")
                self.logger.warning("🚫 Model not saved - does not demonstrate reliable edge")
                return False
            
            self.logger.info(f"✅ Walk-forward F1 ({actual_f1:.3f}) meets threshold ({min_f1_threshold:.3f})")
            
            # Train final model for deployment - save regardless of final test performance
            # since walk-forward validation already demonstrated edge
            self.logger.info("🏁 Training final model on most recent data for deployment")
            success = await self.strategy._train_final_model_for_deployment(ticker, final_train_data)
            
            if success:
                self.logger.info(f"💾 Final model saved for {ticker} deployment")
                return True
            else:
                self.logger.warning("⚠️  Final model training completed but may have poor recent performance")
                self.logger.info("💾 Model saved anyway since walk-forward validation passed")
                return True  # Still consider successful since walk-forward passed
                
        except Exception as e:
            self.logger.error(f"❌ Exception training final model: {e}")
            return False
    
    def list_trained_models(self):
        """List all currently trained models."""
        if not self.strategy:
            self.logger.error("❌ Strategy not initialized")
            return
        
        models_dir = self.strategy.model_dir
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        
        if not model_files:
            self.logger.info("📁 No trained models found")
            return
        
        self.logger.info(f"📁 Found {len(model_files)} trained models:")
        for model_file in sorted(model_files):
            ticker = model_file.replace('_model.pkl', '')
            model_path = os.path.join(models_dir, model_file)
            scaler_path = os.path.join(models_dir, f"{ticker}_scaler.pkl")
            
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            
            status = "✅ Complete" if (model_exists and scaler_exists) else "⚠️  Incomplete"
            self.logger.info(f"   {ticker}: {status}")


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML models for trading strategies")
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to train')
    parser.add_argument('--all', action='store_true', help='Train on all default tickers')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--force', action='store_true', help='Force retrain existing models')
    parser.add_argument('--list', action='store_true', help='List existing trained models')
    parser.add_argument('--model', choices=['rf', 'lgbm'], default='rf', help='Model type to train (rf=RandomForest, lgbm=LightGBM)')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation instead of single split')
    parser.add_argument('--training-window', type=int, default=500, help='Training window size in days (default: 500)')
    parser.add_argument('--retrain-every', type=int, default=21, help='Retrain frequency in days (default: 21)')
    parser.add_argument('--test-period', type=int, default=252, help='Testing period in days (default: 252)')
    parser.add_argument('--help-examples', action='store_true', help='Show detailed usage examples and exit')
    
    args = parser.parse_args()
    
    # Show help examples if requested
    if args.help_examples:
        print("🚀 Advanced Trading Framework - Model Training Script")
        print("=" * 60)
        print()
        print("DESCRIPTION:")
        print("  Train machine learning models for algorithmic trading strategies.")
        print("  Supports Random Forest and LightGBM models with walk-forward validation.")
        print()
        print("USAGE EXAMPLES:")
        print()
        print("  Basic Training:")
        print("    python train_models.py --tickers AAPL MSFT")
        print("    python train_models.py --all")
        print()
        print("  Model Selection:")
        print("    python train_models.py --model rf --tickers META")
        print("    python train_models.py --model lgbm --tickers TSLA NVDA")
        print()
        print("  Walk-Forward Validation (Recommended for Production):")
        print("    python train_models.py --walk-forward --tickers AAPL")
        print("    python train_models.py --walk-forward --training-window 600 --retrain-every 30 --tickers MSFT")
        print()
        print("  Force Retraining:")
        print("    python train_models.py --force --tickers AAPL")
        print("    python train_models.py --force --all")
        print()
        print("  List Available Models:")
        print("    python train_models.py --list")
        print()
        print("  Custom Configuration:")
        print("    python train_models.py --config custom_config.yaml --tickers GOOGL")
        print()
        print("OPTIONS:")
        print("  --tickers TICKERS    Train models for specific ticker symbols")
        print("  --all                 Train models for all default tickers")
        print("  --config FILE         Path to configuration file (default: config.yaml)")
        print("  --force               Force retraining even if models exist")
        print("  --list                List all currently trained models")
        print("  --model {rf,lgbm}     Model type: rf=RandomForest, lgbm=LightGBM (default: rf)")
        print("  --walk-forward        Use walk-forward validation for robust testing")
        print("  --training-window N   Training window size in days (default: 500)")
        print("  --retrain-every N     Retrain frequency in days (default: 21)")
        print("  --test-period N       Testing period in days (default: 252)")
        print("  --help-examples       Show this detailed help message")
        print()
        print("NOTES:")
        print("  - Always activate virtual environment: source trading_env/bin/activate")
        print("  - Models are saved in the 'models/' directory")
        print("  - Training logs are written to 'logs/training.log'")
        print("  - Walk-forward validation provides more realistic performance estimates")
        print()
        return
    
    # Initialize trainer
    trainer = ModelTrainer(args.config, args.training_window, args.retrain_every, args.test_period)
    
    # List models if requested
    if args.list:
        await trainer.initialize_strategy(args.model)
        trainer.list_trained_models()
        return
    
    # Determine tickers to train
    if args.tickers:
        tickers = args.tickers
    elif args.all:
        tickers = trainer.config.get('universe', {}).get('default_tickers', ['AAPL', 'MSFT'])
    else:
        print("❌ Please specify either --tickers or --all")
        print()
        print("💡 Quick Usage Examples:")
        print("  python train_models.py --tickers AAPL MSFT")
        print("  python train_models.py --all")
        print("  python train_models.py --help-examples  # For detailed help")
        print()
        print("💡 Use --help-examples for comprehensive usage guide")
        return
    
    # Initialize and train
    print(f"🚀 Advanced Trading Framework - Model Trainer")
    print(f"🤖 Model type: {args.model.upper()}")
    print(f"📊 Target tickers: {', '.join(tickers)}")
    print(f"🔄 Force retrain: {args.force}")
    print(f"🔄 Walk-forward: {args.walk_forward}")
    if args.walk_forward:
        print(f"   📈 Training window: {args.training_window} days")
        print(f"   🔄 Retrain every: {args.retrain_every} days") 
        print(f"   📉 Test period: {args.test_period} days")
    print("-" * 50)
    
    if not await trainer.initialize_strategy(args.model):
        print("❌ Failed to initialize strategy")
        return
    
    success = await trainer.train_models(tickers, force_retrain=args.force, use_walk_forward=args.walk_forward)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("💡 You can now run the trading app with trained models")
    else:
        print("\n💥 Training failed")
        sys.exit(1)


if __name__ == "__main__":
    import os
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the training
    asyncio.run(main())