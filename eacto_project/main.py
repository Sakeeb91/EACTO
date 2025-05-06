#!/usr/bin/env python
"""
Main script to run the EACTO system.
This script demonstrates a complete workflow from data ingestion to backtesting.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from eacto.data_ingestion.fetcher import DataFetcher
from eacto.data_ingestion.preprocessor import DataPreprocessor
from eacto.entropy_calculation.calculators import EntropyCalculator
from eacto.system_model.predictors import GARCHPredictor, QuantileRegressionPredictor, MLPredictor
from eacto.mpc_controller.controller import MPCController
from eacto.risk_management.threshold_manager import ThresholdManager
from eacto.backtesting.engine import BacktestEngine
from eacto.backtesting.metrics import evaluate_backtest_results
from eacto.utils.helpers import setup_logging, create_output_directory, format_summary_stats, save_results
from eacto.utils.plotting import create_dashboard

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EACTO System')
    
    # Data arguments
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-saved-data', action='store_true', help='Use saved data if available')
    parser.add_argument('--data-file', type=str, default='data/processed/data.csv', help='Path to save/load data')
    
    # System configuration
    parser.add_argument('--predictor', type=str, default='garch', choices=['garch', 'qr', 'ml'], 
                       help='System dynamics predictor to use')
    parser.add_argument('--target-breach-prob', type=float, default=0.05, 
                       help='Target breach probability (p*)')
    
    # Backtest configuration
    parser.add_argument('--test-start', type=str, default=None, 
                       help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--warmup-periods', type=int, default=252, 
                       help='Number of periods for initial training')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results', 
                       help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Logging level')
    
    return parser.parse_args()

def load_or_fetch_data(args):
    """Load saved data or fetch new data."""
    if args.use_saved_data and os.path.exists(args.data_file):
        logging.info(f"Loading data from {args.data_file}")
        return pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    
    logging.info(f"Fetching data for {args.ticker} from {args.start_date} to {args.end_date}")
    
    # Create data fetcher and fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_from_yahoo(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if data is None or data.empty:
        logging.error("Failed to fetch data")
        return None
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.calculate_returns(data, price_col='close', method='log')
    data = preprocessor.calculate_realized_volatility(data, window=20, returns_col='log_return')
    
    # Save processed data
    if not os.path.exists(os.path.dirname(args.data_file)):
        os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
    
    data.to_csv(args.data_file)
    logging.info(f"Saved processed data to {args.data_file}")
    
    return data

def create_predictor(args):
    """Create the system dynamics predictor based on arguments."""
    predictor_type = args.predictor.lower()
    
    if predictor_type == 'garch':
        return GARCHPredictor(p=1, q=1, mean_model="Constant", use_entropy_in_variance=True)
    elif predictor_type == 'qr':
        return QuantileRegressionPredictor(quantile=0.05, lags=5, include_entropy=True)
    elif predictor_type == 'ml':
        return MLPredictor(mode='classifier', n_estimators=100, lags=5, include_entropy=True)
    else:
        logging.error(f"Unknown predictor type: {predictor_type}")
        return GARCHPredictor()  # Default to GARCH

def run_eacto_system(args):
    """Run the complete EACTO system workflow."""
    # Load or fetch data
    data = load_or_fetch_data(args)
    if data is None:
        return
    
    # Create entropy calculator and calculate entropy
    entropy_calculator = EntropyCalculator()
    entropy_config = {
        'calculate_shannon': True,
        'shannon_window': 60,
        'shannon_bins': 10
    }
    data_with_entropy = entropy_calculator.calculate_all_entropies(
        data, 'log_return', entropy_config
    )
    
    # Create system components
    system_predictor = create_predictor(args)
    
    # Create MPC controller
    mpc_config = {
        'prediction_horizon_n': 10,
        'control_horizon_m': 5,
        'weight_error_lambda': 1.0,
        'weight_control_gamma': 0.1,
        'target_breach_prob_p_star': args.target_breach_prob,
        'alpha_min': 0.001,
        'alpha_max_base': 0.1,
        'entropy_scaling_enabled': True,
        'base_alpha': 0.02
    }
    mpc_controller = MPCController(system_predictor, mpc_config)
    
    # Create threshold manager
    threshold_manager = ThresholdManager()
    
    # Create backtest engine
    backtest_config = {
        'returns_col': 'log_return',
        'entropy_col': 'shannon_entropy',
        'test_start': args.test_start,
        'warmup_periods': args.warmup_periods,
        'refit_frequency': 20,
        'verbose': args.log_level == 'DEBUG'
    }
    
    backtest_engine = BacktestEngine(
        data_with_entropy, entropy_calculator, system_predictor,
        mpc_controller, threshold_manager, backtest_config
    )
    
    # Run backtest
    logging.info("Running backtest")
    results = backtest_engine.run_backtest()
    
    if results is None:
        logging.error("Backtest failed")
        return
    
    # Calculate summary statistics
    summary_stats = backtest_engine.calculate_summary_statistics()
    
    # Evaluate results
    evaluation = evaluate_backtest_results(results, args.target_breach_prob)
    
    # Create output directory
    output_dir = create_output_directory(
        base_dir=args.output_dir,
        experiment_name=f"{args.ticker}_{args.predictor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Save results
    results.to_csv(os.path.join(output_dir, 'backtest_results.csv'))
    save_results(summary_stats, os.path.join(output_dir, 'summary_stats.json'))
    save_results(evaluation, os.path.join(output_dir, 'evaluation.json'))
    
    # Create visualization dashboard
    logging.info("Generating visualization dashboard")
    create_dashboard(results, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print(f"EACTO BACKTEST RESULTS: {args.ticker} with {args.predictor.upper()} predictor")
    print("="*80)
    print(format_summary_stats(summary_stats))
    print("\n" + "="*80)
    print(f"Detailed results saved to: {output_dir}")
    print("="*80)

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=os.path.join(args.output_dir, 'eacto.log')
    )
    
    try:
        run_eacto_system(args)
    except Exception as e:
        logging.exception(f"Error running EACTO system: {e}")

if __name__ == "__main__":
    main() 