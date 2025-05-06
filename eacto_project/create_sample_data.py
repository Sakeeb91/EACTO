#!/usr/bin/env python
"""
Create synthetic market data for EACTO testing.
This script generates a synthetic price series with varying volatility regimes.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_synthetic_market_data(start_date='2022-01-01', end_date='2023-12-31', 
                                 seed=42, output_file='data/processed/synthetic_data.csv'):
    """
    Generate synthetic market data with different volatility regimes to test EACTO.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        seed (int): Random seed for reproducibility
        output_file (str): Path to save the generated data
        
    Returns:
        pandas.DataFrame: Generated data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    num_days = (end - start).days + 1
    
    # Create trading days (exclude weekends)
    dates = []
    for i in range(num_days):
        date = start + timedelta(days=i)
        # Only include weekdays (0 = Monday, 6 = Sunday)
        if date.weekday() < 5:
            dates.append(date)
    
    # Number of trading days
    n = len(dates)
    
    # Generate price data with different volatility regimes
    # Initial price
    price = 400.0  # Starting price (similar to SPY in recent years)
    prices = [price]
    
    # Create returns with different volatility regimes
    returns = []
    volatilities = []
    
    # Define volatility regimes
    regimes = [
        {'duration': int(n * 0.2), 'volatility': 0.005, 'drift': 0.0005},  # Low volatility, positive drift
        {'duration': int(n * 0.15), 'volatility': 0.015, 'drift': -0.001},  # High volatility, negative drift (market stress)
        {'duration': int(n * 0.2), 'volatility': 0.008, 'drift': 0.0003},  # Medium volatility, positive drift
        {'duration': int(n * 0.15), 'volatility': 0.02, 'drift': -0.0015},  # Very high volatility, negative drift (crisis)
        {'duration': int(n * 0.3), 'volatility': 0.01, 'drift': 0.0008}    # Medium volatility, positive drift (recovery)
    ]
    
    # Generate returns based on volatility regimes
    current_index = 0
    for regime in regimes:
        duration = regime['duration']
        vol = regime['volatility']
        drift = regime['drift']
        
        for _ in range(duration):
            if current_index < n:
                # Generate return as drift + volatility * random shock
                ret = drift + vol * np.random.normal()
                returns.append(ret)
                volatilities.append(vol)
                
                # Update price
                price = price * (1 + ret)
                prices.append(price)
                
                current_index += 1
    
    # Ensure we have the correct number of data points
    returns = returns[:n]
    prices = prices[:n]
    volatilities = volatilities[:n]
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(r) * 0.5) for p, r in zip(prices, returns)],
        'low': [p * (1 - abs(r) * 0.5) for p, r in zip(prices, returns)],
        'close': [p * (1 + r) for p, r in zip(prices, returns)],
        'volume': [int(1e6 * (1 + abs(r) * 10)) for r in returns],
        'true_volatility': volatilities
    }, index=dates)
    
    # Calculate log returns and realized volatility
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate realized volatility (20-day rolling standard deviation of log returns)
    df['realized_vol'] = df['log_return'].rolling(window=20).std()
    
    # Create simulated entropy measures
    # Higher entropy during high volatility periods with some noise
    df['shannon_entropy'] = df['realized_vol'] * 10 + np.random.normal(0, 0.1, len(df))
    # Ensure entropy is positive and scaled appropriately (between 0 and 2)
    df['shannon_entropy'] = (df['shannon_entropy'] - df['shannon_entropy'].min()) / \
                           (df['shannon_entropy'].max() - df['shannon_entropy'].min()) * 2
    
    # Drop NaN values
    df = df.dropna()
    
    # Save to CSV
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file)
        print(f"Synthetic data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Create synthetic data
    data = create_synthetic_market_data()
    print(f"Created {len(data)} days of synthetic market data")
    print(data.head()) 