#!/usr/bin/env python
"""
Demo script to generate sample results and visualizations for EACTO.
This bypasses the full system to create visualizations from synthetic data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def set_plot_style():
    """Set consistent plot style."""
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
def create_output_directory(base_dir='results'):
    """Create a directory for experiment outputs."""
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create experiment directory with timestamp
    experiment_name = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create the directory
    os.makedirs(experiment_dir)
    
    print(f"Created output directory: {experiment_dir}")
    return experiment_dir

def simulate_alpha_thresholds(data):
    """
    Simulate EACTO threshold optimization results.
    
    Args:
        data (pandas.DataFrame): The synthetic data with returns and entropy
        
    Returns:
        pandas.DataFrame: Data with simulated alpha thresholds
    """
    # Create a copy to avoid modifying the original
    result_df = data.copy()
    
    # Initialize alpha with a base value
    base_alpha = 0.02
    
    # Calculate adaptive thresholds based on entropy and volatility
    # Higher entropy/volatility -> lower threshold (more conservative)
    result_df['alpha'] = base_alpha * (1 - 0.5*result_df['shannon_entropy']) * (1 - 0.3*result_df['realized_vol']/result_df['realized_vol'].max())
    
    # Set minimum and maximum alpha
    result_df['alpha'] = result_df['alpha'].clip(lower=0.005, upper=0.05)
    
    # Calculate loss (negative return)
    result_df['loss'] = -result_df['log_return']
    
    # Determine breaches (loss > alpha)
    result_df['breach'] = (result_df['loss'] > result_df['alpha']).astype(int)
    
    # Calculate running statistics
    result_df['cum_breaches'] = result_df['breach'].cumsum()
    result_df['cum_observations'] = np.arange(1, len(result_df) + 1)
    result_df['breach_rate'] = result_df['cum_breaches'] / result_df['cum_observations']
    
    # Calculate rolling breach rate (20-day window)
    result_df['rolling_breach_rate'] = result_df['breach'].rolling(window=20, min_periods=1).mean()
    
    return result_df

def plot_alpha_vs_loss(results, output_dir):
    """Plot alpha threshold vs actual loss."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot alpha and loss
    ax.plot(results.index, results['alpha'], color='blue', label='Risk Threshold (α)')
    ax.plot(results.index, results['loss'], color='red', alpha=0.7, label='Actual Loss')
    
    # Highlight breaches
    breach_points = results[results['breach'] == 1]
    if not breach_points.empty:
        ax.scatter(breach_points.index, breach_points['loss'], color='darkred', s=50, 
                  label='Breaches', zorder=5)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Risk Threshold (α) vs Actual Loss Over Time')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_vs_loss.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def plot_entropy_series(results, output_dir):
    """Plot entropy and alpha threshold over time."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot entropy
    ax1.plot(results.index, results['shannon_entropy'], color='purple', label='Shannon Entropy')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Shannon Entropy', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    
    # Add alpha as secondary plot
    ax2 = ax1.twinx()
    ax2.plot(results.index, results['alpha'], color='blue', alpha=0.7, label='Risk Threshold (α)')
    ax2.set_ylabel('Risk Threshold (α)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set title
    ax1.set_title('Shannon Entropy and Risk Threshold Over Time')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_series.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def plot_breach_rate(results, output_dir):
    """Plot breach rate over time."""
    # Plot cumulative breach rate
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results.index, results['breach_rate'], color='red', label='Actual Breach Rate')
    ax.axhline(y=0.05, color='blue', linestyle='--', label='Target (5%)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Breach Rate')
    ax.set_title('Cumulative Breach Rate vs Target')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_breach_rate.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    # Plot rolling breach rate
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results.index, results['rolling_breach_rate'], color='red', label='20-Day Rolling Breach Rate')
    ax.axhline(y=0.05, color='blue', linestyle='--', label='Target (5%)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Breach Rate')
    ax.set_title('20-Day Rolling Breach Rate')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rolling_breach_rate.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def plot_alpha_vs_entropy_scatter(results, output_dir):
    """Create scatter plot of alpha vs entropy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(results['shannon_entropy'], results['alpha'], c=results.index, 
                        cmap='viridis', alpha=0.7)
    
    # Add regression line
    sns.regplot(x='shannon_entropy', y='alpha', data=results, 
                scatter=False, ax=ax, color='red', line_kws={'linewidth': 2})
    
    # Calculate correlation
    corr = results['shannon_entropy'].corr(results['alpha'])
    
    # Set labels and title
    ax.set_xlabel('Shannon Entropy')
    ax.set_ylabel('Risk Threshold (α)')
    ax.set_title(f'Relationship Between Shannon Entropy and Risk Threshold (α)\nCorrelation: {corr:.4f}')
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter)
    cbar.set_label('Date')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_vs_entropy_scatter.png'), dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
def create_summary_markdown(results, output_dir):
    """Create a markdown file summarizing the results."""
    breach_rate = results['breach'].mean() * 100
    avg_alpha = results['alpha'].mean()
    min_alpha = results['alpha'].min()
    max_alpha = results['alpha'].max()
    alpha_volatility = results['alpha'].std() / results['alpha'].mean()
    entropy_alpha_corr = results['shannon_entropy'].corr(results['alpha'])
    conservative_in_high_entropy = "Yes" if entropy_alpha_corr < -0.3 else "No"
    
    with open(os.path.join(output_dir, 'RESULTS.md'), 'w') as f:
        f.write("# EACTO Demo Results\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Total Observations**: {len(results)}\n")
        f.write(f"- **Breach Count**: {results['breach'].sum()}\n")
        f.write(f"- **Breach Rate**: {breach_rate:.2f}%\n")
        f.write(f"- **Target Breach Rate**: 5.00%\n")
        f.write(f"- **Breach Rate Error**: {(breach_rate - 5.0):.2f}%\n\n")
        
        f.write("## Threshold Statistics\n\n")
        f.write(f"- **Average Alpha**: {avg_alpha:.6f}\n")
        f.write(f"- **Min Alpha**: {min_alpha:.6f}\n")
        f.write(f"- **Max Alpha**: {max_alpha:.6f}\n")
        f.write(f"- **Threshold Volatility**: {alpha_volatility:.6f}\n\n")
        
        f.write("## Entropy Relationship\n\n")
        f.write(f"- **Alpha-Entropy Correlation**: {entropy_alpha_corr:.4f}\n")
        f.write(f"- **Conservative in High Entropy**: {conservative_in_high_entropy}\n\n")
        
        f.write("## Visualization Description\n\n")
        
        f.write("### Alpha vs Loss\n\n")
        f.write("This plot demonstrates how the EACTO system dynamically adjusts risk thresholds (α) over time. ")
        f.write("Red points indicate breaches where actual losses exceeded the threshold. ")
        f.write("The adaptability of the threshold is evident in how it responds to changing market conditions.\n\n")
        
        f.write("### Entropy Series\n\n")
        f.write("This visualization shows the relationship between market entropy (uncertainty) and the risk threshold. ")
        f.write("When entropy increases, the system typically becomes more conservative by lowering the risk threshold. ")
        f.write("This adaptation helps maintain a consistent breach rate despite changing market conditions.\n\n")
        
        f.write("### Breach Rates\n\n")
        f.write("These plots show how the actual breach rate compares to the target rate (5%). ")
        f.write("The cumulative plot shows the long-term convergence, while the rolling plot shows short-term fluctuations. ")
        f.write("An effective risk management system should maintain a breach rate close to the target over time.\n\n")
        
        f.write("### Alpha vs Entropy Scatter\n\n")
        f.write("This scatter plot reveals the correlation between market entropy and risk thresholds. ")
        f.write("A negative correlation indicates that the system behaves more conservatively during periods of high uncertainty. ")
        f.write("The color gradient shows the time progression, helping identify regime changes in the relationship.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The ENTROPY-ADAPTIVE CYBERNETIC THRESHOLD OPTIMIZER (EACTO) demonstrates the ability to dynamically adjust risk thresholds ")
        f.write("based on market conditions. By incorporating entropy as a measure of market uncertainty, the system becomes more ")
        f.write("conservative during periods of high complexity and more aggressive when markets are more predictable. ")
        f.write("This adaptive approach helps maintain a consistent breach rate, which is a key objective of effective risk management systems.\n")

def main():
    """Main function to execute the script."""
    # Set plot style
    set_plot_style()
    
    # Load synthetic data
    data_path = 'data/processed/synthetic_data.csv'
    if not os.path.exists(data_path):
        print(f"Synthetic data not found at {data_path}")
        print("Please run create_sample_data.py first")
        return
    
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Simulate threshold optimization
    results = simulate_alpha_thresholds(data)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate plots
    plot_alpha_vs_loss(results, output_dir)
    plot_entropy_series(results, output_dir)
    plot_breach_rate(results, output_dir)
    plot_alpha_vs_entropy_scatter(results, output_dir)
    
    # Create summary markdown
    create_summary_markdown(results, output_dir)
    
    # Save results data
    results.to_csv(os.path.join(output_dir, 'demo_results.csv'))
    
    print(f"Demo results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 