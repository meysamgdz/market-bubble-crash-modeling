"""
Complete example: Simulating a market bubble and crash

This example demonstrates:
1. Building hierarchical agent network
2. Running market simulation with bubble scenario
3. Detecting log-periodic patterns
4. Visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add project root to path (works on all platforms)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket
from analysis.analysis import analyze_log_periodicity, plot_lppl_fit


def main():
    """Run complete bubble and crash simulation."""
    
    print("="*60)
    print("HIERARCHICAL MARKET MODEL: BUBBLE & CRASH SIMULATION")
    print("="*60)
    print()
    
    # Create market with hierarchical structure
    print("Step 1: Creating hierarchical market...")
    print("-" * 60)
    
    market = HierarchicalMarket(
        n_levels=4,              # 4 levels of hierarchy
        branching_factor=2,      # Each parent influences 2 children
        lambda_ratio=2.0,        # Time scales double each level
        initial_price=100.0,
        feedback_strength=0.05,  # Moderate positive feedback
        drift=0.0002,           # Small upward drift
        volatility=0.01         # 1% volatility
    )
    
    market.network.print_hierarchy_info()
    print()
    
    # Run bubble scenario
    print("\nStep 2: Simulating bubble formation and crash...")
    print("-" * 60)
    
    n_steps = 2000
    bubble_start = 1100
    crash_time = 1700
    
    prices, magnetizations = market.create_bubble_scenario(
        n_steps=n_steps,
        bubble_start=bubble_start,
        crash_time=crash_time
    )
    
    market.print_simulation_summary()
    print()
    
    # Analyze log-periodicity
    print("\nStep 3: Analyzing log-periodic patterns...")
    print("-" * 60)
    
    # Analyze pre-crash data (up to crash point)
    time = np.arange(len(prices))
    analysis_window = time <= crash_time
    
    results = analyze_log_periodicity(
        time[analysis_window],
        prices[analysis_window],
        verbose=True
    )
    print()
    
    # Visualize results
    print("Step 4: Creating visualizations...")
    print("-" * 60)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Price evolution with phases
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, prices, 'b-', linewidth=2)
    ax1.axvline(bubble_start, color='orange', linestyle='--', 
               label='Bubble Start', alpha=0.7)
    ax1.axvline(crash_time, color='red', linestyle='--',
               label='Crash', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Evolution: Bubble Formation and Crash', 
                 fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log price (shows super-exponential growth)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(time, np.log(prices), 'g-', linewidth=2)
    ax2.axvline(bubble_start, color='orange', linestyle='--', alpha=0.7)
    ax2.axvline(crash_time, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Log Price')
    ax2.set_title('Log Price (Super-exponential Growth)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnetization (market sentiment)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time, magnetizations, 'purple', linewidth=2)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax3.axvline(bubble_start, color='orange', linestyle='--', alpha=0.7)
    ax3.axvline(crash_time, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Magnetization')
    ax3.set_title('Market Sentiment (Herding)', fontweight='bold')
    ax3.set_ylim([-1.1, 1.1])
    ax3.grid(True, alpha=0.3)
    
    # 4. Magnetization by level
    ax4 = plt.subplot(3, 2, 4)
    mag_by_level = market.get_magnetization_by_level()
    colors = ['blue', 'green', 'orange', 'red']
    for level in sorted(mag_by_level.keys()):
        ax4.plot(time, mag_by_level[level], 
                label=f'Level {level} (τ={market.network._compute_time_scale(level):.1f})',
                color=colors[level % len(colors)],
                linewidth=2,
                alpha=0.7)
    ax4.axvline(crash_time, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Magnetization')
    ax4.set_title('Hierarchical Cascading (Multiple Time Scales)', 
                 fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. LPPL fit (pre-crash data)
    ax5 = plt.subplot(3, 2, 5)
    if results['lppl_detected']:
        plot_lppl_fit(
            time[analysis_window],
            prices[analysis_window],
            results['lppl_params'],
            ax=ax5
        )
    else:
        ax5.text(0.5, 0.5, 'LPPL pattern not detected',
                ha='center', va='center', fontsize=14)
        ax5.set_title('LPPL Fit', fontweight='bold')
    
    # 6. Returns distribution
    ax6 = plt.subplot(3, 2, 6)
    ts = market.get_time_series()
    returns = ts['returns']
    
    # Separate pre-crash and crash returns
    pre_crash_returns = returns[:crash_time]
    crash_returns = returns[crash_time:]
    
    ax6.hist(pre_crash_returns, bins=50, alpha=0.6, 
            label='Pre-crash', color='blue', density=True)
    ax6.hist(crash_returns, bins=30, alpha=0.6,
            label='Crash period', color='red', density=True)
    ax6.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Returns')
    ax6.set_ylabel('Density')
    ax6.set_title('Returns Distribution (Fat Tails)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Visualization saved as 'bubble_crash_simulation.png'")
    print()
    
    # Key insights
    print("="*60)
    print("KEY INSIGHTS FROM SIMULATION")
    print("="*60)
    
    print("\n1. HIERARCHICAL STRUCTURE:")
    print(f"   - Created {market.network.n_levels} levels of agents")
    print(f"   - Time scales: {[f'{market.network._compute_time_scale(l):.1f}' for l in range(market.network.n_levels)]}")
    print(f"   - Scaling ratio λ = {market.network.lambda_ratio}")
    
    print("\n2. BUBBLE FORMATION:")
    print(f"   - Normal phase: steps 0-{bubble_start}")
    print(f"   - Bubble phase: steps {bubble_start}-{crash_time}")
    print(f"   - Price increase: {(prices[crash_time]/prices[0]-1)*100:.1f}%")
    print(f"   - Mechanism: Increasing positive feedback → herding")
    
    print("\n3. LOG-PERIODIC PATTERN:")
    if results['lppl_detected']:
        print(f"   - Pattern detected: YES (confidence {results['lppl_confidence']:.2%})")
        print(f"   - Power law exponent m: {results['lppl_params']['m']:.3f}")
        print(f"   - Frequency ω: {results['lppl_params']['omega']:.2f}")
        print(f"   - Scaling ratio λ: {results['lppl_params']['lambda']:.2f}")
        print(f"   - Predicted crash: t = {results['predicted_crash_time']:.0f}")
        print(f"   - Actual crash: t = {crash_time}")
    else:
        print(f"   - Pattern detected: NO (confidence {results['lppl_confidence']:.2%})")
        print(f"   - May need more pronounced bubble dynamics")
    
    print("\n4. CRASH DYNAMICS:")
    crash_magnitude = (prices[crash_time] - prices[-1]) / prices[crash_time]
    print(f"   - Crash magnitude: {crash_magnitude*100:.1f}%")
    print(f"   - Mechanism: Noise increase breaks herding consensus")
    print(f"   - Final magnetization: {magnetizations[-1]:.3f}")
    
    print("\n5. EMERGENT PROPERTIES:")
    print(f"   - Discrete scale invariance: λ ≈ {market.network.lambda_ratio}")
    print(f"   - Super-exponential growth during bubble")
    print(f"   - Fat-tailed returns (visible in histogram)")
    print(f"   - Hierarchical cascade from top to bottom")
    
    print("\n" + "="*60)
    print("Simulation complete! Check the visualization for details.")
    print("="*60)
    
    return market, results, fig


if __name__ == "__main__":
    market, results, fig = main()
    
    # Keep plot open if running interactively
    # plt.show()
