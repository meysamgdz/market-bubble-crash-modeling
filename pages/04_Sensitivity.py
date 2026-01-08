"""Sensitivity Analysis Page"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket
from analysis.analysis import analyze_log_periodicity

st.set_page_config(layout="wide", page_title="Sensitivity")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("🔬 Analysis Settings")
    
    analysis_type = st.selectbox(
        "Parameter to Vary",
        ["Lambda (λ)", "Feedback Strength", "Hierarchy Depth"],
        help="Choose which parameter to test - others held constant"
    )
    
    st.subheader("Settings")
    col1, col2 = st.columns(2)
    
    num_replicates = col1.slider("Replicates", 1, 10, 3,
                                 help="Number of runs per parameter value - more = better statistics")
    sa_steps = col2.slider("Steps", 500, 2000, 1000, 100,
                          help="Simulation length for each run")
    
    if analysis_type == "Lambda (λ)":
        lambda_min = col1.slider("Min λ", 1.5, 2.0, 1.5, 0.1,
                                help="Minimum lambda to test")
        lambda_max = col2.slider("Max λ", 2.0, 3.0, 3.0, 0.1,
                                help="Maximum lambda to test")
        n_points = col1.slider("Points", 3, 10, 5,
                              help="Number of values to test between min and max")
        param_values = np.linspace(lambda_min, lambda_max, n_points)
        param_name = "Lambda"
    
    elif analysis_type == "Feedback Strength":
        fb_min = col1.slider("Min α", 0.01, 0.10, 0.03, 0.01,
                            help="Minimum feedback strength")
        fb_max = col2.slider("Max α", 0.10, 0.30, 0.15, 0.01,
                            help="Maximum feedback strength")
        n_points = col1.slider("Points", 3, 10, 5,
                              help="Number of values to test")
        param_values = np.linspace(fb_min, fb_max, n_points)
        param_name = "Feedback"
    
    else:  # Hierarchy Depth
        level_min = col1.slider("Min Levels", 2, 3, 2,
                               help="Minimum hierarchy depth")
        level_max = col2.slider("Max Levels", 3, 6, 5,
                               help="Maximum hierarchy depth")
        param_values = np.arange(level_min, level_max + 1)
        param_name = "Depth"
    
    run_analysis = st.button("▶️ Run Analysis", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔬 Sensitivity Analysis")

if not run_analysis:
    st.info("👈 Configure settings and click **Run Analysis**")
    
    st.markdown(f"""
    ### {analysis_type} Sensitivity
    
    This analysis varies **{param_name.lower()}** while keeping other parameters fixed.
    
    **What we measure:**
    - Detection rate (% of runs with LPPL pattern)
    - Mean R² (fit quality)
    - Observed λ (scaling ratio)
    - Crash magnitude
    
    **Settings:**
    - Will test {len(param_values)} parameter values
    - {num_replicates} replicate(s) per value
    - {sa_steps} simulation steps each
    - Total runs: {len(param_values) * num_replicates}
    
    **Expected time:** ~{len(param_values) * num_replicates * 3} seconds
    """)

else:
    st.subheader("📈 Running Analysis...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_runs = len(param_values) * num_replicates
    current_run = 0
    
    for param_val in param_values:
        replicate_results = []
        
        for rep in range(num_replicates):
            status_text.text(f"Testing {param_name}={param_val:.2f}, replicate {rep+1}/{num_replicates}")
            
            # Create market
            if analysis_type == "Lambda (λ)":
                market = HierarchicalMarket(
                    n_levels=4,
                    branching_factor=2,
                    lambda_ratio=param_val,
                    feedback_strength=0.08
                )
            elif analysis_type == "Feedback Strength":
                market = HierarchicalMarket(
                    n_levels=4,
                    branching_factor=2,
                    lambda_ratio=2.0,
                    feedback_strength=param_val
                )
            else:  # Depth
                market = HierarchicalMarket(
                    n_levels=int(param_val),
                    branching_factor=2,
                    lambda_ratio=2.0,
                    feedback_strength=0.08
                )
            
            # Run
            prices, _ = market.create_bubble_scenario(
                n_steps=sa_steps,
                bubble_start=int(sa_steps*0.2),
                crash_time=int(sa_steps*0.8)
            )
            
            # Analyze
            time = np.arange(int(sa_steps*0.8))
            crash_idx = int(sa_steps*0.8)
            
            try:
                lppl_results = analyze_log_periodicity(
                    time, 
                    prices[:crash_idx],
                    verbose=False
                )
                
                replicate_results.append({
                    'detected': lppl_results['lppl_detected'],
                    'confidence': lppl_results['lppl_confidence'],
                    'lambda': lppl_results['lppl_lambda'],
                    'crash': (prices[crash_idx] - prices[-1]) / prices[crash_idx]
                })
            except:
                replicate_results.append({
                    'detected': False,
                    'confidence': 0.0,
                    'lambda': 0.0,
                    'crash': 0.0
                })
            
            current_run += 1
            progress_bar.progress(current_run / total_runs)
        
        # Aggregate
        results.append({
            'param': param_val,
            'detection_rate': np.mean([r['detected'] for r in replicate_results]),
            'mean_confidence': np.mean([r['confidence'] for r in replicate_results]),
            'mean_lambda': np.mean([r['lambda'] for r in replicate_results if r['lambda'] > 0]),
            'mean_crash': np.mean([r['crash'] for r in replicate_results]),
            'std_crash': np.std([r['crash'] for r in replicate_results])
        })
    
    status_text.text("✅ Analysis complete!")
    progress_bar.empty()
    
    # Plot results
    st.subheader("📊 Results")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    params = [r['param'] for r in results]
    
    # Detection rate
    axes[0, 0].plot(params, [r['detection_rate'] for r in results], 
                  'o-', linewidth=2, markersize=8, color='#2ecc71')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel("Detection Rate")
    axes[0, 0].set_title("LPPL Pattern Detection Rate\n% of runs with R²>0.85")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(alpha=0.3)
    
    # Confidence
    axes[0, 1].plot(params, [r['mean_confidence'] for r in results],
                  'o-', linewidth=2, markersize=8, color='#3498db')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel("Mean R²")
    axes[0, 1].set_title("Mean Fit Quality (R²)\nHigher = better fit")
    axes[0, 1].grid(alpha=0.3)
    
    # Lambda
    axes[1, 0].plot(params, [r['mean_lambda'] for r in results],
                  'o-', linewidth=2, markersize=8, color='#e74c3c')
    if analysis_type == "Lambda (λ)":
        axes[1, 0].plot(params, params, '--', color='gray', label='Expected', linewidth=2)
        axes[1, 0].legend()
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel("Observed λ")
    axes[1, 0].set_title("Observed Scaling Ratio λ\nShould match input (if testing λ)")
    axes[1, 0].grid(alpha=0.3)
    
    # Crash magnitude
    crashes = [r['mean_crash'] for r in results]
    errors = [r['std_crash'] for r in results]
    axes[1, 1].errorbar(params, crashes, yerr=errors,
                      fmt='o-', linewidth=2, markersize=8, 
                      capsize=5, color='#9b59b6')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel("Crash Magnitude")
    axes[1, 1].set_title("Mean Crash Size\n(Price drop as fraction)")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Summary table
    st.subheader("📊 Summary Table")
    df = pd.DataFrame(results)
    df.columns = [param_name, 'Detection Rate', 'Mean R²', 'Mean λ', 
                 'Mean Crash', 'Std Crash']
    st.dataframe(df.style.format({
        'Detection Rate': '{:.2%}',
        'Mean R²': '{:.3f}',
        'Mean λ': '{:.2f}',
        'Mean Crash': '{:.2%}',
        'Std Crash': '{:.3f}'
    }), use_container_width=True)
    
    # Key findings
    st.subheader("💡 Key Findings")
    
    best_idx = np.argmax([r['detection_rate'] for r in results])
    worst_idx = np.argmin([r['detection_rate'] for r in results])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best Detection:**")
        st.markdown(f"- {param_name} = {results[best_idx]['param']:.2f}")
        st.markdown(f"- Detection rate: {results[best_idx]['detection_rate']:.1%}")
        st.markdown(f"- Mean R²: {results[best_idx]['mean_confidence']:.3f}")
    
    with col2:
        st.markdown("**Worst Detection:**")
        st.markdown(f"- {param_name} = {results[worst_idx]['param']:.2f}")
        st.markdown(f"- Detection rate: {results[worst_idx]['detection_rate']:.1%}")
        st.markdown(f"- Mean R²: {results[worst_idx]['mean_confidence']:.3f}")


st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if st.button("⬅️ Previous: LPPL Analysis", use_container_width=True):
        st.switch_page("pages/3_LPPL_Analysis.py")