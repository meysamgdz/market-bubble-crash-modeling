"""LPPL Analysis Page"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket
from analysis.analysis import analyze_log_periodicity, plot_lppl_fit

st.set_page_config(layout="wide", page_title="LPPL Analysis")

# ============================================================================
# SESSION STATE
# ============================================================================

if 'sim_prices' not in st.session_state:
    st.session_state.sim_prices = None
    st.session_state.sim_complete = False
    st.session_state.analyze_done = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Analysis Controls")
    
    st.subheader("Hierarchy")
    col1, col2 = st.columns(2)
    n_levels = col1.slider("Levels", 2, 5, 4, 1,
                          help="Number of hierarchy levels")
    branching_factor = col2.slider("Branching", 2, 3, 2, 1,
                                   help="Children per parent node")
    n_top_agents = col1.slider("Top Agents", 1, 5, 1, 1,
                               help="Number of agents at highest level")
    lambda_ratio = col2.slider("λ (Lambda)", 1.5, 3.0, 2.0, 0.1,
                               help="Time scale ratio - expect patterns with λ≈2")
    
    st.subheader("Market Dynamics")
    col1, col2 = st.columns(2)
    feedback = col1.slider("Feedback (α)", 0.01, 0.30, 0.10, 0.01,
                          help="Positive feedback strength - higher creates stronger bubbles")
    volatility = col2.slider("Volatility (σ)", 0.005, 0.05, 0.01, 0.005,
                            help="Random noise level")
    
    st.subheader("Agent Parameters")
    col1, col2 = st.columns(2)
    J_horizontal = col1.slider("J₀ (Peer)", 0.5, 2.0, 1.0, 0.1,
                              help="Horizontal coupling strength")
    J_vertical = col2.slider("J₁ (Authority)", 1.0, 5.0, 2.0, 0.5,
                            help="Vertical coupling strength")
    temperature = col1.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                             help="Noise level in agent decisions")
    
    st.subheader("Scenario Settings")
    scenario_type = st.selectbox(
        "Scenario Type",
        ["Bubble & Crash", "Custom Dynamics"],
        help="Bubble & Crash uses preset dynamics, Custom lets you control timing"
    )
    
    n_steps = st.slider("Total Steps", 500, 2000, 1000, 100,
                       help="Total simulation length")
    
    if scenario_type == "Bubble & Crash":
        col1, col2 = st.columns(2)
        bubble_start = col1.slider("Bubble Start", 100, 500, 200, 50,
                                   help="When bubble formation begins")
        crash_time = col2.slider("Crash Time", 500, 1500, 800, 50,
                                 help="When crash occurs")
    
    if st.button("🎲 Generate Bubble", use_container_width=True):
        with st.spinner("Generating bubble scenario..."):
            market = HierarchicalMarket(
                n_levels=n_levels,
                branching_factor=branching_factor,
                lambda_ratio=lambda_ratio,
                n_top_agents=n_top_agents,
                initial_price=100.0,
                feedback_strength=feedback,
                drift=0.0002,
                volatility=volatility
            )
            
            # Set agent parameters
            market.J_horizontal = J_horizontal
            market.J_vertical = J_vertical
            market.temperature = temperature
            
            # Run scenario
            if scenario_type == "Bubble & Crash":
                prices, _ = market.create_bubble_scenario(
                    n_steps=n_steps,
                    bubble_start=bubble_start,
                    crash_time=crash_time
                )
            else:
                # Custom dynamics - just run normal simulation
                prices, _ = market.simulate(n_steps=n_steps, show_progress=False)
            
            st.session_state.sim_prices = prices
            st.session_state.sim_complete = True
            st.session_state.analyze_done = False
            st.session_state.sim_params = {
                'n_levels': n_levels,
                'branching_factor': branching_factor,
                'n_top_agents': n_top_agents,
                'lambda_ratio': lambda_ratio,
                'feedback': feedback,
                'volatility': volatility,
                'J_horizontal': J_horizontal,
                'J_vertical': J_vertical,
                'temperature': temperature,
                'scenario_type': scenario_type
            }
            st.rerun()
    
    if st.session_state.sim_complete:
        st.success("✅ Data ready")
        
        st.subheader("Analysis Window")
        window_end = st.slider(
            "Analyze up to",
            100, 
            len(st.session_state.sim_prices),
            int(len(st.session_state.sim_prices)*0.8),
            help="Select data range for LPPL fitting - should end BEFORE crash"
        )
        
        if st.button("🔍 Analyze Pattern", use_container_width=True):
            with st.spinner("Fitting LPPL model (30-60 seconds)..."):
                time = np.arange(window_end)
                prices = st.session_state.sim_prices[:window_end]
                
                try:
                    results = analyze_log_periodicity(time, prices, verbose=False)
                    st.session_state.lppl_results = results
                    st.session_state.window_end = window_end
                    st.session_state.analyze_done = True
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.analyze_done = False
            
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 LPPL Pattern Analysis")

if not st.session_state.sim_complete:
    st.info("👈 Configure parameters and click **Generate Bubble** to begin")
    
    st.markdown("""
    ### About LPPL Analysis
    
    This tool detects **Log-Periodic Power Law** patterns that signal approaching crashes.
    
    **Process:**
    1. **Generate** bubble scenario with desired parameters
    2. **Select** analysis window (should end before crash)
    3. **Analyze** to fit LPPL model and detect patterns
    4. **Review** confidence metrics and predictions
    
    **For good detection:**
    - Strong bubble dynamics (feedback α ≥ 0.10)
    - Sufficient data (≥ 500 points)
    - Analysis window captures bubble but not crash
    - Expected λ ≈ 2.0 for hierarchical model
    """)

else:
    # Show raw data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Price Data")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(st.session_state.sim_prices, 'b-', linewidth=2)
        if st.session_state.analyze_done:
            ax.axvline(st.session_state.window_end, color='orange', 
                      linestyle='--', linewidth=2, label='Analysis Window End')
            ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.set_title("Full Price Series")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Data Info")
        st.metric("Total Points", len(st.session_state.sim_prices))
        st.metric("Initial Price", f"${st.session_state.sim_prices[0]:.2f}")
        st.metric("Max Price", f"${st.session_state.sim_prices.max():.2f}")
        change = (st.session_state.sim_prices.max() / st.session_state.sim_prices[0] - 1) * 100
        st.metric("Max Gain", f"{change:.1f}%")
        
        # Show simulation parameters if available
        if hasattr(st.session_state, 'sim_params'):
            with st.expander("📋 Simulation Parameters"):
                params = st.session_state.sim_params
                st.markdown(f"""
                **Hierarchy:**
                - Levels: {params['n_levels']}
                - Branching: {params['branching_factor']}
                - Top Agents: {params['n_top_agents']}
                - λ: {params['lambda_ratio']:.1f}
                
                **Market:**
                - Feedback (α): {params['feedback']:.3f}
                - Volatility (σ): {params['volatility']:.3f}
                
                **Agents:**
                - J₀ (Peer): {params['J_horizontal']:.1f}
                - J₁ (Authority): {params['J_vertical']:.1f}
                - Temperature: {params['temperature']:.1f}
                
                **Scenario:** {params['scenario_type']}
                """)
    
    # Analysis results
    if st.session_state.analyze_done:
        st.markdown("---")
        st.subheader("🔍 LPPL Analysis Results")
        
        results = st.session_state.lppl_results
        
        # Metrics with help text
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            detected = "✅ YES" if results['lppl_detected'] else "❌ NO"
            st.metric("Pattern Detected", detected,
                     help="YES if R²>0.85 and parameters in valid ranges")
        
        with col2:
            st.metric("Confidence (R²)", f"{results['lppl_confidence']:.3f}",
                     help="R-squared fit quality: >0.90 excellent, >0.85 good, <0.70 poor")
        
        with col3:
            if results['lppl_detected']:
                st.metric("Predicted tc", f"{results['predicted_crash_time']:.0f}",
                         help="Predicted crash time - should be after analysis window")
            else:
                st.metric("Predicted tc", "N/A",
                         help="No pattern detected - tc not reliable")
        
        with col4:
            expected_lambda = st.session_state.sim_params['lambda_ratio'] if hasattr(st.session_state, 'sim_params') else 2.0
            st.metric("Fitted λ", f"{results['lppl_lambda']:.2f}",
                     help=f"Scaling ratio from LPPL fit - should match input λ={expected_lambda:.1f}")
        
        # Detailed results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 LPPL Fit")
            
            if results['lppl_detected']:
                fig, ax = plt.subplots(figsize=(8, 6))
                time = np.arange(st.session_state.window_end)
                prices = st.session_state.sim_prices[:st.session_state.window_end]
                plot_lppl_fit(time, prices, results['lppl_params'], ax=ax)
                
                # Add predicted crash line
                tc = results['predicted_crash_time']
                if tc > time[-1] and tc < time[-1] * 1.5:
                    ax.axvline(tc, color='red', linestyle='--', linewidth=2,
                             label=f'Predicted Crash (tc={tc:.0f})')
                    ax.legend()
                
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("⚠️ Pattern not detected - fit not shown")
                st.markdown("""
                **Possible reasons:**
                - Insufficient bubble strength (try higher feedback)
                - Too much noise
                - Window too short or too long
                - Pattern hasn't fully developed
                """)
        
        with col2:
            st.subheader("📋 LPPL Parameters")
            
            if results['lppl_detected']:
                params = results['lppl_params']
                
                st.markdown("**Power Law Component:**")
                m_check = "✓" if 0.2 < params['m'] < 0.5 else "✗"
                st.markdown(f"- **m** = {params['m']:.3f} {m_check}")
                st.caption("Controls acceleration rate (typical: 0.2-0.5)")
                
                st.markdown("\n**Log-Periodic Component:**")
                omega_check = "✓" if 5 < params['omega'] < 15 else "✗"
                st.markdown(f"- **ω** = {params['omega']:.2f} {omega_check}")
                st.caption("Angular frequency (typical: 5-15)")
                
                st.markdown(f"- **C** = {params['C']:.3f}")
                st.caption("Oscillation amplitude (range: -1 to +1)")
                
                expected_lambda = st.session_state.sim_params['lambda_ratio'] if hasattr(st.session_state, 'sim_params') else 2.0
                lambda_check = "✓" if abs(params['lambda'] - expected_lambda) < 0.5 else "⚠"
                st.markdown(f"- **λ** = {params['lambda']:.2f} {lambda_check}")
                st.caption(f"Should match input λ={expected_lambda:.1f}")
                
                st.markdown("\n**Critical Point:**")
                tc_check = "✓" if params['tc'] > time[-1] else "✗"
                st.markdown(f"- **tc** = {params['tc']:.1f} {tc_check}")
                st.caption(f"Predicted crash time (data ends at {time[-1]})")
                
                st.markdown("\n**Time to Crash:**")
                ttc = results['time_to_crash']
                st.markdown(f"- **Δt** = {ttc:.0f} time units")
                
                st.markdown("---")
                
                # Overall quality
                expected_lambda = st.session_state.sim_params['lambda_ratio'] if hasattr(st.session_state, 'sim_params') else 2.0
                all_checks = [
                    0.2 < params['m'] < 0.5,
                    5 < params['omega'] < 15,
                    abs(params['lambda'] - expected_lambda) < 0.5,
                    params['tc'] > time[-1],
                    results['lppl_confidence'] > 0.85
                ]
                
                quality = sum(all_checks)
                if quality >= 4:
                    st.success(f"✅ High Quality Fit ({quality}/5 checks passed)")
                elif quality >= 3:
                    st.warning(f"⚠️ Moderate Quality ({quality}/5 checks passed)")
                else:
                    st.error(f"❌ Low Quality ({quality}/5 checks passed)")
            
            else:
                st.info("Parameters not available - pattern not detected")
                st.markdown("""
                **To improve detection:**
                - Increase feedback strength (α > 0.10)
                - Ensure sufficient data points (> 500)
                - Check that bubble formed before crash
                - Try different analysis window
                """)
        
        # DSI Analysis
        st.markdown("---")
        st.subheader("📏 Discrete Scale Invariance (DSI)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(results['observed_ratios']) > 0:
                st.markdown("**Observed Ratios:**")
                st.caption("Ratios of consecutive time intervals between peaks")
                
                for i, ratio in enumerate(results['observed_ratios']):
                    st.markdown(f"- **Ratio {i+1}:** {ratio:.3f}")
                
                expected_lambda = st.session_state.sim_params['lambda_ratio'] if hasattr(st.session_state, 'sim_params') else 2.0
                
                st.markdown(f"\n**Statistics:**")
                st.markdown(f"- Mean: {results['mean_ratio']:.3f}")
                st.markdown(f"- Std Dev: {results['ratio_std']:.3f}")
                st.markdown(f"- Expected λ: {expected_lambda:.2f}")
                
                if results['dsi_consistent']:
                    st.success("✅ Ratios consistent - DSI detected!")
                    st.caption("Intervals decrease geometrically")
                else:
                    st.warning("⚠️ Ratios variable - weak DSI")
                    st.caption("Pattern may be irregular")
            else:
                st.info("Insufficient peaks for DSI analysis")
                st.caption("Need at least 4 peaks in price series")
        
        with col2:
            if len(results['observed_ratios']) > 0:
                st.markdown("**Visual Check:**")
                fig, ax = plt.subplots(figsize=(6, 4))
                
                expected_lambda = st.session_state.sim_params['lambda_ratio'] if hasattr(st.session_state, 'sim_params') else 2.0
                
                ratio_indices = range(1, len(results['observed_ratios']) + 1)
                ax.bar(ratio_indices, results['observed_ratios'], alpha=0.7, color='steelblue')
                ax.axhline(expected_lambda, color='red', linestyle='--', linewidth=2,
                         label=f'Expected λ={expected_lambda:.1f}')
                ax.axhline(results['mean_ratio'], color='green', linestyle='-', linewidth=1.5,
                         label=f'Mean={results["mean_ratio"]:.2f}')
                
                ax.set_xlabel("Ratio Index")
                ax.set_ylabel("Scaling Ratio")
                ax.set_title("Observed vs Expected Scaling Ratios")
                ax.legend()
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            else:
                st.markdown("**What is DSI?**")
                st.caption("""
                Discrete Scale Invariance means patterns repeat at specific scales.
                
                Time intervals between peaks should follow:
                Δt₁/Δt₂ ≈ Δt₂/Δt₃ ≈ λ
                
                This creates the characteristic log-periodic wobbles!
                """)

# ============================================================================
# NAVIGATION BUTTONS
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if st.button("⬅️ Previous: Simulation", use_container_width=True):
        st.switch_page("pages/2_Simulation.py")

with col3:
    if st.button("Next: Sensitivity ➡️", use_container_width=True):
        st.switch_page("pages/4_Sensitivity.py")