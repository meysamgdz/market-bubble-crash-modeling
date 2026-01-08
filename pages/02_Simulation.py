"""Simulation Page"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket

st.set_page_config(layout="wide", page_title="Simulation")

# ============================================================================
# SESSION STATE
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.step = 0
    st.session_state.market = None
    st.session_state.prices = []
    st.session_state.magnetizations = []
    st.session_state.returns = []
    st.session_state.scenario_type = "bubble"

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Simulation Controls")
    
    st.subheader("Hierarchy")
    col1, col2 = st.columns(2)
    n_levels = col1.slider("Levels", 2, 5, 4, 1)
    branching_factor = col2.slider("Branching", 2, 3, 2, 1)
    n_top_agents = col1.slider("Top Agents", 1, 5, 1, 1, 
                               help="Number of agents at highest level")
    lambda_ratio = col2.slider("λ Ratio", 1.5, 3.0, 2.0, 0.1)
    
    st.subheader("Market")
    col1, col2 = st.columns(2)
    feedback_strength = col1.slider("Feedback (α)", 0.01, 0.30, 0.05, 0.01)
    volatility = col2.slider("Volatility (σ)", 0.005, 0.05, 0.01, 0.005)
    
    st.subheader("Agents")
    col1, col2 = st.columns(2)
    J_horizontal = col1.slider("J₀ (Peer)", 0.5, 2.0, 1.0, 0.1)
    J_vertical = col2.slider("J₁ (Authority)", 1.0, 5.0, 2.0, 0.5)
    temperature = col1.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    
    st.subheader("Settings")
    scenario_type = st.selectbox("Scenario", ["Bubble & Crash", "Normal Market", "Shock Event"])
    n_steps = st.slider("Total Steps", 500, 2000, 1000, 100)
    
    if scenario_type == "Bubble & Crash":
        col1, col2 = st.columns(2)
        bubble_start = col1.slider("Bubble Start", 100, 500, 200, 50)
        crash_time = col2.slider("Crash Time", 500, 1500, 800, 50)
    elif scenario_type == "Shock Event":
        col1, col2 = st.columns(2)
        shock_time = col1.slider("Shock Time", 100, 900, 500, 50)
        shock_magnitude = col2.slider("Shock Mag", -1.0, 1.0, 0.5, 0.1)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    if col1.button("🎬 Start", use_container_width=True):
        st.session_state.market = HierarchicalMarket(
            n_levels=n_levels,
            branching_factor=branching_factor,
            lambda_ratio=lambda_ratio,
            n_top_agents=n_top_agents,
            initial_price=100.0,
            feedback_strength=feedback_strength,
            drift=0.0002,
            volatility=volatility
        )
        st.session_state.market.J_horizontal = J_horizontal
        st.session_state.market.J_vertical = J_vertical
        st.session_state.market.temperature = temperature
        st.session_state.scenario_type = scenario_type
        st.session_state.initialized = True
        st.session_state.running = True
        st.session_state.step = 0
        st.session_state.prices = []
        st.session_state.magnetizations = []
        st.session_state.returns = []
        st.rerun()
    
    if col2.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
        st.rerun()
    
    if col3.button("🔄 Reset", use_container_width=True):
        st.session_state.initialized = False
        st.session_state.running = False
        st.session_state.step = 0
        st.session_state.prices = []
        st.session_state.magnetizations = []
        st.session_state.returns = []
        st.rerun()
    
    steps_per_update = st.slider("Steps/Update", 1, 20, 5, 1)
    
    st.subheader("Visualization")
    viz_type = st.radio("Network View", 
                       ["2D Hierarchy", "Grid (Level 0)", "3D Multi-Level"],
                       index=0)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🎮 Market Simulation")

if not st.session_state.initialized:
    st.info("👈 Configure parameters and click **Start**")
else:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Step", st.session_state.step)
    
    with col2:
        if len(st.session_state.prices) > 0:
            current_price = st.session_state.prices[-1]
            initial_price = st.session_state.prices[0]
            change = (current_price / initial_price - 1) * 100
            st.metric("Price", f"${current_price:.2f}", f"{change:+.1f}%")
        else:
            st.metric("Price", "$100.00")
    
    with col3:
        if len(st.session_state.magnetizations) > 0:
            st.metric("Magnetization", f"{st.session_state.magnetizations[-1]:.3f}")
        else:
            st.metric("Magnetization", "0.000")
    
    with col4:
        st.metric("Agents", len(st.session_state.market.network.agents))
    
    # Visualizations
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Market Dynamics")
        
        if len(st.session_state.prices) > 1:
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            time = np.arange(len(st.session_state.prices))
            
            # Price
            axes[0].plot(time, st.session_state.prices, 'b-', linewidth=2)
            if scenario_type == "Bubble & Crash":
                axes[0].axvline(bubble_start, color='orange', linestyle='--', alpha=0.7)
                axes[0].axvline(crash_time, color='red', linestyle='--', alpha=0.7)
            axes[0].set_ylabel("Price ($)")
            axes[0].set_title("Price Evolution", fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Returns (replacing log price)
            if len(st.session_state.returns) > 1:
                # Returns start from step 1 (since first return is at step 0)
                returns_time = np.arange(len(st.session_state.returns))
                axes[1].plot(returns_time, st.session_state.returns, 'purple', linewidth=1.5, alpha=0.8)
                axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
                axes[1].set_ylabel("Returns")
                axes[1].set_title("Market Returns", fontweight='bold')
                axes[1].grid(alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, "Waiting for data...", 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title("Market Returns", fontweight='bold')
            
            # Magnetization
            axes[2].plot(time, st.session_state.magnetizations, 'green', linewidth=2)
            axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Magnetization")
            axes[2].set_title("Market Sentiment", fontweight='bold')
            axes[2].set_ylim([-1.1, 1.1])
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Running...")
    
    with col2:
        st.subheader("🌳 Agent Network")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if viz_type == "2D Hierarchy":
            st.session_state.market.network.visualize_hierarchy(ax=ax, show_states=True)
        
        elif viz_type == "Grid (Level 0)":
            st.session_state.market.network.visualize_grid_with_network(ax=ax, show_states=True)
        
        else:  # 3D Multi-Level
            plt.close()
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            st.session_state.market.network.visualize_3d_hierarchy(ax=ax, show_states=True)
        
        st.pyplot(fig)
        plt.close()

# Auto-run
if st.session_state.running and st.session_state.initialized:
    for _ in range(steps_per_update):
        if st.session_state.step < n_steps:
            if scenario_type == "Bubble & Crash":
                if st.session_state.step < bubble_start:
                    pass
                elif st.session_state.step < crash_time:
                    progress = (st.session_state.step - bubble_start) / (crash_time - bubble_start)
                    st.session_state.market.feedback_strength = feedback_strength * (1 + 5 * progress)
                    st.session_state.market.temperature = temperature * (1 - 0.5 * progress)
                else:
                    st.session_state.market.feedback_strength = feedback_strength
                    st.session_state.market.temperature = temperature * 3.0
                    st.session_state.market.external_field = -0.5
            
            state = st.session_state.market.step()
            st.session_state.prices.append(state.price)
            st.session_state.magnetizations.append(state.magnetization)
            st.session_state.returns.append(state.returns)
            st.session_state.step += 1
        else:
            st.session_state.running = False
            break
    st.rerun()


st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if st.button("⬅️ Previous: Theory", use_container_width=True):
        st.switch_page("pages/1_Theory.py")

with col3:
    if st.button("Next: LPPL Analysis ➡️", use_container_width=True):
        st.switch_page("pages/3_LPPL_Analysis.py")