"""
Hierarchical Market Model - Main Application

Interactive visualization of financial market crashes through hierarchical
agent-based modeling with log-periodic power law patterns.
"""

import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="Hierarchical Market Crash Model",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📉 Hierarchical Market Crash Model")
st.markdown("Agent-Based Model of Financial Bubbles and Crashes with Log-Periodic Patterns")
st.markdown("""
## Welcome

This interactive application explores how financial market crashes emerge from hierarchical 
agent interactions and exhibit predictable log-periodic patterns.

            """)
col1, col2, col3 = st.columns([1.35, 1.2, 1])

with col1:
    st.markdown("""
    ### Navigation

    Use the sidebar to navigate between:

    1. **Theory**: Mathematical foundations and concepts
    2. **Simulation**: Run real-time market simulations
    3. **LPPL Analysis**: Detect and analyze crash patterns
    4. **Sensitivity Analysis**: Explore parameter effects
""")
    
with col2:
    st.markdown("""
    ### Quick Start

    1. Start with **Theory** to understand concepts
    2. Try **Simulation** with default parameters
    3. Use **LPPL Analysis** to detect patterns
    4. Run **Sensitivity Analysis** to explore

""")
    
with col3:
    st.markdown("""
    ### Key Features

    - Interactive parameter controls
    - Real-time visualization
    - Automatic pattern detection
    - Comprehensive sensitivity analysis
    - Integrated theoretical background

""")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""

**Key References:**
- Sornette, D. (2003). *Why Stock Markets Crash*. Princeton University Press.
- Johansen, A., & Sornette, D. (2001). Finite-time singularity in dynamics.
""")