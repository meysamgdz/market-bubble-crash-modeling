"""Theory Page - Mathematical foundations and concepts"""

import streamlit as st

st.set_page_config(layout="wide", page_title="Theory")

st.title("🎓 Theoretical Background")

# ============================================================================
# SIDEBAR - Theory navigation
# ============================================================================

with st.sidebar:
    st.header("📚 Theory Topics")
    
    theory_section = st.selectbox(
        "Select Topic",
        [
            "Overview",
            "Hierarchical Ising Model", 
            "Discrete Scale Invariance",
            "Log-Periodic Power Laws",
            "Complex Fractal Dimensions",
            "Crash Mechanisms"
        ]
    )

# ============================================================================
# THEORY CONTENT
# ============================================================================

if theory_section == "Overview":
    st.markdown("""
                ## Hierarchies, Complex Fractal Dimensions, and Log-Periodicity
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Question
        **Why do markets crash?** Can we predict them?
                    """)
    with col2:
        st.markdown("""
        ### Main Insight
        Financial crashes are not random "black swans" but rather **predictable "dragon kings"** 
        that emerge from the hierarchical structure of markets.
        """)
    
    st.divider()    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Concepts
        
        **1. Hierarchical Organization**
        - Markets have natural levels: day traders → institutions
        - Each level operates at different time scales
        - Higher levels influence lower levels
        
        **2. Discrete Scale Invariance**
        - Patterns repeat at specific scale ratios (λ ≈ 2)
        - Not continuous fractals, but discrete scales
        - Creates log-periodic oscillations
        """)
    
    with col2:
        st.markdown("""
        ###        
        **3. Log-Periodic Power Laws (LPPL)**
        - P(t) = A + B(tc-t)^m [1 + C·cos(ω·log(tc-t) + φ)]
        - Predicts critical time tc (crash time)
        - Observable accelerating oscillations
        
        **4. Critical Point Dynamics**
        - System approaches instability
        - Small trigger → large crash
        - Similar to phase transitions in physics
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Why This Matters
        
        - ✅ Crashes are somewhat predictable
        - ✅ Warning signs exist (LPPL patterns)
        """)
    
    with col2:
        st.markdown("""
        ### 
        
        - ✅ Understanding mechanism → better risk
        - ✅ Not random - emergent from structure
        """)

elif theory_section == "Hierarchical Ising Model":
    st.markdown("""
        ## Hierarchical Ising Model
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### From Physics to Finance
        
        **Ising Model (1925):**
        - Originally for magnetism
        - Spins: ↑ or ↓
        - Energy minimization
        """)
    with col2:
        st.markdown("""
        ### 
        
        **Applied to Markets:**
        - Traders as "spins"
        - ↑ = Bullish, ↓ = Bearish
        - Herding through energy minimization
        """)
    
    st.divider()
    
    st.markdown("""
        ### Mathematical Framework
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Energy Function:**
        ```
        E = -J₀ Σᵢⱼ sᵢsⱼ - J₁ Σₖₗ Sₖsₗ - h Σᵢ sᵢ
        ```
        
        - sᵢ ∈ {-1, +1}: agent i's state
        - J₀: horizontal coupling (peer)
        - J₁: vertical coupling (authority)
        - h: external field (news)
        """)
    
    with col2:
        st.markdown("""
        **Metropolis Dynamics:**
        ```
        P = min(1, exp(-ΔE/T))
        ```
        
        - T = temperature (noise)
        - Low T → strong herding
        - High T → random behavior
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Hierarchy Structure
        
        **Time Scales:**
        ```
        τₙ = λⁿ · τ₀
        ```
        
        **Example (λ=2):**
        - Level 0: τ = 1 (seconds/minutes)
        - Level 1: τ = 2 (hours)
        - Level 2: τ = 4 (days)
        - Level 3: τ = 8 (weeks)
        """)
    with col2:
        st.markdown("""
        ### Why Hierarchy Matters
        
        - Realistic market organization
        - Creates DSI naturally
        - Amplification through levels
        - Predictable patterns emerge
        """)

elif theory_section == "Discrete Scale Invariance":
    st.markdown("""
        ## Discrete Scale Invariance (DSI)
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""   
        ### Scale Invariance Types
        
        **Continuous (Fractals):**
        - Pattern same at ALL scales
        - f(x) ≈ f(ax) for any a
        - Example: coastline, trees
        """)
    
    with col2:
        st.markdown("""
        ### 
        
        **Discrete:**
        - Pattern at SPECIFIC scales
        - f(x) ≈ f(λx), f(λ²x), f(λ³x)...
        - λ = preferred scaling ratio
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### In Market Crashes
        
        **Oscillations speed up geometrically:**
        
        ```
        Time to crash:  100 days
        Wobble at:      50 days    (λ = 2)
        Wobble at:      25 days    (λ = 2)
        Wobble at:      12.5 days  (λ = 2)
        Wobble at:      6.25 days  (λ = 2)
        ```
        """)
    
    with col2:
        st.markdown("""
        ### Why λ ≈ 2?
        
        **Three reasons:**
        
        - **Binary decisions**: Buy/Sell
        - **Binary branching**: 2 children per parent
        - **Natural organization**: Powers of 2
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Mathematical Origin
        
        **Renormalization group:**
        - Average lower → effective higher
        - Creates preferred scale λ
        - Related by: λ = exp(2π/p)
        
        **Complex fractal dimension:**
        - D = d + ip
        - Real part d: roughness
        - Imaginary part p: log-periodicity
        """)
    
    with col2:
        st.markdown("""
        ### Observable Signatures
        
        - ✓ Accelerating oscillations
        - ✓ Geometric spacing of peaks
        - ✓ Ratio of intervals ≈ constant
        - ✓ Pattern self-similar in log-time
        """)

elif theory_section == "Log-Periodic Power Laws":
    st.markdown("""
        ## Log-Periodic Power Laws (LPPL)
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### The LPPL Formula
        
        ```
        P(t) = A + B(tc - t)^m [1 + C·cos(ω·log(tc - t) + φ)]
        ```
        """)
    
    with col2:
        st.markdown("""
        ### Why "Log-Periodic"?
        
        **Normal periodic:**
        - f(t) = cos(ωt)
        - Period T constant
        - Repeats every T
        
        **Log-periodic:**
        - f(t) = cos(ω·log(t))
        - Period geometric
        - Repeats every factor λ
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Parameter Meanings
        
        **Power Law Component:**
        - **A**: Price level at crash
        - **B**: Amplitude (positive for bubble)
        - **m**: Exponent (typically 0.2-0.5)
        - Gives: super-exponential growth
        
        **Log-Periodic Component:**
        - **ω**: Frequency (typically 5-15)
        - **C**: Oscillation amplitude (-1 to 1)
        - **φ**: Phase shift (0 to 2π)
        - Gives: accelerating wobbles
        """)
    
    with col2:
        st.markdown("""
        ### Critical Point
        
        - **tc**: Time of crash
        - System singular at t = tc
        - Requires trigger to crash
        
        ### Visual Pattern
        
        ```
        Price
          |                /|/|/|/|  ← Very fast
          |           /|/|/|         ← Fast
          |      /|/|/               ← Medium
          |   /|/                    ← Slow
          | /
          |/__________________ Time → tc
        ```
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Historical Examples
        
        **Confirmed crashes:**
        - October 1987: Black Monday
        - Dot-com 2000
        - Financial Crisis 2008
        - Many others...
        """)
    
    with col2:
        st.markdown("""
        ### 
        
        All showed LPPL patterns before crash!
        
        **Key insight:** Pattern is detectable
        in advance, providing warning signal.
        """)

elif theory_section == "Complex Fractal Dimensions":
    st.markdown("""
        ## Complex Fractal Dimensions
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Standard Fractal Dimension
        
        **Real number D:**
        - D = 1: Line
        - D = 1.5: Rough curve
        - D = 2: Surface
        
        Measures "roughness"
        """)
    
    with col2:
        st.markdown("""
        ### Complex Extension
        
        **D = d + ip**
        
        Where:
        - **d**: Real part (roughness)
        - **p**: Imaginary part (oscillations)
        - Both have physical meaning!
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Why Complex?
        
        **Power law with oscillations:**
        
        ```
        f(x) ∝ x^D = x^(d+ip)
        ```
        
        Using: x^(ip) = e^(ip·log x)
        
        Result:
        ```
        f(x) ∝ x^d · cos(p·log x)
        ```
        
        **This is exactly log-periodic!**
        """)
    
    with col2:
        st.markdown("""
        ### Connection to λ
        
        ```
        λ = exp(2π/p)
        ```
        
        **Example:**
        - If p = 9, then λ ≈ 1.9
        - If p = 7, then λ ≈ 2.7
        - Typical: p ≈ 9, λ ≈ 2
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### In Market Crashes
        
        **Typical values:**
        - d ≈ 0.5-0.7 (slightly rough)
        - p ≈ 7-11 (λ ≈ 1.8-2.2)
        """)
    
    with col2:
        st.markdown("""
        ### Why This Matters
        
        - Predicts oscillation frequency
        - Links structure (λ) to patterns (ω)
        - Not arbitrary - from physics!
        """)

else:  # Crash Mechanisms
    st.markdown("""
        ## Crash Mechanisms
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Phase 1: Normal Market
        
        **Characteristics:**
        - Low positive feedback
        - Mixed opinions (M ≈ 0)
        - Normal volatility
        """)
    
    with col2:
        st.markdown("""
        ### Phase 2: Bubble Formation
        
        **Mechanism:**
        1. Small positive shock
        2. Some agents go bullish
        3. Positive feedback amplifies
        4. More agents follow (herding)
        5. Feedback increases further
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Observable:**
        - Accelerating price growth
        - Increasing magnetization
        - Log-periodic oscillations appear
        
        ### Phase 3: Critical Point
        
        **System characteristics:**
        - Very high magnetization (|M| → 1)
        - Low temperature (strong herding)
        - Extremely unstable
        """)
    
    with col2:
        st.markdown("""
        ### Phase 4: Crash
        
        **Trigger:**
        - Random: noise increase
        - External: bad news
        - Internal: profit-taking
        
        **Cascade:**
        1. Top level flips bearish
        2. Influences next level down
        3. Cascade accelerates
        4. Price crashes
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Insights
        
        - ✅ Bubble formation: detectable
        - ✅ Crash likely: yes
        """)
    
    with col2:
        st.markdown("""
        ### 
        
        - ❌ Exact crash time: uncertain
        - ❌ Trigger: unknown
        """)

# Key Equations Reference
with st.expander("📐 Key Equations Reference"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Equations
        
        **1. Energy (Ising Model)**
        ```
        E = -J₀ Σᵢⱼ sᵢsⱼ - J₁ Σₖₗ Sₖsₗ - h Σᵢ sᵢ
        ```
        
        **2. Magnetization**
        ```
        M(t) = (1/N) Σᵢ sᵢ(t)
        ```
        
        **3. Returns**
        ```
        r(t) = μ + α·M(t) + σ·ε(t)
        ```
        
        **4. Price Evolution**
        ```
        P(t) = P(0) · exp(∫₀ᵗ r(τ) dτ)
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 
        
        **5. LPPL Formula**
        ```
        P(t) = A + B(tc - t)^m [1 + C·cos(ω·log(tc - t) + φ)]
        ```
        
        **6. Time Scales**
        ```
        τₙ = λⁿ · τ₀
        ```
        
        **7. Scaling Ratio**
        ```
        λ = exp(2π/p)
        ```
        
        **8. Complex Dimension**
        ```
        D = d + ip
        ```
        """)

st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])

with col3:
    if st.button("Next: Simulation ➡️", use_container_width=True):
        st.switch_page("pages/2_Simulation.py")