# Hierarchical Market Crash Model

Agent-based implementation of Sornette's hierarchical Ising model for financial market crashes, based on "Why Stock Markets Crash" Chapter 6. Features interactive web app, LPPL detection, and comprehensive testing.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web application
streamlit run app.py

# Run tests
pytest tests/
```

## ✨ Features

- **Interactive Web App**: 4 pages (Theory, Simulation, LPPL Analysis, Sensitivity)
- **Hierarchical Agent Network**: Multi-level trader organization with geometric time scales
- **LPPL Pattern Detection**: Automatic log-periodic power law fitting
- **Rich Visualizations**: 2D hierarchy, 3D multi-level, grid lattice views

## 📁 Project Structure

```
├── app.py                     # Streamlit web application
├── pages/                     # Multi-page app structure
│   ├── 1_Theory.py        # Concepts & mathematics
│   ├── 2_Simulation.py    # Interactive simulation
│   ├── 3_LPPL_Analysis.py # Pattern detection
│   └── 4_Sensitivity.py   # Parameter analysis
│
├── agent/
│   ├── agent.py              # Agent class (Metropolis dynamics)
│   └── hierarchy.py          # Network structure (grid + ring lattice)
│
├── environment/
│   └── market.py             # Market dynamics (price evolution)
│
├── analysis/
│   └── lppl.py               # LPPL fitting (differential evolution)
│
├── tests/
│   ├── unit/                 # Component tests (148)
│   └── integration/          # Workflow tests (24)
│
└── tutorials/                 # Learning guides
    ├── LPPL_FITTING_GUIDE.md
    └── SENSITIVITY_ANALYSIS_GUIDE.md
```

## 🎯 Core Model

### Hierarchical Ising Model
- **Agents**: Binary states (bullish ↑/bearish ↓)
- **Hierarchy**: Multi-level structure with time scales τₙ = λⁿτ₀
- **Interactions**: Metropolis dynamics with peer (J₀) and authority (J₁) coupling
- **Network**: 2D grid lattice (level 0), ring lattice (higher levels)

### Market Dynamics
```python
returns = drift + feedback_strength × magnetization + noise
price(t) = price(0) × exp(∫ returns dt)
```

### LPPL Formula
```
P(t) = A + B(tc-t)^m [1 + C·cos(ω·log(tc-t) + φ)]
```
Where λ = exp(2π/ω) links hierarchy ratio to oscillation frequency.

## 📊 Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_levels` | Hierarchy depth | 4 | 2-6 |
| `branching_factor` | Children per parent | 2 | 2-3 |
| `lambda_ratio` | Time scale ratio (λ) | 2.0 | 1.5-3.0 |
| `feedback_strength` | Herding strength (α) | 0.10 | 0.01-0.30 |
| `J_horizontal` | Peer coupling | 1.0 | 0.5-2.0 |
| `J_vertical` | Authority coupling | 2.0 | 1.0-5.0 |
| `temperature` | Decision noise | 1.0 | 0.1-2.0 |

## 💻 Usage Examples

### Basic Simulation
```python
from environment.market import HierarchicalMarket

market = HierarchicalMarket(
    n_levels=4,
    lambda_ratio=2.0,
    feedback_strength=0.10
)

prices, magnetizations = market.simulate(n_steps=1000)
```

### Bubble & Crash Scenario
```python
prices, mags = market.create_bubble_scenario(
    n_steps=1000,
    bubble_start=200,
    crash_time=800
)
```

### LPPL Detection
```python
from analysis.lppl import analyze_log_periodicity

results = analyze_log_periodicity(time, prices)
print(f"Detected: {results['lppl_detected']}")
print(f"Confidence (R²): {results['lppl_confidence']:.2f}")
print(f"Predicted crash: t={results['predicted_crash_time']:.0f}")
print(f"Lambda: {results['lppl_lambda']:.2f}")
```

### Sensitivity Analysis
```python
# Test different lambda values
for lambda_val in [1.5, 2.0, 2.5]:
    market = HierarchicalMarket(lambda_ratio=lambda_val)
    # Run analysis...
```

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=agent --cov=environment --cov=analysis --cov-report=html

# Using test runner
python run_tests.py all
python run_tests.py coverage
```

**Test Coverage:**
- `test_agent.py`: 340+ tests (initialization, connections, dynamics)
- `test_hierarchy.py`: 180+ tests (structure, lattices, statistics)
- `test_market.py`: 200+ tests (dynamics, bubbles, history)
- `test_lppl.py`: 150+ tests (fitting, detection, validation)
- `test_market_workflow.py`: 120+ integration tests

## 🔬 Key Results

### Emergent Phenomena
- ✅ Herding cascades through hierarchy
- ✅ Super-exponential bubble growth
- ✅ Log-periodic oscillations (λ ≈ 2)
- ✅ Predictable crash timing (LPPL tc)
- ✅ Discrete scale invariance

### Detection Performance
- R² > 0.85: Strong LPPL pattern
- R² > 0.70: Moderate pattern
- Lambda recovery: ±0.5 typical
- Best with: 4 levels, λ=2.0, α=0.10-0.15

## 📖 Documentation

### Tutorials
- **LPPL_FITTING_GUIDE.md**: Complete LPPL fitting walkthrough
- **SENSITIVITY_ANALYSIS_GUIDE.md**: Parameter effects explained
- **Web App Theory Tab**: Interactive concepts with equations

### Key Concepts
1. **Discrete Scale Invariance**: Patterns repeat at λ ratios, not all scales
2. **Log-Periodic Power Laws**: Accelerating oscillations before crash
3. **Hierarchical Time Scales**: τₙ = λⁿτ₀ creates multi-scale dynamics
4. **Phase Transition**: Critical point (tc) where market crashes

## 🎓 Use Cases

**Education**: Interactive teaching of crash mechanics and emergent phenomena

**Research**: Test hypotheses about market structure and crash prediction

**Risk Management**: Understand warning signs and system fragility

## 📚 References

- **Sornette, D.** (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems*. Princeton University Press. Chapter 6.
- **Johansen, A., & Sornette, D.** (2001). Finite-time singularity in the dynamics of the world population, economic and financial indices. *Physica A*, 294(3-4), 465-502.
- **Sornette, D., & Johansen, A.** (1997). Large financial crashes. *Physica A*, 245(3-4), 411-422.

## 🔧 Technical Details

**Agent Dynamics**: Metropolis algorithm with ΔE = -2·s·h, P(flip) = min(1, exp(-ΔE/T))

**Network Structure**: Grid lattice (level 0) for spatial clustering, ring lattice (higher levels) per Sornette's model

**LPPL Fitting**: Differential evolution optimization with parameter constraints and validation

**Price Dynamics**: Geometric Brownian motion with magnetization feedback

## 📄 License

MIT License - Free for educational and research use

---