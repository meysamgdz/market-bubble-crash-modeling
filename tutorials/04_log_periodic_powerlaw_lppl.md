# Tutorial 4: Log-Periodicity Detection & LPPL Analysis

**Learning Objectives:**
1. Understand the LPPL formula components
2. Fit LPPL models to price data
3. Detect discrete scale invariance
4. Interpret parameter values
5. Assess prediction confidence

## Introduction

This tutorial teaches you to detect log-periodic patterns that signal approaching crashes. You'll learn to fit the LPPL model and interpret results.

## Theory: LPPL Formula

### The Complete Equation

```
P(t) = A + B(tc - t)^m [1 + C·cos(ω·log(tc - t) + φ)]
```

### Parameter Meanings

**Power Law Component:**
- **A**: Price level at critical point
- **B**: Amplitude (positive for bubbles)
- **m**: Power law exponent
  - Typical: 0.2 - 0.5
  - Controls growth acceleration

**Log-Periodic Component:**
- **ω**: Angular frequency
  - Typical: 5 - 15
  - Determines oscillation speed
- **C**: Oscillation amplitude
  - Range: -1 to +1
  - Strength of wobbles
- **φ**: Phase shift (0 to 2π)

**Critical Point:**
- **tc**: Critical time
  - When crash occurs
  - System singular at t = tc

### Derived Parameter

**Scaling ratio λ:**
```
λ = exp(2π / ω)
```

Expected: λ ≈ 2 from hierarchy structure

## Exercise 1: Manual LPPL Exploration

Understand each parameter's effect:

```python
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from analysis.lppl import lppl_function

# Create synthetic data
time = np.linspace(0, 1, 200)

# Base parameters
tc = 1.1
A = 4.6
B = 0.2
m = 0.3
omega = 10
C = 0.1
phi = 0

# Generate LPPL pattern
log_price = lppl_function(time, tc, A, B, m, omega, C, phi)
price = np.exp(log_price)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Base pattern
axes[0, 0].plot(time, price, 'b-', linewidth=2)
axes[0, 0].axvline(tc, color='red', linestyle='--', label=f'tc={tc}')
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Price")
axes[0, 0].set_title("Base LPPL Pattern")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Vary m (power law exponent)
for m_val in [0.2, 0.3, 0.5]:
    p = np.exp(lppl_function(time, tc, A, B, m_val, omega, C, phi))
    axes[0, 1].plot(time, p, linewidth=2, label=f'm={m_val}')
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Price")
axes[0, 1].set_title("Effect of m (acceleration)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Vary ω (frequency)
for omega_val in [5, 10, 15]:
    p = np.exp(lppl_function(time, tc, A, B, m, omega_val, C, phi))
    axes[1, 0].plot(time, p, linewidth=2, label=f'ω={omega_val}')
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Price")
axes[1, 0].set_title("Effect of ω (oscillation speed)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Vary C (oscillation amplitude)
for C_val in [0, 0.1, 0.3]:
    p = np.exp(lppl_function(time, tc, A, B, m, omega, C_val, phi))
    axes[1, 1].plot(time, p, linewidth=2, label=f'C={C_val}')
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Price")
axes[1, 1].set_title("Effect of C (wobble strength)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Parameter effects:")
print("  m ↑ → Faster acceleration")
print("  ω ↑ → More frequent oscillations")
print("  C ↑ → Stronger wobbles")
```

## Exercise 2: Fitting LPPL to Simulated Data

Fit model to bubble scenario:

```python
from environment.market import HierarchicalMarket
from analysis.lppl import fit_lppl, detect_lppl_pattern

# Generate bubble data
market = HierarchicalMarket(
    n_levels=4,
    branching_factor=2,
    lambda_ratio=2.0,
    feedback_strength=0.10  # Strong feedback for clear pattern
)

prices, _ = market.create_bubble_scenario(
    n_steps=1000,
    bubble_start=200,
    crash_time=800
)

# Analyze pre-crash data only
time = np.arange(800)
prices_pre_crash = prices[:800]

print("Fitting LPPL model to bubble data...")
print("This may take 30-60 seconds...")

# Fit LPPL
params, error = fit_lppl(time, prices_pre_crash, method='differential_evolution')

print("\n=== Fitted Parameters ===")
print(f"tc (predicted crash): {params['tc']:.1f}")
print(f"A (price at crash):   {params['A']:.3f} (log)")
print(f"B (amplitude):        {params['B']:.3f}")
print(f"m (power law):        {params['m']:.3f}")
print(f"ω (frequency):        {params['omega']:.2f}")
print(f"C (oscillation):      {params['C']:.3f}")
print(f"λ (scaling ratio):    {params['lambda']:.2f}")
print(f"\nFit error: {error:.4f}")

# Check if parameters in typical ranges
print("\n=== Parameter Validation ===")
print(f"m in [0.1, 0.9]:      {'✓' if 0.1 < params['m'] < 0.9 else '✗'}")
print(f"ω in [5, 15]:         {'✓' if 5 < params['omega'] < 15 else '✗'}")
print(f"|C| < 1:              {'✓' if abs(params['C']) < 1 else '✗'}")
print(f"tc > last time:       {'✓' if params['tc'] > time[-1] else '✗'}")
print(f"λ ≈ 2:                {'✓' if abs(params['lambda'] - 2.0) < 0.5 else '✗'}")

# Visualize fit
from analysis.lppl import plot_lppl_fit

fig, ax = plt.subplots(figsize=(10, 6))
plot_lppl_fit(time, prices_pre_crash, params, ax=ax)
plt.show()

print(f"\nActual crash time: 800")
print(f"Predicted crash time: {params['tc']:.0f}")
print(f"Error: {abs(params['tc'] - 800):.0f} time units")
```

**Key observations:**
- Well-fitted model should have R² > 0.85
- Predicted tc should be close to actual crash (800)
- λ should be near 2.0 (our hierarchy parameter)

## Exercise 3: Pattern Detection with Quality Check

Use automatic detection:

```python
from analysis.lppl import analyze_log_periodicity

# Generate data
market = HierarchicalMarket(
    n_levels=4,
    lambda_ratio=2.0,
    feedback_strength=0.08
)

prices, _ = market.create_bubble_scenario(n_steps=1000)
time = np.arange(800)
prices_window = prices[:800]

# Analyze
print("Analyzing log-periodic patterns...\n")
results = analyze_log_periodicity(time, prices_window, verbose=True)

# Detailed interpretation
print("\n=== Interpretation ===")

if results['lppl_detected']:
    print("✅ STRONG BUBBLE SIGNAL DETECTED")
    print(f"\nPredicted crash in: {results['time_to_crash']:.0f} time units")
    print(f"Confidence level: {results['lppl_confidence']:.1%}")
    
    # Check consistency
    fitted_lambda = results['lppl_lambda']
    expected_lambda = 2.0
    
    if abs(fitted_lambda - expected_lambda) < 0.3:
        print(f"✓ λ matches hierarchy structure ({expected_lambda:.1f})")
    else:
        print(f"⚠ λ mismatch: fitted={fitted_lambda:.2f}, expected={expected_lambda:.1f}")
    
    # Risk assessment
    m_val = results['lppl_params']['m']
    if m_val < 0.3:
        print("⚠ Low m → slow acceleration, may have time")
    elif m_val > 0.5:
        print("🔴 High m → rapid acceleration, crash imminent!")
    else:
        print("⚙ Moderate m → normal bubble dynamics")
        
else:
    print("❌ No clear pattern detected")
    print(f"Confidence only: {results['lppl_confidence']:.1%}")
    print("\nPossible reasons:")
    print("  - Insufficient data")
    print("  - Weak bubble dynamics (try higher α)")
    print("  - Random fluctuations dominating")
    print("  - Not yet in critical phase")
```

## Exercise 4: Discrete Scale Invariance Analysis

Check for geometric spacing in oscillations:

```python
from scipy.signal import find_peaks
from analysis.lppl import compute_scaling_ratios

# Generate clear bubble
market = HierarchicalMarket(
    n_levels=4,
    lambda_ratio=2.0,
    feedback_strength=0.12  # Very strong for clear pattern
)

prices, _ = market.create_bubble_scenario(n_steps=1000, crash_time=800)

# Focus on bubble phase
bubble_prices = prices[200:800]
bubble_time = np.arange(len(bubble_prices))

# Find peaks in price
peaks, properties = find_peaks(bubble_prices, distance=20, prominence=2)

print("=== Discrete Scale Invariance Check ===\n")
print(f"Found {len(peaks)} peaks in price series")

if len(peaks) >= 4:
    # Get peak times
    peak_times = bubble_time[peaks]
    
    print("\nPeak times:")
    for i, t in enumerate(peak_times[-5:]):  # Last 5 peaks
        print(f"  Peak {i+1}: t = {t}")
    
    # Compute intervals
    intervals = np.diff(peak_times[-5:])
    
    print("\nIntervals between peaks:")
    for i, dt in enumerate(intervals):
        print(f"  Δt_{i+1} = {dt}")
    
    # Compute ratios
    if len(intervals) >= 2:
        ratios = intervals[:-1] / intervals[1:]
        
        print("\nRatios of consecutive intervals:")
        for i, ratio in enumerate(ratios):
            print(f"  Δt_{i+1}/Δt_{i+2} = {ratio:.3f}")
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        print(f"\nMean ratio: {mean_ratio:.3f}")
        print(f"Std dev: {std_ratio:.3f}")
        print(f"Expected λ: 2.0")
        
        # Check consistency
        if std_ratio / mean_ratio < 0.3:
            print("\n✅ DSI DETECTED: Ratios are consistent!")
            if abs(mean_ratio - 2.0) < 0.5:
                print("✓ Ratios match expected λ = 2.0")
        else:
            print("\n⚠ Ratios too variable for strong DSI claim")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bubble_time, bubble_prices, 'b-', linewidth=2, alpha=0.7)
    ax.plot(bubble_time[peaks], bubble_prices[peaks], 'ro', markersize=10, 
            label='Peaks')
    
    # Mark intervals
    for i in range(len(peaks)-1):
        t1, t2 = bubble_time[peaks[i]], bubble_time[peaks[i+1]]
        ax.annotate('', xy=(t2, bubble_prices[peaks[i]]), 
                   xytext=(t1, bubble_prices[peaks[i]]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    ax.set_xlabel("Time (since bubble start)")
    ax.set_ylabel("Price")
    ax.set_title("Peak Detection: Checking for Geometric Spacing")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

else:
    print("Not enough peaks detected for DSI analysis")
```

**DSI signature:** Intervals decrease geometrically (divide by ~2 each time)

## Exercise 5: Quality Assessment

Compare good vs bad fits:

```python
# Generate two scenarios

# Scenario 1: Strong bubble (should detect)
print("=== Scenario 1: Strong Bubble ===")
market1 = HierarchicalMarket(
    n_levels=4,
    lambda_ratio=2.0,
    feedback_strength=0.12,
    volatility=0.01
)
prices1, _ = market1.create_bubble_scenario(n_steps=1000, crash_time=800)
result1 = analyze_log_periodicity(np.arange(800), prices1[:800], verbose=False)

print(f"Pattern detected: {result1['lppl_detected']}")
print(f"Confidence (R²): {result1['lppl_confidence']:.3f}")
print(f"Predicted tc: {result1['predicted_crash_time']:.0f}")
print(f"Fitted λ: {result1['lppl_lambda']:.2f}")

# Scenario 2: Normal market (should NOT detect)
print("\n=== Scenario 2: Normal Market ===")
market2 = HierarchicalMarket(
    n_levels=4,
    lambda_ratio=2.0,
    feedback_strength=0.02,  # Very low
    volatility=0.02           # Higher noise
)
prices2, _ = market2.simulate(n_steps=800, show_progress=False)
result2 = analyze_log_periodicity(np.arange(800), prices2, verbose=False)

print(f"Pattern detected: {result2['lppl_detected']}")
print(f"Confidence (R²): {result2['lppl_confidence']:.3f}")

# Compare
print("\n=== Comparison ===")
print(f"Bubble R²: {result1['lppl_confidence']:.3f} {'→ HIGH' if result1['lppl_confidence'] > 0.85 else ''}")
print(f"Normal R²: {result2['lppl_confidence']:.3f} {'→ LOW' if result2['lppl_confidence'] < 0.85 else ''}")

print("\nKey insight:")
print("  R² > 0.85 → Strong pattern, likely bubble")
print("  R² < 0.70 → No clear pattern, normal market")
print("  0.70 < R² < 0.85 → Uncertain, need more data")
```

## Key Concepts

### 1. LPPL Components

**Power Law:** B(tc-t)^m
- Accelerating growth
- "Super-exponential"
- Unsustainable trajectory

**Log-Periodic:** C·cos(ω·log(tc-t))
- Oscillations speed up
- Geometric spacing
- From hierarchy

**Combined:**
- Accelerating wobbles
- Distinct signature
- Predictive power

### 2. Parameter Interpretation

**m (0.2-0.5):**
- Low m → Gradual acceleration
- High m → Rapid acceleration
- Outside range → Bad fit

**ω (5-15):**
- Determines λ via: λ = exp(2π/ω)
- ω=10 → λ≈1.9
- Should match hierarchy

**C (-1 to +1):**
- Oscillation strength
- C≈0 → No wobbles (pure power law)
- |C|>0.3 → Clear log-periodicity

### 3. Quality Metrics

**R² (Coefficient of determination):**
- R² > 0.90: Excellent fit
- 0.85 < R² < 0.90: Good fit
- 0.70 < R² < 0.85: Moderate
- R² < 0.70: Poor fit

**Parameter consistency:**
- All in typical ranges?
- λ matches hierarchy?
- tc beyond data?

### 4. Detection Confidence

**High confidence requires:**
- ✓ R² > 0.85
- ✓ Parameters in range
- ✓ λ ≈ expected value
- ✓ Visual pattern clear
- ✓ DSI ratios consistent

**Low confidence indicators:**
- ✗ R² < 0.70
- ✗ Parameters out of range
- ✗ tc before data end
- ✗ No visual pattern
- ✗ Inconsistent ratios

## Practical Guidelines

### When to Trust Detection

**Trust HIGH if:**
1. R² > 0.90
2. m, ω, C all in range
3. λ within 0.3 of expected
4. Clear visual pattern
5. Multiple independent checks agree

**Be CAUTIOUS if:**
1. 0.70 < R² < 0.85
2. Some parameters borderline
3. λ differs significantly
4. Pattern not visually obvious
5. Short data window (<500 points)

**REJECT if:**
1. R² < 0.70
2. Parameters way out of range
3. tc before end of data
4. No visual pattern at all
5. Fitting fails to converge

### False Positives

LPPL can fit noise! Always check:
- Is there actual acceleration?
- Do wobbles speed up visually?
- Multiple methods agree?
- Consistent across windows?

### Window Selection

**Too short (<300 points):**
- Insufficient data
- Overfitting risk
- Unreliable tc

**Too long (>1000 points):**
- May include normal phases
- Dilutes bubble signal
- Worse fit

**Optimal: 500-800 points**
- Captures bubble phase
- Enough for fitting
- Good signal/noise

## Challenge Exercise

**Task:** Create function to scan for bubbles

```python
def scan_for_bubble(prices, min_confidence=0.85):
    """
    Scan price series with sliding window to detect bubble formation.
    
    Returns: list of (time, confidence, predicted_tc) tuples
    """
    detections = []
    
    window_size = 500
    step_size = 50
    
    for start in range(0, len(prices) - window_size, step_size):
        end = start + window_size
        window_time = np.arange(window_size)
        window_prices = prices[start:end]
        
        # Analyze window
        result = analyze_log_periodicity(
            window_time, 
            window_prices, 
            verbose=False
        )
        
        if result['lppl_confidence'] > min_confidence:
            detections.append({
                'time': end,
                'confidence': result['lppl_confidence'],
                'predicted_tc': result['predicted_crash_time'] + start,
                'lambda': result['lppl_lambda']
            })
    
    return detections

# Test on bubble data
# ... implement and test ...
```

## Key Takeaways

1. **LPPL formula** captures bubble dynamics mathematically
2. **Seven parameters** each have physical meaning
3. **R² > 0.85** threshold for detection confidence
4. **λ ≈ 2** validates hierarchical origin
5. **DSI ratios** provide independent confirmation
6. **Quality checks** essential to avoid false positives
7. **Visual inspection** always complements statistics

## Next Steps

You now know how to:
- ✅ Understand LPPL components
- ✅ Fit models to data
- ✅ Assess quality
- ✅ Detect DSI
- ✅ Interpret parameters
- ✅ Judge confidence

**Use the web app's LPPL Analysis tab** to explore interactively!

**Experiment with:**
- Different feedback strengths
- Various λ values
- Window sizes
- Real market data (if available)

Remember: LPPL is a warning tool, not a crystal ball. Use it as part of comprehensive risk assessment!