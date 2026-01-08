# Tutorial 3: Market Dynamics & Price Formation

**Learning Objectives:**
1. Connect agent states to market prices
2. Understand positive feedback mechanisms
3. Observe super-exponential bubble growth
4. Learn crash triggering dynamics

## Introduction

Now we connect the hierarchical Ising model to actual market prices. This tutorial shows how collective agent behavior (magnetization) drives price dynamics.

## Theory: From Spins to Prices

### Magnetization → Returns

**Magnetization** = average agent state:
```
M(t) = (1/N) Σᵢ sᵢ(t)
```

- M = +1: All bullish
- M = 0: Mixed opinions
- M = -1: All bearish

**Returns** depend on magnetization:
```
r(t) = μ + α·M(t) + σ·ε(t)
```

Where:
- μ = drift (fundamental growth, typically ~0.0002/day)
- α = feedback strength (key parameter!)
- σ = volatility (noise)
- ε(t) = random shock

### Price Integration

**Price evolves** by cumulative returns:
```
P(t) = P(0) · exp(∫₀ᵗ r(τ) dτ)
```

**Approximation:**
```
P(t+Δt) = P(t) · exp(r(t)·Δt)
```

### Positive Feedback (α)

**Key insight:** α > 0 creates amplification

- **Small α** (0.01-0.03): Weak herding, mild bubbles
- **Medium α** (0.05-0.10): Moderate bubbles
- **Large α** (0.15+): Extreme bubbles, unstable

**Why feedback matters:**
1. M increases → r increases
2. Higher r → price rises
3. Momentum attracts more bulls
4. M increases further → **amplification**

## Exercise 1: Basic Price Formation

```python
from environment.market import HierarchicalMarket
import numpy as np
import matplotlib.pyplot as plt

# Create simple market
market = HierarchicalMarket(
    n_levels=3,
    branching_factor=2,
    feedback_strength=0.05,
    volatility=0.01
)

# Run normal simulation
prices, mags = market.simulate(n_steps=500, show_progress=False)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Price
axes[0].plot(prices, 'b-', linewidth=2)
axes[0].set_ylabel("Price ($)")
axes[0].set_title("Price Evolution (Normal Market)")
axes[0].grid(alpha=0.3)

# Magnetization
axes[1].plot(mags, 'purple', linewidth=2)
axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Magnetization")
axes[1].set_title("Market Sentiment")
axes[1].set_ylim([-1.1, 1.1])
axes[1].grid(alpha=0.3)

plt.tight_layout()

# Statistics
print(f"\nPrice statistics:")
print(f"  Initial: ${prices[0]:.2f}")
print(f"  Final: ${prices[-1]:.2f}")
print(f"  Change: {(prices[-1]/prices[0]-1)*100:.1f}%")
print(f"  Volatility: {np.std(np.diff(np.log(prices)))*100:.2f}%")

print(f"\nMagnetization:")
print(f"  Mean: {np.mean(mags):.3f}")
print(f"  Std: {np.std(mags):.3f}")
```

**Observation:** Price wanders but magnetization stays near zero (no strong consensus).

## Exercise 2: Effect of Feedback Strength

Compare different α values:

```python
alphas = [0.01, 0.05, 0.10, 0.20]
results = []

for alpha in alphas:
    market = HierarchicalMarket(
        n_levels=3,
        branching_factor=2,
        feedback_strength=alpha,
        volatility=0.01
    )
    
    prices, mags = market.simulate(n_steps=500, show_progress=False)
    
    # Record final stats
    final_price = prices[-1]
    max_price = prices.max()
    volatility = np.std(np.diff(np.log(prices)))
    
    results.append({
        'alpha': alpha,
        'final_price': final_price,
        'max_price': max_price,
        'volatility': volatility
    })
    
    print(f"α={alpha:.2f}: Final=${final_price:.2f}, Max=${max_price:.2f}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

alphas_list = [r['alpha'] for r in results]

axes[0].plot(alphas_list, [r['final_price'] for r in results], 'o-', markersize=8)
axes[0].set_xlabel("Feedback Strength (α)")
axes[0].set_ylabel("Final Price ($)")
axes[0].set_title("Final Price vs Feedback")
axes[0].grid(alpha=0.3)

axes[1].plot(alphas_list, [r['max_price'] for r in results], 'o-', markersize=8, color='red')
axes[1].set_xlabel("Feedback Strength (α)")
axes[1].set_ylabel("Maximum Price ($)")
axes[1].set_title("Peak Price vs Feedback")
axes[1].grid(alpha=0.3)

axes[2].plot(alphas_list, [r['volatility']*100 for r in results], 'o-', markersize=8, color='green')
axes[2].set_xlabel("Feedback Strength (α)")
axes[2].set_ylabel("Volatility (%)")
axes[2].set_title("Volatility vs Feedback")
axes[2].grid(alpha=0.3)

plt.tight_layout()
```

**Key finding:** Higher α → higher prices (more amplification).

## Exercise 3: Bubble Scenario

Create controlled bubble and crash:

```python
# Create market
market = HierarchicalMarket(
    n_levels=4,
    branching_factor=2,
    feedback_strength=0.05,
    volatility=0.01
)

# Run bubble scenario
n_steps = 1000
bubble_start = 200
crash_time = 800

prices, mags = market.create_bubble_scenario(
    n_steps=n_steps,
    bubble_start=bubble_start,
    crash_time=crash_time
)

# Plot phases
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

time = np.arange(len(prices))

# Price with phase markers
axes[0].plot(time, prices, 'b-', linewidth=2)
axes[0].axvline(bubble_start, color='orange', linestyle='--', alpha=0.7, label='Bubble Start')
axes[0].axvline(crash_time, color='red', linestyle='--', alpha=0.7, label='Crash')
axes[0].set_ylabel("Price ($)")
axes[0].set_title("Price Evolution: Three Phases", fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Add phase labels
axes[0].text(100, prices.max()*0.9, "Phase 1:\nNormal", ha='center')
axes[0].text(500, prices.max()*0.9, "Phase 2:\nBubble Formation", ha='center')
axes[0].text(900, prices.max()*0.9, "Phase 3:\nCrash", ha='center')

# Log price (shows super-exponential)
axes[1].plot(time, np.log(prices), 'g-', linewidth=2)
axes[1].axvline(bubble_start, color='orange', linestyle='--', alpha=0.7)
axes[1].axvline(crash_time, color='red', linestyle='--', alpha=0.7)
axes[1].set_ylabel("Log Price")
axes[1].set_title("Log Price (Linear = Exponential Growth)", fontweight='bold')
axes[1].grid(alpha=0.3)

# Magnetization
axes[2].plot(time, mags, 'purple', linewidth=2)
axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[2].axvline(bubble_start, color='orange', linestyle='--', alpha=0.7)
axes[2].axvline(crash_time, color='red', linestyle='--', alpha=0.7)
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Magnetization")
axes[2].set_title("Market Sentiment (Herding Dynamics)", fontweight='bold')
axes[2].set_ylim([-1.1, 1.1])
axes[2].grid(alpha=0.3)

plt.tight_layout()

# Analyze each phase
print("\n=== Phase Analysis ===")

phase1 = time < bubble_start
phase2 = (time >= bubble_start) & (time < crash_time)
phase3 = time >= crash_time

for phase_name, mask in [("Normal", phase1), ("Bubble", phase2), ("Crash", phase3)]:
    phase_prices = prices[mask]
    phase_mags = mags[mask]
    
    if len(phase_prices) > 1:
        returns = np.diff(np.log(phase_prices))
        print(f"\n{phase_name} Phase:")
        print(f"  Price: ${phase_prices[0]:.2f} → ${phase_prices[-1]:.2f}")
        print(f"  Change: {(phase_prices[-1]/phase_prices[0]-1)*100:+.1f}%")
        print(f"  Avg Magnetization: {phase_mags.mean():.3f}")
        print(f"  Avg Return: {returns.mean()*100:.4f}%")
```

**Observations:**
- **Phase 1**: Random walk, M ≈ 0
- **Phase 2**: Super-exponential growth, M increases
- **Phase 3**: Rapid crash, M flips negative

## Exercise 4: Super-Exponential Growth

Demonstrate difference between exponential and super-exponential:

```python
# Run bubble scenario
market = HierarchicalMarket(n_levels=4, feedback_strength=0.08)
prices, _ = market.create_bubble_scenario(n_steps=1000, bubble_start=200, crash_time=800)

# Analyze growth during bubble phase
bubble_phase = (200 <= np.arange(len(prices))) & (np.arange(len(prices)) < 800)
bubble_prices = prices[bubble_phase]
bubble_time = np.arange(len(bubble_prices))

# Fit exponential model: P = A * exp(b*t)
log_prices = np.log(bubble_prices)
poly_coef = np.polyfit(bubble_time, log_prices, 1)  # Linear in log → exponential
exponential_fit = np.exp(poly_coef[1] + poly_coef[0] * bubble_time)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear scale
axes[0].plot(bubble_time, bubble_prices, 'b-', linewidth=2, label='Actual (Super-exponential)')
axes[0].plot(bubble_time, exponential_fit, 'r--', linewidth=2, label='Exponential Fit')
axes[0].set_xlabel("Time (since bubble start)")
axes[0].set_ylabel("Price ($)")
axes[0].set_title("Actual vs Exponential Growth", fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Log scale - exponential would be straight line
axes[1].plot(bubble_time, np.log(bubble_prices), 'b-', linewidth=2, label='Actual')
axes[1].plot(bubble_time, np.log(exponential_fit), 'r--', linewidth=2, label='Exponential')
axes[1].set_xlabel("Time (since bubble start)")
axes[1].set_ylabel("Log Price")
axes[1].set_title("Log Scale: Faster Than Exponential", fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

print("\n**Key insight:** Actual growth curves upward even in log scale")
print("This means: faster than exponential = SUPER-exponential!")
```

## Key Concepts

### 1. Magnetization-Returns Link
```
r(t) = α·M(t) + noise
```
- Direct connection: sentiment → returns
- α controls amplification strength
- Enables positive feedback loops

### 2. Positive Feedback Loop
```
Price ↑ → Sentiment ↑ → Returns ↑ → Price ↑
```
- Self-reinforcing process
- Unstable for large α
- Creates bubbles naturally

### 3. Super-Exponential Growth
**Exponential:** P = P₀ · e^(rt)
- Growth rate r constant
- Log plot is straight line

**Super-exponential:** P ~ (tc-t)^(-m)
- Growth rate accelerates
- Log plot curves upward
- Unsustainable → crash

### 4. Three-Phase Dynamics

**Phase 1 (Normal):**
- Low α, moderate T
- M ≈ 0 (mixed opinions)
- Random walk price

**Phase 2 (Bubble):**
- Increasing α, decreasing T
- M → +1 (herding)
- Super-exponential growth

**Phase 3 (Crash):**
- High T (noise shock)
- M flips to -1
- Rapid price collapse

## Mathematical Details

### Price Evolution Derivation

Starting from discrete returns:
```
P(t+1) / P(t) = exp(r(t))
```

Taking product over time:
```
P(t) = P(0) · ∏ exp(r(τ))
     = P(0) · exp(Σ r(τ))
```

In continuous limit:
```
P(t) = P(0) · exp(∫₀ᵗ r(τ) dτ)
```

Substituting r(t) = α·M(t):
```
P(t) = P(0) · exp(α ∫₀ᵗ M(τ) dτ)
```

**Key insight:** Price = exponential of cumulative sentiment!

### Feedback Amplification

Start with small positive shock:
```
M(0) = 0.1
```

With α = 0.1:
```
r(0) = 0.1 × 0.1 = 0.01 (1% return)
```

Price rises → more agents go bullish:
```
M(1) = 0.2
r(1) = 0.1 × 0.2 = 0.02 (2% return)
```

Accelerating growth:
```
M(2) = 0.4
r(2) = 0.1 × 0.4 = 0.04 (4% return)
```

Eventually unsustainable!

## Challenge Exercise

**Task:** Create custom bubble scenario with:
1. Very gradual bubble formation (400 steps)
2. Multiple mini-crashes during bubble
3. Final major crash

**Hints:**
- Vary α gradually over time
- Add periodic negative shocks
- Use `market.external_field` for shocks

**Solution approach:**
```python
for step in range(n_steps):
    # Custom logic here
    if bubble_phase:
        market.feedback_strength = base_alpha * (1 + progress * 3)
    if mini_crash_condition:
        market.external_field = -0.3
    # ... etc
```

## Key Takeaways

1. **M → r → P**: Clear causal chain from sentiment to price
2. **α is crucial**: Controls bubble magnitude
3. **Super-exponential**: Distinguishes bubbles from normal growth
4. **Phase transitions**: Sharp changes in market regime
5. **Positive feedback**: Creates instability naturally

## Next Steps

**Tutorial 4** will cover:
- LPPL model fitting
- Log-periodic pattern detection
- Predicting crash timing
- Discrete scale invariance

You now understand HOW bubbles form. Next: how to DETECT them!