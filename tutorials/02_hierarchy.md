# Tutorial 2: Hierarchical Structure & Discrete Scale Invariance

**Learning Objectives:**
1. Understand multi-level agent organization
2. Learn about time scale separation (λ ratio)
3. Observe cross-level influence (vertical coupling)
4. Discover discrete scale invariance emergence

## Introduction

Real markets have structure: day traders, swing traders, hedge funds, central banks - each operating at different time scales. This tutorial builds that hierarchy.

## Theory: The Hierarchy

### Levels and Time Scales

Each level has a characteristic time scale:
```
τₙ = λⁿ · τ₀
```

Where:
- τ₀: Fastest level (day traders, seconds/minutes)
- λ: Scaling ratio (typically 2)
- τₙ: Time scale at level n

**Example with λ=2:**
- Level 0: τ = 1 (updates every step)
- Level 1: τ = 2 (updates every 2 steps)
- Level 2: τ = 4 (updates every 4 steps)
- Level 3: τ = 8 (updates every 8 steps)

### Vertical Coupling

Higher levels influence lower levels:
```
E = -J₀ Σᵢⱼ sᵢsⱼ  -  J₁ Σₖₗ Sₖsₗ
    ^^^^^^^^^^^^^     ^^^^^^^^^^^^^
    horizontal        vertical
    (same level)      (cross-level)
```

Typically: **J₁ > J₀** (authority has more influence than peers)

## Exercise 1: Build Simple Hierarchy

```python
from src.hierarchy import HierarchicalNetwork
import matplotlib.pyplot as plt

# Create 3-level hierarchy
network = HierarchicalNetwork(
    n_levels=3,
    branching_factor=2,
    lambda_ratio=2.0
)

# Inspect structure
network.print_hierarchy_info()

# Visualize
fig, ax = plt.subplots(figsize=(12, 8))
network.visualize_hierarchy(ax=ax, show_states=True)
plt.savefig('/home/claude/hierarchical-market-model/notebooks/hierarchy_structure.png', 
           dpi=150, bbox_inches='tight')
print("\nHierarchy visualization saved!")

# Get statistics
stats = network.get_network_stats()
print(f"\nTotal agents: {stats['n_agents']}")
print("\nAgents per level:")
for level in range(network.n_levels):
    print(f"  Level {level}: {stats['agents_per_level'][level]} agents, τ={stats['time_scales'][level]}")
```

## Exercise 2: Observe Time Scale Separation

Watch how different levels update at different rates:

```python
import numpy as np

# Track state changes by level
updates_by_level = {level: [] for level in range(network.n_levels)}
time_steps = list(range(100))

for t in time_steps:
    # Update all agents
    for agent in network.agents:
        updated = agent.update(
            current_time=float(t),
            J_horizontal=1.0,
            J_vertical=2.0,
            temperature=0.5
        )
        
        if updated:
            updates_by_level[agent.level].append(t)

# Plot update frequency
fig, axes = plt.subplots(network.n_levels, 1, figsize=(12, 8), sharex=True)

for level in range(network.n_levels):
    ax = axes[level]
    updates = updates_by_level[level]
    
    # Create histogram
    hist, bins = np.histogram(updates, bins=20, range=(0, 100))
    ax.bar(bins[:-1], hist, width=bins[1]-bins[0], 
          alpha=0.7, color=f'C{level}')
    
    ax.set_ylabel(f'Level {level}\n(τ={network._compute_time_scale(level):.0f})')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time')
axes[0].set_title('Update Frequency by Level: Time Scale Separation', 
                 fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/hierarchical-market-model/notebooks/time_scale_separation.png',
           dpi=150, bbox_inches='tight')
print("Time scale separation plot saved!")

# Calculate update rates
print("\nUpdate rates (updates per 100 steps):")
for level in range(network.n_levels):
    rate = len(updates_by_level[level])
    expected = 100 / network._compute_time_scale(level)
    print(f"  Level {level}: {rate} updates (expected ~{expected:.0f})")
```

## Exercise 3: Cascading Influence

Flip the top-level agent and watch the cascade:

```python
from src.agents import AgentPopulation

# Create fresh network
network = HierarchicalNetwork(n_levels=4, branching_factor=2, lambda_ratio=2.0)

# Set all agents to bearish initially
for agent in network.agents:
    agent.state = -1

population = AgentPopulation(network.agents)

print("Initial state: All bearish")
print(f"Magnetization: {population.get_magnetization():.3f}")

# Flip top-level agent to bullish
top_level = network.n_levels - 1
top_agent = population.agents_by_level[top_level][0]
top_agent.state = 1
print(f"\nFlipped Level {top_level} agent to BULLISH")

# Track magnetization by level over time
mag_history = {level: [] for level in range(network.n_levels)}
time_history = []

for t in range(200):
    # Update agents
    for agent in network.agents:
        agent.update(
            current_time=float(t),
            J_horizontal=1.0,
            J_vertical=3.0,  # Strong vertical influence
            temperature=0.3   # Low noise
        )
    
    # Record magnetization by level
    for level in range(network.n_levels):
        mag = population.get_magnetization(level)
        mag_history[level].append(mag)
    time_history.append(t)

# Plot cascade
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['red', 'orange', 'green', 'blue']

for level in range(network.n_levels):
    ax.plot(time_history, mag_history[level],
           label=f'Level {level} (τ={network._compute_time_scale(level):.0f})',
           color=colors[level],
           linewidth=2)

ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Magnetization', fontsize=12)
ax.set_title('Hierarchical Cascade: Top-Down Influence', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig('/home/claude/hierarchical-market-model/notebooks/cascade.png',
           dpi=150, bbox_inches='tight')
print("\nCascade plot saved!")

# Analyze cascade timing
print("\nCascade timing:")
for level in range(network.n_levels):
    # Find when magnetization crosses 0 (becomes bullish)
    mags = np.array(mag_history[level])
    cross_times = np.where(np.diff(np.sign(mags)))[0]
    
    if len(cross_times) > 0:
        first_cross = time_history[cross_times[0]]
        print(f"  Level {level}: Turned bullish at t={first_cross}")
    else:
        print(f"  Level {level}: Remained bearish")
```

## Exercise 4: Discrete Scale Invariance

Demonstrate that patterns repeat at geometric time intervals:

```python
# Create network and induce oscillations
network = HierarchicalNetwork(n_levels=4, branching_factor=2, lambda_ratio=2.0)

# External field that oscillates
magnetizations = []
times = []

for t in range(500):
    # Oscillating external field
    h_ext = 0.5 * np.sin(2 * np.pi * t / 50)
    
    # Update agents
    for agent in network.agents:
        agent.update(
            current_time=float(t),
            J_horizontal=1.0,
            J_vertical=2.0,
            temperature=0.5,
            external_field=h_ext
        )
    
    mag = population.get_magnetization()
    magnetizations.append(mag)
    times.append(t)

# Find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(magnetizations, distance=10, prominence=0.1)

if len(peaks) >= 4:
    peak_times = np.array(times)[peaks[-4:]]
    intervals = np.diff(peak_times)
    
    print("Last 4 peaks at times:", peak_times)
    print("Intervals between peaks:", intervals)
    
    if len(intervals) >= 2:
        ratios = intervals[:-1] / intervals[1:]
        print("Ratios of consecutive intervals:", ratios)
        print(f"Mean ratio: {np.mean(ratios):.2f}")
        print(f"Expected λ: {network.lambda_ratio:.2f}")
        
        if np.abs(np.mean(ratios) - network.lambda_ratio) < 0.5:
            print("\n✓ Discrete Scale Invariance detected!")
            print(f"  Pattern repeats with ratio ≈ {network.lambda_ratio}")

# Plot with peaks marked
plt.figure(figsize=(12, 6))
plt.plot(times, magnetizations, 'b-', linewidth=2, alpha=0.7)
plt.plot(np.array(times)[peaks], np.array(magnetizations)[peaks], 
        'ro', markersize=10, label='Peaks')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.title('Discrete Scale Invariance: Geometrically Spaced Oscillations',
         fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/claude/hierarchical-market-model/notebooks/dsi_oscillations.png',
           dpi=150, bbox_inches='tight')
print("\nOscillation plot saved!")
```

## Key Concepts

### 1. Time Scale Hierarchy
```
Level 3:  [____Updates____slowly____]
Level 2:    [__Updates__moderately__]
Level 1:      [Updates_frequently]
Level 0:        [Updates every step]
```

### 2. Vertical Coupling (J₁ > J₀)
- Parent agents have authority over children
- Higher levels set the "trend"
- Lower levels fluctuate around trend

### 3. Discrete Scale Invariance
- Pattern at time t similar to pattern at time λt
- NOT continuous (like fractals)
- DISCRETE preferred ratios (λ, λ², λ³, ...)

### 4. Why λ ≈ 2?
- Binary branching (2 children per parent)
- Binary decisions (buy/sell)
- Powers of 2 are natural organizational structure

## Mathematical Insight

### Renormalization Group
The hierarchy creates effective "renormalization":
```
Level n: Average of children → Effective spin at level n+1
```

This renormalization naturally produces:
- Power law behavior
- Log-periodic corrections
- Scaling ratio λ

### Complex Fractal Dimension
The system has dimension:
```
D = d + i·p
```
Where:
- d: Real part (roughness)
- p: Imaginary part (creates log-periodicity)
- Related by: λ = exp(2π/p)

## Challenge Exercise

1. Create network with λ=3 instead of λ=2
2. Predict what happens to log-periodic frequency
3. Verify with simulation
4. Hint: ω = 2π/log(λ)

## Key Takeaways

1. **Hierarchy**: Multiple time scales organized geometrically
2. **Time scale separation**: τₙ = λⁿ · τ₀
3. **Vertical coupling**: J₁ > J₀ creates top-down influence
4. **Cascades**: Changes propagate down hierarchy
5. **DSI**: Patterns repeat at times t, λt, λ²t, ...
6. **λ ≈ 2**: Natural from binary structure

## Next Steps

Tutorial 3: Connect this hierarchy to market prices and see how log-periodic bubbles emerge!
