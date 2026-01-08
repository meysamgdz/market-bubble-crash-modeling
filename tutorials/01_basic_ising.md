# Tutorial 1: Basic Ising Model

**Learning Objectives:**
1. Understand binary agent states (spins)
2. Learn energy minimization and Metropolis dynamics
3. Observe phase transitions and magnetization
4. Explore temperature effects

## Introduction

The Ising model is the foundation of our hierarchical market model. Each agent is like a tiny magnet that can point up (↑ = bullish) or down (↓ = bearish).

## Theory

### Energy Function
The system tries to minimize energy:
```
E = -J Σᵢⱼ sᵢsⱼ - h Σᵢ sᵢ
```

Where:
- `sᵢ ∈ {-1, +1}`: state of agent i
- `J`: coupling strength (how much agents influence each other)
- `h`: external field (like news or fundamentals)

### Magnetization
Market sentiment is the average state:
```
M = (1/N) Σ sᵢ
```
- M = +1: Everyone bullish
- M = 0: Mixed opinions
- M = -1: Everyone bearish

### Temperature
Controls randomness:
- Low T: Agents strongly follow neighbors (herding)
- High T: Agents act more randomly (noise dominates)

## Exercise 1: Single Agent Behavior

```python
import numpy as np
from src.agents import Agent

# Create an agent
agent = Agent(agent_id=0, level=0, initial_state=1)
print(f"Initial state: {agent}")

# Create some neighbors
neighbors = [Agent(i, 0, initial_state=1) for i in range(1, 4)]
for neighbor in neighbors:
    agent.add_neighbor(neighbor)

print(f"Agent has {len(agent.neighbors)} neighbors")
print(f"All neighbors are bullish: {all(n.state == 1 for n in neighbors)}")

# Compute local field
h = agent.compute_local_field(J_horizontal=1.0)
print(f"\nLocal field h = {h:.2f}")
print("Positive field → pressure to stay bullish")

# Try to flip
p_flip = agent.compute_flip_probability(temperature=1.0)
print(f"\nProbability of flipping: {p_flip:.3f}")
print("Low probability because all neighbors are bullish (aligned)")

# Change one neighbor to bearish
neighbors[0].state = -1
h = agent.compute_local_field(J_horizontal=1.0)
p_flip = agent.compute_flip_probability(temperature=1.0)
print(f"\nAfter one neighbor becomes bearish:")
print(f"Local field h = {h:.2f}")
print(f"Probability of flipping: {p_flip:.3f}")
print("Higher probability now due to mixed signals")
```

## Exercise 2: Population Dynamics

```python
from src.agents import AgentPopulation

# Create population
n_agents = 50
agents = [Agent(i, level=0) for i in range(n_agents)]

# Connect as a ring (each agent has 2 neighbors)
for i in range(n_agents):
    agents[i].add_neighbor(agents[(i+1) % n_agents])

population = AgentPopulation(agents)

print(f"Population: {population}")
print(f"Initial magnetization: {population.get_magnetization():.3f}")
print(f"Bullish: {population.count_bullish()}, Bearish: {population.count_bearish()}")

# Simulate dynamics
import matplotlib.pyplot as plt

magnetizations = []
times = []

for t in range(200):
    # Update all agents
    for agent in agents:
        agent.update(
            current_time=float(t),
            J_horizontal=1.0,
            temperature=0.5  # Low temperature → herding
        )
    
    mag = population.get_magnetization()
    magnetizations.append(mag)
    times.append(t)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(times, magnetizations, 'b-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.title('Population Dynamics: Emergence of Consensus')
plt.grid(True, alpha=0.3)
plt.savefig('/home/claude/hierarchical-market-model/notebooks/exercise2_magnetization.png')
print("Plot saved!")

# Final state
final_mag = magnetizations[-1]
if abs(final_mag) > 0.7:
    print(f"\nConsensus reached! M = {final_mag:.3f}")
    if final_mag > 0:
        print("Market is BULLISH (most agents ↑)")
    else:
        print("Market is BEARISH (most agents ↓)")
else:
    print(f"\nNo consensus. M = {final_mag:.3f} (mixed opinions)")
```

## Exercise 3: Phase Transition

Explore how temperature affects consensus formation:

```python
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
final_magnetizations = []

for T in temperatures:
    # Reset population
    agents = [Agent(i, level=0) for i in range(50)]
    for i in range(50):
        agents[i].add_neighbor(agents[(i+1) % 50])
    
    # Run simulation
    for t in range(500):
        for agent in agents:
            agent.update(
                current_time=float(t),
                J_horizontal=1.0,
                temperature=T
            )
    
    # Record final magnetization
    pop = AgentPopulation(agents)
    final_magnetizations.append(abs(pop.get_magnetization()))

# Plot phase transition
plt.figure(figsize=(10, 6))
plt.plot(temperatures, final_magnetizations, 'ro-', linewidth=2, markersize=10)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, 
           label='Partial consensus')
plt.xlabel('Temperature T', fontsize=12)
plt.ylabel('|Magnetization|', fontsize=12)
plt.title('Phase Transition: Order vs Disorder', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/home/claude/hierarchical-market-model/notebooks/exercise3_phase_transition.png')
print("Phase transition plot saved!")

print("\nObservation:")
print(f"Low T ({temperatures[0]}): |M| = {final_magnetizations[0]:.3f} → Strong consensus (ordered)")
print(f"High T ({temperatures[-1]}): |M| = {final_magnetizations[-1]:.3f} → No consensus (disordered)")
```

## Key Takeaways

1. **Agent States**: Binary opinions (bullish/bearish) are like spins
2. **Local Field**: Sum of neighbor influences determines flip tendency
3. **Metropolis Dynamics**: Energy-minimizing updates with thermal noise
4. **Magnetization**: Measures collective sentiment
5. **Phase Transition**: Low temperature → consensus, High temperature → randomness
6. **Herding**: Emerges naturally from local interactions

## Challenge Exercise

Create a scenario where:
1. Start with random states
2. Gradually decrease temperature (simulate increasing herding)
3. Observe spontaneous emergence of consensus
4. Plot how quickly consensus forms at different cooling rates

## Next Steps

In Tutorial 2, we'll add **hierarchy** - multiple levels of agents operating at different time scales. This is where discrete scale invariance emerges!
