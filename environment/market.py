"""
Market dynamics and price formation from hierarchical Ising model.

Converts agent states (spins) into market prices through:
- Magnetization → returns
- Cumulative integration → price
- Positive feedback mechanisms
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from agent.hierarchy import HierarchicalNetwork
from tqdm import tqdm


@dataclass
class MarketState:
    """Snapshot of market at a given time."""
    time: float
    price: float
    log_price: float
    magnetization: float
    returns: float
    magnetization_by_level: Dict[int, float]


class HierarchicalMarket:
    """
    Complete market model with hierarchical agents and price dynamics.
    
    Price evolution:
        r(t) = μ + α·M(t) + noise
        P(t) = P(0) · exp(∫ r(τ) dτ)
        
    Where:
        - r(t): returns at time t
        - M(t): magnetization (average agent state)
        - α: feedback strength (positive → herding amplification)
        - μ: drift (fundamental growth)
    """
    
    def __init__(self,
                 n_levels: int = 3,
                 branching_factor: int = 2,
                 lambda_ratio: float = 2.0,
                 n_top_agents: int = 1,
                 initial_price: float = 100.0,
                 feedback_strength: float = 0.1,
                 drift: float = 0.0001,
                 volatility: float = 0.01):
        """
        Initialize hierarchical market.
        
        Args:
            n_levels: Hierarchy depth
            branching_factor: Children per parent
            lambda_ratio: Time scale ratio between levels
            n_top_agents: Number of agents at top level
            initial_price: Starting price
            feedback_strength: α parameter (positive feedback)
            drift: μ parameter (fundamental return)
            volatility: Random noise level
        """
        # Build hierarchical network
        self.network = HierarchicalNetwork(
            n_levels=n_levels,
            branching_factor=branching_factor,
            lambda_ratio=lambda_ratio,
            n_top_agents=n_top_agents
        )
        
        # Market parameters
        self.initial_price = initial_price
        self.feedback_strength = feedback_strength
        self.drift = drift
        self.volatility = volatility
        
        # Current state
        self.current_price = initial_price
        self.current_time = 0.0
        
        # History tracking
        self.history: List[MarketState] = []
        
        # Model parameters (Ising)
        self.J_horizontal = 1.0
        self.J_vertical = 2.0
        self.temperature = 1.0
        self.external_field = 0.0
        
    def compute_returns(self, magnetization: float) -> float:
        """
        Compute returns based on magnetization.
        
        r(t) = μ + α·M(t) + σ·ε(t)
        
        Args:
            magnetization: Current market sentiment M ∈ [-1, 1]
            
        Returns:
            Instantaneous return
        """
        fundamental = self.drift
        feedback = self.feedback_strength * magnetization
        noise = self.volatility * np.random.randn()
        
        return fundamental + feedback + noise
    
    def update_price(self, returns: float, dt: float = 1.0):
        """
        Update price based on returns.
        
        P(t+dt) = P(t) · exp(r·dt)
        
        Args:
            returns: Current return
            dt: Time step
        """
        self.current_price *= np.exp(returns * dt)
        
    def step(self, dt: float = 1.0) -> MarketState:
        """
        Perform one simulation step.
        
        1. Update agents (Ising dynamics)
        2. Compute magnetization
        3. Calculate returns
        4. Update price
        
        Args:
            dt: Time increment
            
        Returns:
            Current market state
        """
        # Update all agents
        for agent in self.network.agents:
            agent.update(
                current_time=self.current_time,
                J_horizontal=self.J_horizontal,
                J_vertical=self.J_vertical,
                temperature=self.temperature,
                external_field=self.external_field
            )
        
        # Compute magnetization (market sentiment)
        magnetization = self.network.population.get_magnetization()
        
        # Magnetization by level (for analysis)
        mag_by_level = {
            level: self.network.population.get_magnetization(level)
            for level in range(self.network.n_levels)
        }
        
        # Compute returns from magnetization
        returns = self.compute_returns(magnetization)
        
        # Update price
        self.update_price(returns, dt)
        
        # Record state
        state = MarketState(
            time=self.current_time,
            price=self.current_price,
            log_price=np.log(self.current_price),
            magnetization=magnetization,
            returns=returns,
            magnetization_by_level=mag_by_level
        )
        self.history.append(state)
        
        # Increment time
        self.current_time += dt
        
        return state
    
    def simulate(self, 
                 n_steps: int = 1000,
                 dt: float = 1.0,
                 show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run complete market simulation.
        
        Args:
            n_steps: Number of time steps
            dt: Time increment per step
            show_progress: Show progress bar
            
        Returns:
            (prices, magnetizations) arrays
        """
        # Reset
        self.current_time = 0.0
        self.current_price = self.initial_price
        self.history = []
        
        # Run simulation
        iterator = tqdm(range(n_steps)) if show_progress else range(n_steps)
        
        for _ in iterator:
            state = self.step(dt)
            
            if show_progress and _ % 100 == 0:
                iterator.set_description(
                    f"Price: {state.price:.2f}, M: {state.magnetization:.3f}"
                )
        
        # Extract arrays
        prices = np.array([s.price for s in self.history])
        magnetizations = np.array([s.magnetization for s in self.history])
        
        return prices, magnetizations
    
    def simulate_with_shock(self,
                           n_steps: int = 1000,
                           shock_time: int = 500,
                           shock_magnitude: float = 0.5,
                           shock_duration: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with external shock (news event, policy change).
        
        Args:
            n_steps: Total steps
            shock_time: When shock occurs
            shock_magnitude: Size of external field shock
            shock_duration: How long shock lasts
            
        Returns:
            (prices, magnetizations) arrays
        """
        self.current_time = 0.0
        self.current_price = self.initial_price
        self.history = []
        
        original_field = self.external_field
        
        for step in tqdm(range(n_steps)):
            # Apply shock
            if shock_time <= step < shock_time + shock_duration:
                self.external_field = original_field + shock_magnitude
            else:
                self.external_field = original_field
                
            self.step()
        
        prices = np.array([s.price for s in self.history])
        magnetizations = np.array([s.magnetization for s in self.history])
        
        return prices, magnetizations
    
    def create_bubble_scenario(self,
                               n_steps: int = 1000,
                               bubble_start: int = 200,
                               crash_time: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create scenario leading to bubble and crash.
        
        Mechanism:
        1. Normal phase: low feedback
        2. Bubble formation: increase feedback (positive feedback amplifies)
        3. Critical point: very high feedback, system unstable
        4. Crash: sudden increase in temperature (noise) triggers collapse
        
        Args:
            n_steps: Total simulation steps
            bubble_start: When bubble formation begins
            crash_time: When crash is triggered (None = auto)
            
        Returns:
            (prices, magnetizations) arrays
        """
        self.current_time = 0.0
        self.current_price = self.initial_price
        self.history = []
        
        # Store original parameters
        original_feedback = self.feedback_strength
        original_temp = self.temperature
        
        if crash_time is None:
            crash_time = int(n_steps * 0.8)  # 80% through
        
        for step in tqdm(range(n_steps)):
            # Phase 1: Normal market
            if step < bubble_start:
                self.feedback_strength = original_feedback
                self.temperature = original_temp
                
            # Phase 2: Bubble formation (increasing positive feedback)
            elif bubble_start <= step < crash_time:
                progress = (step - bubble_start) / (crash_time - bubble_start)
                # Gradually increase feedback
                self.feedback_strength = original_feedback * (1 + 5 * progress)
                # Gradually decrease temperature (less noise → herding)
                self.temperature = original_temp * (1 - 0.5 * progress)
                
            # Phase 3: Crash (sudden noise increase breaks herding)
            else:
                self.feedback_strength = original_feedback
                self.temperature = original_temp * 3.0  # High noise
                self.external_field = -0.5  # Negative shock
                
            self.step()
        
        # Restore parameters
        self.feedback_strength = original_feedback
        self.temperature = original_temp
        self.external_field = 0.0
        
        prices = np.array([s.price for s in self.history])
        magnetizations = np.array([s.magnetization for s in self.history])
        
        return prices, magnetizations
    
    def get_time_series(self) -> Dict[str, np.ndarray]:
        """
        Extract all time series from history.
        
        Returns:
            Dictionary with arrays: time, price, log_price, magnetization, returns
        """
        return {
            'time': np.array([s.time for s in self.history]),
            'price': np.array([s.price for s in self.history]),
            'log_price': np.array([s.log_price for s in self.history]),
            'magnetization': np.array([s.magnetization for s in self.history]),
            'returns': np.array([s.returns for s in self.history]),
        }
    
    def get_magnetization_by_level(self) -> Dict[int, np.ndarray]:
        """
        Extract magnetization time series for each level.
        
        Returns:
            Dictionary mapping level to magnetization array
        """
        result = {level: [] for level in range(self.network.n_levels)}
        
        for state in self.history:
            for level, mag in state.magnetization_by_level.items():
                result[level].append(mag)
                
        return {level: np.array(vals) for level, vals in result.items()}
    
    def print_simulation_summary(self):
        """Print summary statistics from simulation."""
        if not self.history:
            print("No simulation run yet.")
            return
            
        ts = self.get_time_series()
        
        print(f"=== Simulation Summary ===")
        print(f"Duration: {self.current_time:.0f} time units")
        print(f"Steps: {len(self.history)}")
        print(f"\nPrice:")
        print(f"  Initial: ${self.initial_price:.2f}")
        print(f"  Final: ${self.current_price:.2f}")
        print(f"  Change: {(self.current_price/self.initial_price - 1)*100:.1f}%")
        print(f"  Min: ${ts['price'].min():.2f}")
        print(f"  Max: ${ts['price'].max():.2f}")
        print(f"\nReturns:")
        print(f"  Mean: {ts['returns'].mean()*100:.4f}%")
        print(f"  Std: {ts['returns'].std()*100:.4f}%")
        print(f"  Min: {ts['returns'].min()*100:.2f}%")
        print(f"  Max: {ts['returns'].max()*100:.2f}%")
        print(f"\nMagnetization:")
        print(f"  Mean: {ts['magnetization'].mean():.3f}")
        print(f"  Std: {ts['magnetization'].std():.3f}")
        print(f"  Final: {ts['magnetization'][-1]:.3f}")
    
    def get_price_history(self) -> np.ndarray:
        """Get array of historical prices."""
        return np.array([s.price for s in self.history])
    
    def get_magnetization_history(self) -> np.ndarray:
        """Get array of historical magnetizations."""
        return np.array([s.magnetization for s in self.history])
    
    def get_returns_history(self) -> np.ndarray:
        """Get array of historical returns."""
        return np.array([s.returns for s in self.history])
    
    def reset(self):
        """Reset market to initial state."""
        self.current_price = self.initial_price
        self.current_time = 0.0
        self.history = []
        self.external_field = 0.0