"""
Agent class for hierarchical Ising model.

Each agent represents a trader with:
- Binary state (bullish +1 or bearish -1)
- Position in hierarchy (level)
- Connections to neighbors and parent/children
- Update dynamics based on local interactions
"""

import numpy as np
from typing import List, Optional, Set


class Agent:
    """
    Individual agent (trader) in the hierarchical market model.
    
    Attributes:
        agent_id: Unique identifier
        level: Position in hierarchy (0=fastest, higher=slower)
        state: Current opinion (+1 bullish, -1 bearish)
        neighbors: Same-level connections (horizontal)
        parent: Higher-level influencer
        children: Lower-level agents influenced by this agent
        time_scale: Characteristic update time (increases with level)
    """
    
    def __init__(self, 
                 agent_id: int,
                 level: int,
                 initial_state: Optional[int] = None,
                 time_scale: float = 1.0):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique ID
            level: Hierarchy level (0, 1, 2, ...)
            initial_state: +1 or -1, random if None
            time_scale: How often agent updates (larger = slower)
        """
        self.agent_id = agent_id
        self.level = level
        self.state = initial_state if initial_state is not None else np.random.choice([-1, 1])
        self.time_scale = time_scale
        
        # Connections
        self.neighbors: Set[Agent] = set()
        self.parent: Optional[Agent] = None
        self.children: Set[Agent] = set()
        
        # For tracking
        self.state_history: List[int] = [self.state]
        self.last_update_time: float = 0.0
        
    def add_neighbor(self, other: 'Agent'):
        """Add horizontal (same-level) connection."""
        self.neighbors.add(other)
        other.neighbors.add(self)
        
    def set_parent(self, parent: 'Agent'):
        """Set vertical (higher-level) connection."""
        self.parent = parent
        parent.children.add(self)
        
    def compute_local_field(self, 
                           J_horizontal: float = 1.0,
                           J_vertical: float = 2.0,
                           external_field: float = 0.0) -> float:
        """
        Compute local field (effective influence) on this agent.
        
        The local field h_i determines the tendency to flip state.
        
        Args:
            J_horizontal: Interaction strength with same-level neighbors
            J_vertical: Interaction strength with parent (typically > J_horizontal)
            external_field: External influence (news, fundamentals)
            
        Returns:
            Local field h_i = J₀ Σ sⱼ + J₁ S_parent + h_ext
        """
        field = external_field
        
        # Horizontal interactions (peer pressure)
        if self.neighbors:
            neighbor_sum = sum(neighbor.state for neighbor in self.neighbors)
            field += J_horizontal * neighbor_sum / len(self.neighbors)
        
        # Vertical interaction (authority/institution influence)
        if self.parent is not None:
            field += J_vertical * self.parent.state
            
        return field
    
    def compute_flip_probability(self,
                                 J_horizontal: float = 1.0,
                                 J_vertical: float = 2.0,
                                 temperature: float = 1.0,
                                 external_field: float = 0.0) -> float:
        """
        Compute probability of flipping state using Metropolis dynamics.
        
        Based on energy difference ΔE = -2s_i h_i:
        - If flip lowers energy: always flip (P=1)
        - If flip raises energy: flip with P = exp(-ΔE/T)
        
        Args:
            J_horizontal: Horizontal coupling
            J_vertical: Vertical coupling
            temperature: Noise level (higher = more random)
            external_field: External influence
            
        Returns:
            Probability of flipping state (0 to 1)
        """
        h = self.compute_local_field(J_horizontal, J_vertical, external_field)
        
        # Energy difference if we flip
        delta_E = -2 * self.state * h
        
        # Metropolis acceptance probability
        if delta_E <= 0:
            # Flip lowers energy - always accept
            return 1.0
        else:
            # Flip raises energy - accept with Boltzmann probability
            return np.exp(-delta_E / temperature)
    
    def update(self,
              current_time: float,
              J_horizontal: float = 1.0,
              J_vertical: float = 2.0,
              temperature: float = 1.0,
              external_field: float = 0.0) -> bool:
        """
        Attempt to update agent's state based on local interactions.
        
        Agents at higher levels update less frequently (time_scale effect).
        
        Args:
            current_time: Current simulation time
            J_horizontal: Horizontal coupling
            J_vertical: Vertical coupling  
            temperature: Noise level
            external_field: External influence
            
        Returns:
            True if state was updated, False otherwise
        """
        # Check if enough time has passed for this agent's time scale
        if current_time - self.last_update_time < self.time_scale:
            return False
            
        # Compute flip probability
        p_flip = self.compute_flip_probability(
            J_horizontal, J_vertical, temperature, external_field
        )
        
        # Flip with computed probability
        if np.random.random() < p_flip:
            self.state *= -1
            self.state_history.append(self.state)
            self.last_update_time = current_time
            return True
            
        return False
    
    def __repr__(self) -> str:
        state_symbol = "↑" if self.state == 1 else "↓"
        return f"Agent(id={self.agent_id}, level={self.level}, state={state_symbol})"
    
    def get_state_history(self) -> np.ndarray:
        """Return state history as numpy array."""
        return np.array(self.state_history)


class AgentPopulation:
    """
    Collection of agents with utility methods.
    """
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.agents_by_level = self._organize_by_level()
        self.agent_dict = {agent.agent_id: agent for agent in agents}  # Fast lookup
        
    def _organize_by_level(self):
        """Organize agents by hierarchy level."""
        levels = {}
        for agent in self.agents:
            if agent.level not in levels:
                levels[agent.level] = []
            levels[agent.level].append(agent)
        return levels
    
    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """
        Get agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent object or None if not found
        """
        return self.agent_dict.get(agent_id)
    
    def get_magnetization(self, level: Optional[int] = None) -> float:
        """
        Compute magnetization (average state).
        
        M = ⟨s_i⟩ = (1/N) Σ s_i
        
        Args:
            level: Specific level, or None for all agents
            
        Returns:
            Magnetization in [-1, 1]
        """
        if level is not None:
            agents = self.agents_by_level.get(level, [])
        else:
            agents = self.agents
            
        if not agents:
            return 0.0
            
        return np.mean([agent.state for agent in agents])
    
    def get_state_vector(self, level: Optional[int] = None) -> np.ndarray:
        """Get state vector for all agents or specific level."""
        if level is not None:
            agents = self.agents_by_level.get(level, [])
        else:
            agents = self.agents
            
        return np.array([agent.state for agent in agents])
    
    def count_bullish(self, level: Optional[int] = None) -> int:
        """Count number of bullish agents."""
        states = self.get_state_vector(level)
        return np.sum(states == 1)
    
    def count_bearish(self, level: Optional[int] = None) -> int:
        """Count number of bearish agents."""
        states = self.get_state_vector(level)
        return np.sum(states == -1)
    
    def get_consensus(self, threshold: float = 0.8) -> Optional[int]:
        """
        Check if population has reached consensus.
        
        Args:
            threshold: Fraction needed for consensus (default 80%)
            
        Returns:
            +1 if bullish consensus, -1 if bearish, None if no consensus
        """
        mag = abs(self.get_magnetization())
        if mag >= threshold:
            return 1 if self.get_magnetization() > 0 else -1
        return None
    
    def __len__(self) -> int:
        return len(self.agents)
    
    def __repr__(self) -> str:
        n_levels = len(self.agents_by_level)
        mag = self.get_magnetization()
        return f"AgentPopulation(n_agents={len(self)}, n_levels={n_levels}, M={mag:.3f})"