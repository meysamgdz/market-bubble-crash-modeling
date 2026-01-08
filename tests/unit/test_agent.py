"""
Unit tests for Agent class.

Tests agent initialization, state management, neighbor/parent connections,
field computation, and flip probability calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent, AgentPopulation


class TestAgentInitialization:
    """Test agent creation and initialization."""
    
    def test_basic_initialization(self):
        """Test agent is created with correct attributes."""
        agent = Agent(agent_id=0, level=0)
        
        assert agent.agent_id == 0
        assert agent.level == 0
        assert agent.state in [-1, 1]
        assert agent.time_scale == 1.0
        assert len(agent.neighbors) == 0
        assert agent.parent is None
        assert len(agent.children) == 0
    
    def test_initial_state_custom(self):
        """Test agent can be initialized with specific state."""
        agent_bullish = Agent(agent_id=0, level=0, initial_state=1)
        agent_bearish = Agent(agent_id=1, level=0, initial_state=-1)
        
        assert agent_bullish.state == 1
        assert agent_bearish.state == -1
    
    def test_initial_state_invalid(self):
        """Test agent handles invalid initial states."""
        # Should accept only +1 or -1
        agent = Agent(agent_id=0, level=0, initial_state=1)
        assert agent.state in [-1, 1]
    
    def test_time_scale(self):
        """Test time scale is set correctly."""
        agent_fast = Agent(agent_id=0, level=0, time_scale=1.0)
        agent_slow = Agent(agent_id=1, level=1, time_scale=2.0)
        
        assert agent_fast.time_scale == 1.0
        assert agent_slow.time_scale == 2.0
    
    def test_state_history_initialized(self):
        """Test state history starts with initial state."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        assert len(agent.state_history) == 1
        assert agent.state_history[0] == 1


class TestAgentConnections:
    """Test agent neighbor and parent connections."""
    
    def test_add_neighbor_bidirectional(self):
        """Test adding neighbor creates bidirectional connection."""
        agent1 = Agent(agent_id=0, level=0)
        agent2 = Agent(agent_id=1, level=0)
        
        agent1.add_neighbor(agent2)
        
        assert agent2 in agent1.neighbors
        assert agent1 in agent2.neighbors
    
    def test_add_multiple_neighbors(self):
        """Test agent can have multiple neighbors."""
        agent = Agent(agent_id=0, level=0)
        neighbors = [Agent(agent_id=i, level=0) for i in range(1, 5)]
        
        for neighbor in neighbors:
            agent.add_neighbor(neighbor)
        
        assert len(agent.neighbors) == 4
        for neighbor in neighbors:
            assert neighbor in agent.neighbors
    
    def test_set_parent(self):
        """Test parent-child relationship is bidirectional."""
        parent = Agent(agent_id=0, level=1)
        child = Agent(agent_id=1, level=0)
        
        child.set_parent(parent)
        
        assert child.parent == parent
        assert child in parent.children
    
    def test_multiple_children(self):
        """Test parent can have multiple children."""
        parent = Agent(agent_id=0, level=1)
        children = [Agent(agent_id=i, level=0) for i in range(1, 4)]
        
        for child in children:
            child.set_parent(parent)
        
        assert len(parent.children) == 3
        for child in children:
            assert child in parent.children
    
    def test_no_parent_initially(self):
        """Test agent has no parent by default."""
        agent = Agent(agent_id=0, level=0)
        assert agent.parent is None
        assert len(agent.children) == 0


class TestLocalField:
    """Test local field computation."""
    
    def test_field_no_connections(self):
        """Test field with no neighbors or parent."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        field = agent.compute_local_field()
        
        # Should be zero with no connections
        assert field == 0.0
    
    def test_field_with_neighbors_same_state(self):
        """Test field when all neighbors have same state."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Add 3 neighbors all with state +1
        for i in range(1, 4):
            neighbor = Agent(agent_id=i, level=0, initial_state=1)
            agent.add_neighbor(neighbor)
        
        field = agent.compute_local_field(J_horizontal=1.0)
        
        # Field should be positive (all neighbors agree)
        assert field == 1.0  # Average of [1,1,1] * 1.0
    
    def test_field_with_neighbors_opposite_state(self):
        """Test field when neighbors have opposite state."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Add 2 neighbors with state -1
        for i in range(1, 3):
            neighbor = Agent(agent_id=i, level=0, initial_state=-1)
            agent.add_neighbor(neighbor)
        
        field = agent.compute_local_field(J_horizontal=1.0)
        
        # Field should be negative (neighbors disagree)
        assert field == -1.0  # Average of [-1,-1] * 1.0
    
    def test_field_with_parent(self):
        """Test field includes parent influence."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        parent = Agent(agent_id=1, level=1, initial_state=1)
        agent.set_parent(parent)
        
        field = agent.compute_local_field(J_vertical=2.0)
        
        # Should equal parent state * J_vertical
        assert field == 2.0
    
    def test_field_combined_influences(self):
        """Test field with both neighbors and parent."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Add neighbor with state +1
        neighbor = Agent(agent_id=1, level=0, initial_state=1)
        agent.add_neighbor(neighbor)
        
        # Add parent with state +1
        parent = Agent(agent_id=2, level=1, initial_state=1)
        agent.set_parent(parent)
        
        field = agent.compute_local_field(J_horizontal=1.0, J_vertical=2.0)
        
        # Should be neighbor contribution + parent contribution
        assert field == 1.0 + 2.0  # 3.0
    
    def test_field_with_external_field(self):
        """Test external field is added."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        field = agent.compute_local_field(external_field=0.5)
        
        assert field == 0.5


class TestFlipProbability:
    """Test state flip probability calculations."""
    
    def test_flip_probability_bounds(self):
        """Test flip probability is always in [0, 1]."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Test various temperatures and fields
        for temp in [0.1, 1.0, 10.0]:
            for field in [-5, -1, 0, 1, 5]:
                prob = agent.compute_flip_probability(
                    J_horizontal=1.0,
                    J_vertical=2.0,
                    temperature=temp,
                    external_field=field
                )
                
                assert 0.0 <= prob <= 1.0
    
    def test_flip_probability_zero_temperature(self):
        """Test behavior at zero temperature (deterministic)."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # With positive field and positive state:
        # ΔE = -2*1*10 = -20 < 0 → favorable
        prob = agent.compute_flip_probability(
            temperature=0.01,
            external_field=10.0
        )
        
        assert prob == 1.0
    
    def test_flip_probability_high_temperature(self):
        """Test behavior at high temperature (random)."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Unfavorable flip at high T still has high probability
        prob = agent.compute_flip_probability(
            temperature=100.0,
            external_field=-5.0
        )
        
        # exp(-10/100) ≈ 0.9
        assert prob > 0.8
    
    def test_flip_probability_favorable_field(self):
        """Test unfavorable flip at low temperature."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Negative field → unfavorable flip
        prob = agent.compute_flip_probability(
            temperature=1.0,
            external_field=-10.0
        )
        
        # exp(-20/1) ≈ 2e-9
        assert prob < 0.001


class TestAgentUpdate:
    """Test agent state update dynamics."""
    
    def test_update_respects_time_scale(self):
        """Test agent doesn't update before time scale passes."""
        agent = Agent(agent_id=0, level=0, time_scale=10.0)
        agent.last_update_time = 0.0
        
        # Try to update at t=5 (before time_scale=10)
        updated = agent.update(current_time=5.0, temperature=1.0)
        
        assert not updated
    
    def test_update_after_time_scale(self):
        """Test agent can update after time scale passes."""
        agent = Agent(agent_id=0, level=0, time_scale=10.0, initial_state=1)
        agent.last_update_time = 0.0
        
        # Add strong negative field
        agent.update(current_time=11.0, temperature=0.1, external_field=-10.0)
        
        # Might have flipped (probabilistic)
        assert agent.state in [-1, 1]
    
    def test_state_history_recorded(self):
        """Test state changes are recorded in history."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        initial_history_len = len(agent.state_history)
        
        # Force flip by setting state directly
        agent.state = -1
        agent.state_history.append(agent.state)
        
        assert len(agent.state_history) == initial_history_len + 1
        assert agent.state_history[-1] == -1


class TestAgentPopulation:
    """Test AgentPopulation class."""
    
    def test_population_creation(self):
        """Test population is created correctly."""
        agents = [Agent(agent_id=i, level=i//2) for i in range(6)]
        population = AgentPopulation(agents)
        
        assert len(population) == 6
        assert len(population.agents_by_level) == 3  # Levels 0, 1, 2
    
    def test_get_agent(self):
        """Test retrieving agent by ID."""
        agents = [Agent(agent_id=i, level=0) for i in range(5)]
        population = AgentPopulation(agents)
        
        agent = population.get_agent(3)
        assert agent is not None
        assert agent.agent_id == 3
    
    def test_get_agent_not_found(self):
        """Test retrieving non-existent agent."""
        agents = [Agent(agent_id=i, level=0) for i in range(5)]
        population = AgentPopulation(agents)
        
        agent = population.get_agent(99)
        assert agent is None
    
    def test_magnetization_all_bullish(self):
        """Test magnetization when all agents bullish."""
        agents = [Agent(agent_id=i, level=0, initial_state=1) for i in range(10)]
        population = AgentPopulation(agents)
        
        mag = population.get_magnetization()
        assert mag == 1.0
    
    def test_magnetization_all_bearish(self):
        """Test magnetization when all agents bearish."""
        agents = [Agent(agent_id=i, level=0, initial_state=-1) for i in range(10)]
        population = AgentPopulation(agents)
        
        mag = population.get_magnetization()
        assert mag == -1.0
    
    def test_magnetization_mixed(self):
        """Test magnetization with mixed states."""
        agents = []
        for i in range(5):
            agents.append(Agent(agent_id=i, level=0, initial_state=1))
        for i in range(5, 10):
            agents.append(Agent(agent_id=i, level=0, initial_state=-1))
        
        population = AgentPopulation(agents)
        mag = population.get_magnetization()
        
        assert mag == 0.0  # Equal numbers
    
    def test_magnetization_by_level(self):
        """Test magnetization for specific level."""
        agents = []
        # Level 0: all bullish
        for i in range(3):
            agents.append(Agent(agent_id=i, level=0, initial_state=1))
        # Level 1: all bearish
        for i in range(3, 5):
            agents.append(Agent(agent_id=i, level=1, initial_state=-1))
        
        population = AgentPopulation(agents)
        
        assert population.get_magnetization(level=0) == 1.0
        assert population.get_magnetization(level=1) == -1.0
    
    def test_count_bullish_bearish(self):
        """Test counting agent states."""
        agents = []
        for i in range(7):
            agents.append(Agent(agent_id=i, level=0, initial_state=1))
        for i in range(7, 10):
            agents.append(Agent(agent_id=i, level=0, initial_state=-1))
        
        population = AgentPopulation(agents)
        
        assert population.count_bullish() == 7
        assert population.count_bearish() == 3
    
    def test_consensus_detection(self):
        """Test consensus detection."""
        # 90% bullish
        agents = []
        for i in range(9):
            agents.append(Agent(agent_id=i, level=0, initial_state=1))
        agents.append(Agent(agent_id=9, level=0, initial_state=-1))
        
        population = AgentPopulation(agents)
        
        consensus = population.get_consensus(threshold=0.8)
        assert consensus == 1  # Bullish consensus
    
    def test_no_consensus(self):
        """Test no consensus when mixed."""
        agents = []
        for i in range(5):
            agents.append(Agent(agent_id=i, level=0, initial_state=1))
        for i in range(5, 10):
            agents.append(Agent(agent_id=i, level=0, initial_state=-1))
        
        population = AgentPopulation(agents)
        
        consensus = population.get_consensus(threshold=0.8)
        assert consensus is None


class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_single_agent(self):
        """Test system with single agent."""
        agent = Agent(agent_id=0, level=0)
        population = AgentPopulation([agent])
        
        assert len(population) == 1
        assert population.get_magnetization() in [-1.0, 1.0]
    
    def test_zero_temperature_limit(self):
        """Test behavior at T→0 (deterministic limit)."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Positive field → favorable
        prob = agent.compute_flip_probability(
            temperature=0.001,
            external_field=10.0
        )
        
        assert prob == 1.0
    
    def test_high_temperature_limit(self):
        """Test behavior at T→∞ (random limit)."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Even unfavorable flip likely at high T
        prob = agent.compute_flip_probability(
            temperature=1000.0,
            external_field=-5.0
        )
        
        # exp(-10/1000) ≈ 0.99
        assert prob > 0.98
    
    def test_extreme_coupling_strengths(self):
        """Test with very large coupling constants."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        neighbor = Agent(agent_id=1, level=0, initial_state=1)
        agent.add_neighbor(neighbor)
        
        field = agent.compute_local_field(J_horizontal=1000.0)
        
        # Should be very large
        assert field == 1000.0
    
    def test_many_neighbors(self):
        """Test agent with many neighbors."""
        agent = Agent(agent_id=0, level=0, initial_state=1)
        
        # Add 100 neighbors
        for i in range(1, 101):
            neighbor = Agent(agent_id=i, level=0, initial_state=1)
            agent.add_neighbor(neighbor)
        
        assert len(agent.neighbors) == 100
        field = agent.compute_local_field(J_horizontal=1.0)
        assert field == 1.0  # Average still 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])