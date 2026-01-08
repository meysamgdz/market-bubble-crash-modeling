"""
Unit tests for HierarchicalNetwork class.

Tests hierarchy construction, connection topologies, network statistics,
and grid/ring lattice structures.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.hierarchy import HierarchicalNetwork


class TestHierarchyConstruction:
    """Test hierarchy construction and basic properties."""
    
    def test_basic_hierarchy(self):
        """Test standard hierarchy is created correctly."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0
        )
        
        assert network.n_levels == 3
        assert network.branching_factor == 2
        assert network.lambda_ratio == 2.0
    
    def test_agent_count_calculation(self):
        """Test correct number of agents created."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        # With branching=2, n_top=1: levels have 1, 2, 4 agents
        expected_total = 1 + 2 + 4
        assert len(network.agents) == expected_total
    
    def test_multiple_top_agents(self):
        """Test hierarchy with multiple top agents."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=3
        )
        
        # Top level should have 3 agents
        top_level = network.n_levels - 1
        assert len(network.population.agents_by_level[top_level]) == 3
        
        # Next level should have 3*2 = 6 agents
        assert len(network.population.agents_by_level[top_level-1]) == 6
    
    def test_time_scale_progression(self):
        """Test time scales increase geometrically."""
        network = HierarchicalNetwork(
            n_levels=4,
            lambda_ratio=2.0,
            base_time_scale=1.0
        )
        
        # Level 0: τ = 1.0
        # Level 1: τ = 2.0
        # Level 2: τ = 4.0
        # Level 3: τ = 8.0
        
        for level in range(4):
            expected_tau = 2.0 ** level
            agents_at_level = network.population.agents_by_level[level]
            
            for agent in agents_at_level:
                assert agent.time_scale == expected_tau
    
    def test_different_branching_factors(self):
        """Test different branching factors."""
        for branching in [2, 3]:
            network = HierarchicalNetwork(
                n_levels=3,
                branching_factor=branching,
                n_top_agents=1
            )
            
            # Check each level has correct number
            # Top level: 1 agent
            # Level n-2: 1 * branching
            # Level n-3: 1 * branching^2
            top_level = network.n_levels - 1
            
            assert len(network.population.agents_by_level[top_level]) == 1
            assert len(network.population.agents_by_level[top_level-1]) == branching
            assert len(network.population.agents_by_level[top_level-2]) == branching**2


class TestVerticalConnections:
    """Test parent-child (vertical) connections."""
    
    def test_parent_child_links(self):
        """Test all agents have correct parent-child links."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        # Check level 0 agents have parents
        level_0 = network.population.agents_by_level[0]
        for agent in level_0:
            assert agent.parent is not None
            assert agent in agent.parent.children
    
    def test_branching_respected(self):
        """Test each parent has correct number of children."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        # Check parents have 2 children each
        for level in range(1, 3):  # Levels with children below
            agents = network.population.agents_by_level[level]
            for agent in agents:
                assert len(agent.children) == 2
    
    def test_top_level_no_parents(self):
        """Test top level agents have no parents."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=2
        )
        
        top_level = network.n_levels - 1
        top_agents = network.population.agents_by_level[top_level]
        
        for agent in top_agents:
            assert agent.parent is None
    
    def test_bottom_level_no_children(self):
        """Test bottom level agents have no children."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2
        )
        
        bottom_agents = network.population.agents_by_level[0]
        
        for agent in bottom_agents:
            assert len(agent.children) == 0


class TestHorizontalConnections:
    """Test same-level (horizontal) connections."""
    
    def test_level_0_grid_structure(self):
        """Test level 0 uses 2D grid lattice."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        level_0 = network.population.agents_by_level[0]
        
        # Check each agent has neighbors
        for agent in level_0:
            assert len(agent.neighbors) > 0
            
            # In grid, should have 2-4 neighbors typically
            assert len(agent.neighbors) <= 4
    
    def test_higher_level_ring_structure(self):
        """Test higher levels use ring lattice."""
        network = HierarchicalNetwork(
            n_levels=5,  # Need 5 levels so level 2 has 4 agents
            branching_factor=2,
            n_top_agents=1
        )
        
        # Level 2 has 4 agents in ring
        level_2 = network.population.agents_by_level[2]
        
        if len(level_2) >= 3:
            # With 3+ agents, ring gives 2 neighbors each
            for agent in level_2:
                assert len(agent.neighbors) == 2
    
    def test_single_agent_level_no_neighbors(self):
        """Test single agent at level has no horizontal neighbors."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        # Top level has 1 agent
        top_level = network.n_levels - 1
        top_agent = network.population.agents_by_level[top_level][0]
        
        assert len(top_agent.neighbors) == 0
    
    def test_connections_are_symmetric(self):
        """Test horizontal connections are bidirectional."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2
        )
        
        for level in range(network.n_levels):
            agents = network.population.agents_by_level[level]
            
            for agent in agents:
                for neighbor in agent.neighbors:
                    # Neighbor should list this agent as neighbor too
                    assert agent in neighbor.neighbors


class TestGridPositions:
    """Test grid positioning for level 0."""
    
    def test_grid_positions_unique(self):
        """Test each agent gets unique grid position."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        positions = network.get_grid_positions()
        
        # All positions should be unique
        position_list = list(positions.values())
        assert len(position_list) == len(set(position_list))
    
    def test_grid_positions_valid_range(self):
        """Test grid positions are in valid range."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        positions = network.get_grid_positions()
        n_agents = len(network.population.agents_by_level[0])
        grid_width = int(np.ceil(np.sqrt(n_agents)))
        
        for pos in positions.values():
            row, col = pos
            assert 0 <= row < grid_width + 1
            assert 0 <= col < grid_width
    
    def test_custom_grid_width(self):
        """Test custom grid width parameter."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        positions = network.get_grid_positions(grid_width=3)
        
        # All columns should be 0, 1, or 2
        for row, col in positions.values():
            assert 0 <= col < 3


class TestNetworkStatistics:
    """Test network statistics and metrics."""
    
    def test_network_stats(self):
        """Test get_network_stats returns correct info."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        stats = network.get_network_stats()
        
        assert stats['n_levels'] == 3
        assert stats['branching_factor'] == 2
        assert stats['lambda_ratio'] == 2.0
        assert stats['n_agents'] == 7  # 1 + 2 + 4
    
    def test_agents_per_level(self):
        """Test agents_per_level is correct."""
        network = HierarchicalNetwork(
            n_levels=4,
            branching_factor=2,
            n_top_agents=1
        )
        
        stats = network.get_network_stats()
        
        # Should be [8, 4, 2, 1]
        assert stats['agents_per_level'][0] == 8
        assert stats['agents_per_level'][1] == 4
        assert stats['agents_per_level'][2] == 2
        assert stats['agents_per_level'][3] == 1
    
    def test_time_scales_in_stats(self):
        """Test time scales are reported correctly."""
        network = HierarchicalNetwork(
            n_levels=3,
            lambda_ratio=2.0,
            base_time_scale=1.0
        )
        
        stats = network.get_network_stats()
        
        assert stats['time_scales'][0] == 1.0
        assert stats['time_scales'][1] == 2.0
        assert stats['time_scales'][2] == 4.0


class TestGraphConstruction:
    """Test NetworkX graph construction."""
    
    def test_graph_created(self):
        """Test NetworkX graph is created."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2
        )
        
        assert network.graph is not None
        assert network.graph.number_of_nodes() == len(network.agents)
    
    def test_vertical_edges(self):
        """Test vertical edges are marked correctly."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        vertical_edges = [
            e for e in network.graph.edges(data=True)
            if e[2].get('edge_type') == 'vertical'
        ]
        
        # Should have 2 + 4 = 6 parent-child connections
        assert len(vertical_edges) == 6
    
    def test_horizontal_edges(self):
        """Test horizontal edges are marked correctly."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        horizontal_edges = [
            e for e in network.graph.edges(data=True)
            if e[2].get('edge_type') == 'horizontal'
        ]
        
        # Should have multiple horizontal connections
        assert len(horizontal_edges) > 0


class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_minimal_hierarchy(self):
        """Test minimum viable hierarchy (2 levels)."""
        network = HierarchicalNetwork(
            n_levels=2,
            branching_factor=2,
            n_top_agents=1
        )
        
        assert network.n_levels == 2
        assert len(network.agents) == 3  # 1 + 2
    
    def test_single_branch(self):
        """Test hierarchy with branching factor 1 (linear)."""
        # This creates a chain rather than tree
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=1,
            n_top_agents=1
        )
        
        # Should have 1 agent per level = 3 total
        assert len(network.agents) == 3
    
    def test_large_branching(self):
        """Test large branching factor."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=3,
            n_top_agents=1
        )
        
        # Should have 1 + 3 + 9 = 13 agents
        assert len(network.agents) == 13
    
    def test_many_levels(self):
        """Test hierarchy with many levels."""
        network = HierarchicalNetwork(
            n_levels=6,
            branching_factor=2,
            n_top_agents=1
        )
        
        # Should have 1+2+4+8+16+32 = 63 agents
        assert len(network.agents) == 63
    
    def test_extreme_lambda(self):
        """Test extreme lambda values."""
        network_low = HierarchicalNetwork(
            n_levels=3,
            lambda_ratio=1.1
        )
        
        network_high = HierarchicalNetwork(
            n_levels=3,
            lambda_ratio=10.0
        )
        
        # Both should construct successfully
        assert network_low.lambda_ratio == 1.1
        assert network_high.lambda_ratio == 10.0
    
    def test_many_top_agents(self):
        """Test hierarchy starting with many top agents."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=5
        )
        
        # Top level: 5, next: 10, bottom: 20
        assert len(network.agents) == 35


class TestVisualization:
    """Test visualization methods don't crash."""
    
    def test_visualize_hierarchy(self):
        """Test 2D hierarchy visualization."""
        import matplotlib.pyplot as plt
        
        network = HierarchicalNetwork(n_levels=3, branching_factor=2)
        
        fig, ax = plt.subplots()
        network.visualize_hierarchy(ax=ax)
        plt.close()
        
        # Should not raise exception
        assert True
    
    def test_visualize_grid(self):
        """Test grid visualization."""
        import matplotlib.pyplot as plt
        
        network = HierarchicalNetwork(n_levels=3, branching_factor=2)
        
        fig, ax = plt.subplots()
        network.visualize_grid_with_network(ax=ax)
        plt.close()
        
        assert True
    
    def test_visualize_3d(self):
        """Test 3D visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        network = HierarchicalNetwork(n_levels=3, branching_factor=2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        network.visualize_3d_hierarchy(ax=ax)
        plt.close()
        
        assert True


class TestConsistency:
    """Test internal consistency of hierarchy."""
    
    def test_all_agents_have_level(self):
        """Test all agents are assigned to a level."""
        network = HierarchicalNetwork(
            n_levels=4,
            branching_factor=2,
            n_top_agents=1
        )
        
        # Count agents in levels
        total_in_levels = sum(
            len(agents) for agents in network.population.agents_by_level.values()
        )
        
        assert total_in_levels == len(network.agents)
    
    def test_no_orphan_agents(self):
        """Test all non-top agents have parents."""
        network = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            n_top_agents=1
        )
        
        for level in range(network.n_levels - 1):  # Exclude top
            agents = network.population.agents_by_level[level]
            for agent in agents:
                assert agent.parent is not None
    
    def test_parent_level_correct(self):
        """Test parents are always one level higher."""
        network = HierarchicalNetwork(
            n_levels=4,
            branching_factor=2,
            n_top_agents=1
        )
        
        for level in range(network.n_levels - 1):
            agents = network.population.agents_by_level[level]
            for agent in agents:
                assert agent.parent.level == level + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])