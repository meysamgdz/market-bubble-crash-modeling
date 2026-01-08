"""
Hierarchical network construction for multi-scale agent systems.

Implements tree-like hierarchy where:
- Each level has characteristic time scale
- Higher levels influence lower levels
- Branching factor determines width at each level
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from agent.agent import Agent, AgentPopulation


class HierarchicalNetwork:
    """
    Constructs and manages hierarchical agent network.
    
    Structure:
        Level n:       [Agent]                    (1 agent, slowest)
                      /   |   \\
        Level n-1:  [A]  [A]  [A]                (b agents)
                    /|\\ /|\\ /|\\
        Level n-2: [...........] ...              (b² agents)
        ...
        Level 0:   [...................]          (b^n agents, fastest)
    
    Time scales: τₙ = λ^n · τ₀ (geometric progression)
    """
    
    def __init__(self,
                 n_levels: int = 3,
                 branching_factor: int = 2,
                 base_time_scale: float = 1.0,
                 lambda_ratio: float = 2.0,
                 n_top_agents: int = 1):
        """
        Initialize hierarchical network.
        
        Args:
            n_levels: Number of hierarchy levels (3-5 typical)
            branching_factor: Children per parent (2-3 typical)
            base_time_scale: Time scale for level 0 (fastest agents)
            lambda_ratio: Time scale ratio between levels (typically 2)
            n_top_agents: Number of agents at top level (default 1)
        """
        self.n_levels = n_levels
        self.branching_factor = branching_factor
        self.base_time_scale = base_time_scale
        self.lambda_ratio = lambda_ratio
        self.n_top_agents = n_top_agents
        
        # Build the hierarchy
        self.agents = self._build_hierarchy()
        self.population = AgentPopulation(self.agents)
        
        # Build network graph for visualization
        self.graph = self._build_graph()
        
    def _compute_time_scale(self, level: int) -> float:
        """
        Compute time scale for given level.
        
        τₙ = λ^n · τ₀
        
        Higher levels have longer time scales (update less frequently).
        """
        return self.base_time_scale * (self.lambda_ratio ** level)
    
    def _build_hierarchy(self) -> List[Agent]:
        """
        Build complete hierarchical agent structure.
        
        Returns:
            List of all agents across all levels
        """
        all_agents = []
        agents_by_level = {}
        agent_counter = 0
        
        # Build from top (slowest) to bottom (fastest)
        for level in range(self.n_levels - 1, -1, -1):
            # Top level has n_top_agents, others follow branching pattern
            if level == self.n_levels - 1:
                n_agents_this_level = self.n_top_agents
            else:
                # Each parent at level+1 has branching_factor children
                n_agents_this_level = len(agents_by_level[level + 1]) * self.branching_factor
            
            time_scale = self._compute_time_scale(level)
            
            level_agents = []
            for i in range(n_agents_this_level):
                agent = Agent(
                    agent_id=agent_counter,
                    level=level,
                    time_scale=time_scale
                )
                level_agents.append(agent)
                all_agents.append(agent)
                agent_counter += 1
                
            agents_by_level[level] = level_agents
            
        # Connect hierarchy (parent-child relationships)
        for level in range(self.n_levels - 1):
            parents = agents_by_level[level + 1]
            children = agents_by_level[level]
            
            # Each parent gets branching_factor children
            for i, parent in enumerate(parents):
                start_idx = i * self.branching_factor
                end_idx = start_idx + self.branching_factor
                for child in children[start_idx:end_idx]:
                    child.set_parent(parent)
                    
        # Add horizontal connections (within level)
        self._add_horizontal_connections(agents_by_level)
        
        return all_agents
    
    def _add_horizontal_connections(self, 
                                   agents_by_level: Dict[int, List[Agent]]):
        """
        Add horizontal connections within each level.
        
        Level 0: 2D grid lattice (4 neighbors: up, down, left, right)
        Higher levels: Ring lattice (following Sornette's model)
        
        Args:
            agents_by_level: Dictionary mapping level to agents
        """
        for level, agents in agents_by_level.items():
            n_agents = len(agents)
            
            if n_agents <= 1:
                continue
            
            if level == 0:
                # Level 0: 2D GRID LATTICE
                self._connect_grid_lattice(agents)
            else:
                # Higher levels: RING LATTICE (book's approach)
                self._connect_ring_lattice(agents)
    
    def _connect_grid_lattice(self, agents: List[Agent]):
        """
        Connect agents in 2D grid with periodic boundary conditions.
        
        Each agent has up to 4 neighbors (north, south, east, west).
        Uses periodic boundaries (wraps around edges).
        
        Args:
            agents: List of agents at level 0
        """
        n_agents = len(agents)
        grid_width = int(np.ceil(np.sqrt(n_agents)))
        grid_height = int(np.ceil(n_agents / grid_width))
        
        # Create grid mapping
        grid = {}
        for i, agent in enumerate(agents):
            row = i // grid_width
            col = i % grid_width
            grid[(row, col)] = agent
        
        # Connect neighbors (4-connected grid)
        for i, agent in enumerate(agents):
            row = i // grid_width
            col = i % grid_width
            
            # North (with periodic boundary)
            north_row = (row - 1) % grid_height
            if (north_row, col) in grid:
                agent.add_neighbor(grid[(north_row, col)])
            
            # South (with periodic boundary)
            south_row = (row + 1) % grid_height
            if (south_row, col) in grid:
                agent.add_neighbor(grid[(south_row, col)])
            
            # East (with periodic boundary)
            east_col = (col + 1) % grid_width
            if (row, east_col) in grid:
                agent.add_neighbor(grid[(row, east_col)])
            
            # West (with periodic boundary)
            west_col = (col - 1) % grid_width
            if (row, west_col) in grid:
                agent.add_neighbor(grid[(row, west_col)])
    
    def _connect_ring_lattice(self, agents: List[Agent]):
        """
        Connect agents in ring lattice (Sornette's approach for higher levels).
        
        Each agent connects to immediate neighbors in circular arrangement.
        This creates a 1D periodic lattice.
        
        Args:
            agents: List of agents at this level
        """
        n_agents = len(agents)
        
        # Connect to neighbors in a ring
        for i, agent in enumerate(agents):
            # Connect to next neighbor (forward)
            next_agent = agents[(i + 1) % n_agents]
            agent.add_neighbor(next_agent)
            
            # Connect to previous neighbor (backward) 
            # This ensures bidirectional connections
            prev_agent = agents[(i - 1) % n_agents]
            agent.add_neighbor(prev_agent)
    
    def _build_graph(self) -> nx.DiGraph:
        """
        Build NetworkX graph for visualization.
        
        Returns:
            Directed graph with hierarchical structure
        """
        G = nx.DiGraph()
        
        # Add nodes
        for agent in self.agents:
            G.add_node(agent.agent_id, 
                      level=agent.level,
                      state=agent.state)
        
        # Add edges
        for agent in self.agents:
            # Parent-child edges (vertical)
            if agent.parent:
                G.add_edge(agent.parent.agent_id, agent.agent_id, 
                          edge_type='vertical')
            
            # Neighbor edges (horizontal)
            for neighbor in agent.neighbors:
                if agent.agent_id < neighbor.agent_id:  # Avoid duplicates
                    G.add_edge(agent.agent_id, neighbor.agent_id,
                              edge_type='horizontal')
                    
        return G
    
    def get_network_stats(self) -> Dict:
        """
        Compute network statistics.
        
        Returns:
            Dictionary with network properties
        """
        stats = {
            'n_agents': len(self.agents),
            'n_levels': self.n_levels,
            'branching_factor': self.branching_factor,
            'lambda_ratio': self.lambda_ratio,
            'agents_per_level': {},
            'time_scales': {},
            'avg_degree': {},
        }
        
        for level in range(self.n_levels):
            agents = self.population.agents_by_level[level]
            stats['agents_per_level'][level] = len(agents)
            stats['time_scales'][level] = self._compute_time_scale(level)
            
            # Average degree (connections)
            degrees = [len(agent.neighbors) for agent in agents]
            stats['avg_degree'][level] = np.mean(degrees) if degrees else 0
            
        return stats
    
    def visualize_hierarchy(self, ax=None, show_states=True):
        """
        Visualize hierarchical structure.
        
        Args:
            ax: Matplotlib axis (creates new if None)
            show_states: Color nodes by state
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
        # Compute positions (hierarchical layout)
        pos = self._compute_hierarchical_positions()
        
        # Node colors based on state
        if show_states:
            node_colors = ['red' if self.agents[i].state == -1 else 'green' 
                          for i in range(len(self.agents))]
        else:
            node_colors = 'lightblue'
            
        # Node sizes based on level (higher = larger)
        node_sizes = [200 * (2 ** self.agents[i].level) 
                     for i in range(len(self.agents))]
        
        # Draw
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.7,
                              ax=ax)
        
        # Draw vertical edges (parent-child) as solid
        vertical_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                         if d.get('edge_type') == 'vertical']
        nx.draw_networkx_edges(self.graph, pos,
                              edgelist=vertical_edges,
                              edge_color='black',
                              arrows=True,
                              ax=ax)
        
        # Draw horizontal edges (neighbors) as dashed
        horizontal_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                           if d.get('edge_type') == 'horizontal']
        nx.draw_networkx_edges(self.graph, pos,
                              edgelist=horizontal_edges,
                              edge_color='gray',
                              style='dashed',
                              arrows=False,
                              alpha=0.5,
                              ax=ax)
        
        ax.set_title(f"Hierarchical Network: {self.n_levels} levels, "
                    f"branching factor {self.branching_factor}")
        ax.axis('off')
        
        if show_states:
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Bullish (↑)'),
                Patch(facecolor='red', label='Bearish (↓)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
        return ax
    
    def _compute_hierarchical_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute node positions for hierarchical layout.
        
        Returns:
            Dictionary mapping agent_id to (x, y) position
        """
        pos = {}
        
        for level in range(self.n_levels):
            agents = self.population.agents_by_level[level]
            n_agents = len(agents)
            
            # Y position based on level
            y = level
            
            # X positions spread across width
            for i, agent in enumerate(agents):
                x = (i + 0.5) / n_agents - 0.5  # Center around 0
                pos[agent.agent_id] = (x, y)
                
        return pos
    
    def get_grid_positions(self, grid_width: int = None) -> Dict[int, Tuple[int, int]]:
        """
        Get grid positions for level 0 agents.
        
        Args:
            grid_width: Width of grid (auto-computed if None)
            
        Returns:
            Dictionary mapping agent_id to (row, col) grid position
        """
        level_0_agents = self.population.agents_by_level[0]
        n_agents = len(level_0_agents)
        
        if grid_width is None:
            grid_width = int(np.ceil(np.sqrt(n_agents)))
        
        positions = {}
        for i, agent in enumerate(level_0_agents):
            row = i // grid_width
            col = i % grid_width
            positions[agent.agent_id] = (row, col)
        
        return positions
    
    def visualize_3d_hierarchy(self, ax=None, show_states=True):
        """
        Create 3D visualization of hierarchy with grid at bottom.
        
        Args:
            ax: matplotlib 3D axis (created if None)
            show_states: Color nodes by agent state
            
        Returns:
            The axis object
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get grid positions for level 0
        grid_pos = self.get_grid_positions()
        
        # Plot each level
        for level in range(self.n_levels):
            agents = self.population.agents_by_level[level]
            n_agents = len(agents)
            
            if level == 0:
                # Bottom level on grid
                for agent in agents:
                    row, col = grid_pos[agent.agent_id]
                    
                    if show_states:
                        color = 'green' if agent.state == 1 else 'red'
                    else:
                        color = 'blue'
                    
                    ax.scatter(col, row, level, c=color, s=100, alpha=0.8, 
                             edgecolors='black', linewidth=1)
            else:
                # Higher levels spread out
                for i, agent in enumerate(agents):
                    x = (i + 0.5) / n_agents * 10 - 5  # Spread across
                    y = 5  # Fixed y
                    
                    if show_states:
                        color = 'green' if agent.state == 1 else 'red'
                    else:
                        color = 'blue'
                    
                    ax.scatter(x, y, level, c=color, s=200, alpha=0.8,
                             edgecolors='black', linewidth=1.5)
        
        # Draw connections
        for u, v, data in self.graph.edges(data=True):
            agent_u = self.population.get_agent(u)
            agent_v = self.population.get_agent(v)
            
            # Get positions
            if agent_u.level == 0:
                x1, y1 = grid_pos[u]
            else:
                agents_at_level = self.population.agents_by_level[agent_u.level]
                idx = agents_at_level.index(agent_u)
                x1 = (idx + 0.5) / len(agents_at_level) * 10 - 5
                y1 = 5
            
            if agent_v.level == 0:
                x2, y2 = grid_pos[v]
            else:
                agents_at_level = self.population.agents_by_level[agent_v.level]
                idx = agents_at_level.index(agent_v)
                x2 = (idx + 0.5) / len(agents_at_level) * 10 - 5
                y2 = 5
            
            z1, z2 = agent_u.level, agent_v.level
            
            if data.get('edge_type') == 'vertical':
                ax.plot([x1, x2], [y1, y2], [z1, z2], 
                       'k-', alpha=0.3, linewidth=1)
            else:
                ax.plot([x1, x2], [y1, y2], [z1, z2], 
                       'gray', alpha=0.2, linewidth=0.5, linestyle='--')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Level')
        ax.set_title(f'3D Hierarchy: {self.n_levels} levels')
        
        return ax
    
    def visualize_grid_with_network(self, ax=None, show_states=True):
        """
        Visualize level 0 agents on grid showing clustering.
        
        Args:
            ax: matplotlib axis (created if None)
            show_states: Color by agent state
            
        Returns:
            The axis object
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        grid_pos = self.get_grid_positions()
        level_0_agents = self.population.agents_by_level[0]
        
        # Create grid visualization
        grid_width = int(np.ceil(np.sqrt(len(level_0_agents))))
        grid_height = int(np.ceil(len(level_0_agents) / grid_width))
        
        for agent in level_0_agents:
            row, col = grid_pos[agent.agent_id]
            
            if show_states:
                color = 'green' if agent.state == 1 else 'red'
            else:
                color = 'lightblue'
            
            # Draw square
            rect = plt.Rectangle((col, row), 1, 1, 
                                facecolor=color, 
                                edgecolor='black', 
                                linewidth=1)
            ax.add_patch(rect)
        
        # Draw horizontal connections between level 0 agents
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'horizontal':
                agent_u = self.population.get_agent(u)
                agent_v = self.population.get_agent(v)
                
                if agent_u.level == 0 and agent_v.level == 0:
                    r1, c1 = grid_pos[u]
                    r2, c2 = grid_pos[v]
                    
                    ax.plot([c1+0.5, c2+0.5], [r1+0.5, r2+0.5], 
                           'blue', alpha=0.3, linewidth=2)
        
        ax.set_xlim(0, grid_width)
        ax.set_ylim(0, grid_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Row 0 at top
        ax.set_title('Level 0 Agent Grid (Clustering Visible)')
        ax.axis('off')
        
        if show_states:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Bullish (↑)'),
                Patch(facecolor='red', label='Bearish (↓)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        return ax
    
    def print_hierarchy_info(self):
        """Print detailed hierarchy information."""
        stats = self.get_network_stats()
        
        print(f"=== Hierarchical Network ===")
        print(f"Total agents: {stats['n_agents']}")
        print(f"Number of levels: {stats['n_levels']}")
        print(f"Branching factor: {stats['branching_factor']}")
        print(f"Lambda (time scale ratio): {stats['lambda_ratio']}")
        print(f"\nLevel structure:")
        
        for level in range(self.n_levels - 1, -1, -1):
            n_agents = stats['agents_per_level'][level]
            time_scale = stats['time_scales'][level]
            avg_deg = stats['avg_degree'][level]
            
            # Connection type
            if level == 0:
                conn_type = "2D Grid Lattice (4 neighbors)"
            else:
                conn_type = "Ring Lattice (2 neighbors)"
            
            print(f"  Level {level}: {n_agents} agents, "
                  f"τ={time_scale:.2f}, "
                  f"avg_degree={avg_deg:.1f}, "
                  f"{conn_type}")
                  
        print(f"\nTotal connections:")
        print(f"  Vertical (parent-child): {len([e for e in self.graph.edges(data=True) if e[2].get('edge_type')=='vertical'])}")
        print(f"  Horizontal (same-level): {len([e for e in self.graph.edges(data=True) if e[2].get('edge_type')=='horizontal'])}")
        
        # Show level 0 grid dimensions
        level_0_agents = stats['agents_per_level'][0]
        grid_width = int(np.ceil(np.sqrt(level_0_agents)))
        grid_height = int(np.ceil(level_0_agents / grid_width))
        print(f"\nLevel 0 grid: {grid_width} × {grid_height} (periodic boundaries)")


def create_simple_hierarchy(n_levels: int = 3) -> HierarchicalNetwork:
    """
    Create standard hierarchical network with default parameters.
    
    Args:
        n_levels: Number of levels (default 3)
        
    Returns:
        HierarchicalNetwork instance
    """
    return HierarchicalNetwork(
        n_levels=n_levels,
        branching_factor=2,
        base_time_scale=1.0,
        lambda_ratio=2.0
    )