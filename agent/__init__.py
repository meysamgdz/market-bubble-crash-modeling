"""
Agent package: Individual agents and hierarchical network structure.
"""

from agent.agent import Agent, AgentPopulation
from agent.hierarchy import HierarchicalNetwork, create_simple_hierarchy

__all__ = [
    'Agent',
    'AgentPopulation', 
    'HierarchicalNetwork',
    'create_simple_hierarchy',
]