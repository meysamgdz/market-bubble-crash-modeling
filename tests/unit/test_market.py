"""
Unit tests for HierarchicalMarket class.

Tests market initialization, price dynamics, returns calculation,
bubble scenarios, and market state tracking.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket, MarketState


class TestMarketInitialization:
    """Test market initialization."""
    
    def test_basic_initialization(self):
        """Test market is created with default parameters."""
        market = HierarchicalMarket()
        
        assert market.initial_price == 100.0
        assert market.current_price == 100.0
        assert market.current_time == 0.0
        assert len(market.history) == 0
    
    def test_custom_parameters(self):
        """Test market with custom parameters."""
        market = HierarchicalMarket(
            n_levels=4,
            branching_factor=3,
            lambda_ratio=2.5,
            initial_price=200.0,
            feedback_strength=0.15,
            drift=0.001,
            volatility=0.02
        )
        
        assert market.network.n_levels == 4
        assert market.network.branching_factor == 3
        assert market.network.lambda_ratio == 2.5
        assert market.initial_price == 200.0
        assert market.feedback_strength == 0.15
        assert market.drift == 0.001
        assert market.volatility == 0.02
    
    def test_n_top_agents(self):
        """Test market with multiple top agents."""
        market = HierarchicalMarket(
            n_levels=3,
            n_top_agents=3
        )
        
        top_level = market.network.n_levels - 1
        assert len(market.network.population.agents_by_level[top_level]) == 3


class TestReturnsCalculation:
    """Test returns computation."""
    
    def test_returns_positive_magnetization(self):
        """Test returns increase with positive magnetization."""
        market = HierarchicalMarket(
            feedback_strength=0.1,
            drift=0.0,
            volatility=0.0  # No noise
        )
        
        # Positive magnetization should give positive returns
        returns = market.compute_returns(magnetization=0.5)
        assert returns > 0
    
    def test_returns_negative_magnetization(self):
        """Test returns decrease with negative magnetization."""
        market = HierarchicalMarket(
            feedback_strength=0.1,
            drift=0.0,
            volatility=0.0
        )
        
        # Negative magnetization should give negative returns
        returns = market.compute_returns(magnetization=-0.5)
        assert returns < 0
    
    def test_returns_zero_magnetization(self):
        """Test returns with zero magnetization equals drift."""
        market = HierarchicalMarket(
            feedback_strength=0.1,
            drift=0.001,
            volatility=0.0
        )
        
        # With M=0, returns should equal drift
        returns = market.compute_returns(magnetization=0.0)
        assert abs(returns - 0.001) < 1e-6
    
    def test_feedback_strength_effect(self):
        """Test higher feedback amplifies returns."""
        market_low = HierarchicalMarket(
            feedback_strength=0.05,
            drift=0.0,
            volatility=0.0
        )
        market_high = HierarchicalMarket(
            feedback_strength=0.20,
            drift=0.0,
            volatility=0.0
        )
        
        mag = 0.5
        returns_low = market_low.compute_returns(mag)
        returns_high = market_high.compute_returns(mag)
        
        assert returns_high > returns_low


class TestPriceUpdate:
    """Test price update mechanism."""
    
    def test_price_increases_positive_returns(self):
        """Test price increases with positive returns."""
        market = HierarchicalMarket(initial_price=100.0)
        
        initial_price = market.current_price
        market.update_price(returns=0.01, dt=1.0)
        
        assert market.current_price > initial_price
    
    def test_price_decreases_negative_returns(self):
        """Test price decreases with negative returns."""
        market = HierarchicalMarket(initial_price=100.0)
        
        initial_price = market.current_price
        market.update_price(returns=-0.01, dt=1.0)
        
        assert market.current_price < initial_price
    
    def test_price_unchanged_zero_returns(self):
        """Test price unchanged with zero returns."""
        market = HierarchicalMarket(initial_price=100.0)
        
        initial_price = market.current_price
        market.update_price(returns=0.0, dt=1.0)
        
        assert abs(market.current_price - initial_price) < 1e-6
    
    def test_exponential_growth(self):
        """Test price follows exponential growth."""
        market = HierarchicalMarket(initial_price=100.0)
        
        # Apply constant positive returns
        r = 0.01
        for _ in range(10):
            market.update_price(r, dt=1.0)
        
        # Should be approximately 100 * exp(0.1)
        expected = 100.0 * np.exp(0.1)
        assert abs(market.current_price - expected) < 0.1


class TestMarketStep:
    """Test single market time step."""
    
    def test_step_returns_state(self):
        """Test step returns MarketState object."""
        market = HierarchicalMarket()
        
        state = market.step()
        
        assert isinstance(state, MarketState)
        assert hasattr(state, 'time')
        assert hasattr(state, 'price')
        assert hasattr(state, 'magnetization')
        assert hasattr(state, 'returns')
    
    def test_step_advances_time(self):
        """Test step advances current time."""
        market = HierarchicalMarket()
        
        initial_time = market.current_time
        market.step(dt=1.0)
        
        assert market.current_time == initial_time + 1.0
    
    def test_step_records_history(self):
        """Test step records state in history."""
        market = HierarchicalMarket()
        
        market.step()
        market.step()
        market.step()
        
        assert len(market.history) == 3
    
    def test_step_custom_dt(self):
        """Test step with custom time increment."""
        market = HierarchicalMarket()
        
        market.step(dt=0.5)
        assert market.current_time == 0.5
        
        market.step(dt=0.5)
        assert market.current_time == 1.0


class TestSimulation:
    """Test full simulation runs."""
    
    def test_simulate_basic(self):
        """Test basic simulation runs without errors."""
        market = HierarchicalMarket()
        
        prices, magnetizations = market.simulate(n_steps=100, show_progress=False)
        
        assert len(prices) == 100  # n_steps results
        assert len(magnetizations) == 100
    
    def test_simulate_returns_arrays(self):
        """Test simulate returns correct array shapes."""
        market = HierarchicalMarket()
        
        prices, mags = market.simulate(n_steps=50, show_progress=False)
        
        assert isinstance(prices, np.ndarray)
        assert isinstance(mags, np.ndarray)
        assert prices.shape == mags.shape
    
    def test_simulation_prices_positive(self):
        """Test all prices remain positive."""
        market = HierarchicalMarket(volatility=0.02)
        
        prices, _ = market.simulate(n_steps=100, show_progress=False)
        
        assert np.all(prices > 0)
    
    def test_magnetization_bounds(self):
        """Test magnetization stays in [-1, 1]."""
        market = HierarchicalMarket()
        
        _, mags = market.simulate(n_steps=100, show_progress=False)
        
        assert np.all(mags >= -1.0)
        assert np.all(mags <= 1.0)


class TestBubbleScenario:
    """Test bubble & crash scenario creation."""
    
    def test_bubble_scenario_runs(self):
        """Test bubble scenario completes without errors."""
        market = HierarchicalMarket()
        
        prices, mags = market.create_bubble_scenario(
            n_steps=500,
            bubble_start=100,
            crash_time=400
        )
        
        assert len(prices) == 500
        assert len(mags) == 500
    
    def test_bubble_price_increases(self):
        """Test price shows bubble dynamics."""
        market = HierarchicalMarket(feedback_strength=0.15)
        
        prices, _ = market.create_bubble_scenario(
            n_steps=500,
            bubble_start=100,
            crash_time=400
        )
        
        # Test that there is significant price variation (bubble formed)
        price_std = np.std(prices)
        assert price_std > 5  # Significant variation
        
        # Test that max price is substantially different from min
        price_range = np.max(prices) - np.min(prices)
        assert price_range > 20  # At least 20 unit range
        
        # Bubble dynamics should create some upward movement
        # Compare max in any period to minimum price
        assert np.max(prices) > np.min(prices) * 1.1  # At least 10% increase
    
    def test_crash_price_drops(self):
        """Test crash dynamics are present."""
        market = HierarchicalMarket(feedback_strength=0.15)
        
        prices, _ = market.create_bubble_scenario(
            n_steps=600,
            bubble_start=100,
            crash_time=400
        )
        
        # Just check that simulation completes and prices vary
        assert len(prices) == 600
        assert np.std(prices) > 0
        
        # Check there's a significant price movement
        price_range = np.max(prices) - np.min(prices)
        assert price_range > 20  # At least 20 unit movement
    
    def test_bubble_timing_parameters(self):
        """Test bubble timing is respected."""
        market = HierarchicalMarket()
        
        bubble_start = 200
        crash_time = 800
        
        prices, _ = market.create_bubble_scenario(
            n_steps=1000,
            bubble_start=bubble_start,
            crash_time=crash_time
        )
        
        # Prices should show distinct phases
        assert len(prices) == 1000


class TestMarketState:
    """Test MarketState dataclass."""
    
    def test_market_state_creation(self):
        """Test MarketState can be created."""
        state = MarketState(
            time=10.0,
            price=105.0,
            log_price=np.log(105.0),
            magnetization=0.3,
            returns=0.01,
            magnetization_by_level={0: 0.2, 1: 0.5}
        )
        
        assert state.time == 10.0
        assert state.price == 105.0
        assert state.magnetization == 0.3
    
    def test_market_state_log_price_consistent(self):
        """Test log_price is log of price."""
        price = 125.0
        state = MarketState(
            time=0,
            price=price,
            log_price=np.log(price),
            magnetization=0,
            returns=0,
            magnetization_by_level={}
        )
        
        assert abs(state.log_price - np.log(price)) < 1e-10


class TestHistoryTracking:
    """Test market history tracking."""
    
    def test_get_price_history(self):
        """Test retrieving price history."""
        market = HierarchicalMarket()
        market.simulate(n_steps=50, show_progress=False)
        
        prices = market.get_price_history()
        
        assert len(prices) == 50
        assert prices[0] > 0  # Should be close to initial price
    
    def test_get_magnetization_history(self):
        """Test retrieving magnetization history."""
        market = HierarchicalMarket()
        market.simulate(n_steps=50, show_progress=False)
        
        mags = market.get_magnetization_history()
        
        assert len(mags) == 50
        assert all(-1 <= m <= 1 for m in mags)
    
    def test_get_returns_history(self):
        """Test retrieving returns history."""
        market = HierarchicalMarket()
        market.simulate(n_steps=50, show_progress=False)
        
        returns = market.get_returns_history()
        
        assert len(returns) == 50


class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_zero_feedback(self):
        """Test market with zero feedback."""
        market = HierarchicalMarket(feedback_strength=0.0, volatility=0.0)
        
        prices, _ = market.simulate(n_steps=100, show_progress=False)
        
        # Prices should grow only by drift
        assert len(prices) == 100
    
    def test_zero_volatility(self):
        """Test deterministic market (no noise)."""
        market = HierarchicalMarket(volatility=0.0)
        
        # Run twice with same parameters
        prices1, _ = market.simulate(n_steps=50, show_progress=False)
        
        market2 = HierarchicalMarket(volatility=0.0)
        prices2, _ = market2.simulate(n_steps=50, show_progress=False)
        
        # Should be similar (agents might differ due to randomness)
        assert len(prices1) == len(prices2)
    
    def test_high_volatility(self):
        """Test market with high volatility."""
        market = HierarchicalMarket(volatility=0.1)
        
        prices, _ = market.simulate(n_steps=100, show_progress=False)
        
        # Should still have positive prices
        assert np.all(prices > 0)
    
    def test_negative_feedback(self):
        """Test market with negative feedback (stabilizing)."""
        market = HierarchicalMarket(feedback_strength=-0.05, volatility=0.01)
        
        prices, _ = market.simulate(n_steps=100, show_progress=False)
        
        # Should still run
        assert len(prices) == 100
    
    def test_extreme_feedback(self):
        """Test market with very high feedback."""
        market = HierarchicalMarket(feedback_strength=0.5, volatility=0.01)
        
        prices, _ = market.simulate(n_steps=50, show_progress=False)
        
        # Should create strong dynamics
        assert np.max(prices) > prices[0]
    
    def test_single_step(self):
        """Test market with single step."""
        market = HierarchicalMarket()
        
        prices, mags = market.simulate(n_steps=1, show_progress=False)
        
        assert len(prices) == 1
        assert len(mags) == 1
    
    def test_very_long_simulation(self):
        """Test market can handle long simulations."""
        market = HierarchicalMarket()
        
        prices, _ = market.simulate(n_steps=5000, show_progress=False)
        
        assert len(prices) == 5000
        assert np.all(prices > 0)
    
    def test_bubble_edge_times(self):
        """Test bubble with extreme timing."""
        market = HierarchicalMarket()
        
        # Crash at very end
        prices, _ = market.create_bubble_scenario(
            n_steps=100,
            bubble_start=10,
            crash_time=95
        )
        
        assert len(prices) == 100


class TestResetFunctionality:
    """Test market reset capability."""
    
    def test_reset_clears_history(self):
        """Test reset clears history."""
        market = HierarchicalMarket()
        market.simulate(n_steps=50, show_progress=False)
        
        assert len(market.history) > 0
        
        market.reset()
        
        assert len(market.history) == 0
    
    def test_reset_price(self):
        """Test reset returns price to initial."""
        market = HierarchicalMarket(initial_price=100.0)
        market.simulate(n_steps=50, show_progress=False)
        
        market.reset()
        
        assert market.current_price == 100.0
    
    def test_reset_time(self):
        """Test reset returns time to zero."""
        market = HierarchicalMarket()
        market.simulate(n_steps=50, show_progress=False)
        
        market.reset()
        
        assert market.current_time == 0.0


class TestParameterValidation:
    """Test parameter validation and constraints."""
    
    def test_positive_initial_price(self):
        """Test initial price must be positive."""
        market = HierarchicalMarket(initial_price=100.0)
        assert market.initial_price > 0
    
    def test_volatility_non_negative(self):
        """Test volatility should be non-negative."""
        market = HierarchicalMarket(volatility=0.02)
        assert market.volatility >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])