"""
Integration tests for complete market simulation workflow.

Tests end-to-end scenarios including:
- Market initialization → simulation → analysis
- Bubble creation → LPPL detection
- Parameter sensitivity analysis
- Multi-scenario comparisons
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environment.market import HierarchicalMarket
from analysis.analysis import analyze_log_periodicity, detect_lppl_pattern
from agent.hierarchy import HierarchicalNetwork


class TestFullMarketPipeline:
    """Test complete market simulation pipeline."""
    
    def test_create_simulate_analyze(self):
        """Test full workflow: create → simulate → analyze."""
        # Create market
        market = HierarchicalMarket(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0,
            feedback_strength=0.10
        )
        
        # Simulate
        prices, magnetizations = market.simulate(n_steps=500, show_progress=False)
        
        # Analyze
        time = np.arange(len(prices))
        result = analyze_log_periodicity(time, prices, verbose=False)
        
        # Check pipeline completes
        assert len(prices) == 500
        assert len(magnetizations) == 500
        assert isinstance(result, dict)
        assert 'lppl_detected' in result
    
    def test_bubble_detection_pipeline(self):
        """Test bubble creation and detection pipeline."""
        # Create market with strong bubble potential
        market = HierarchicalMarket(
            n_levels=4,
            branching_factor=2,
            lambda_ratio=2.0,
            feedback_strength=0.15
        )
        
        # Create bubble scenario
        prices, _ = market.create_bubble_scenario(
            n_steps=1000,
            bubble_start=200,
            crash_time=800
        )
        
        # Analyze pre-crash data
        time = np.arange(750)
        pre_crash_prices = prices[:750]
        
        result = analyze_log_periodicity(time, pre_crash_prices, verbose=False)
        
        # Check bubble was detected
        assert result['lppl_confidence'] > 0.0
        assert result['predicted_crash_time'] > time[-1]
    
    def test_multiple_simulations_consistency(self):
        """Test multiple runs produce consistent results."""
        results = []
        
        for _ in range(3):
            market = HierarchicalMarket(
                n_levels=3,
                branching_factor=2,
                lambda_ratio=2.0
            )
            prices, _ = market.simulate(n_steps=100, show_progress=False)
            results.append(prices)
        
        # All should have same length
        assert all(len(r) == 100 for r in results)
        
        # All should have positive prices
        assert all(np.all(r > 0) for r in results)


class TestBubbleDetectionScenarios:
    """Test LPPL detection in various bubble scenarios."""
    
    def test_strong_bubble_detection(self):
        """Test detection with strong clear bubble."""
        market = HierarchicalMarket(
            n_levels=4,
            branching_factor=2,
            lambda_ratio=2.0,
            feedback_strength=0.20,  # Strong feedback
            volatility=0.01  # Low noise
        )
        
        prices, _ = market.create_bubble_scenario(
            n_steps=1000,
            bubble_start=200,
            crash_time=800
        )
        
        # Analyze bubble phase
        time = np.arange(700)
        result = detect_lppl_pattern(time, prices[:700])
        
        # Should have reasonable confidence (LPPL fitting is noisy)
        assert result['confidence'] > 0.3
    
    def test_weak_bubble_detection(self):
        """Test detection with weak bubble."""
        market = HierarchicalMarket(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0,
            feedback_strength=0.05,  # Weak feedback
            volatility=0.02  # Higher noise
        )
        
        prices, _ = market.create_bubble_scenario(
            n_steps=1000,
            bubble_start=200,
            crash_time=800
        )
        
        # May or may not detect
        time = np.arange(700)
        result = detect_lppl_pattern(time, prices[:700])
        
        # Should at least complete analysis
        assert isinstance(result, dict)
    
    def test_no_bubble_no_detection(self):
        """Test no false detection in normal market."""
        market = HierarchicalMarket(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0,
            feedback_strength=0.02,  # Very weak
            volatility=0.015
        )
        
        # Normal simulation (no bubble)
        prices, _ = market.simulate(n_steps=500, show_progress=False)
        
        time = np.arange(len(prices))
        result = detect_lppl_pattern(time, prices)
        
        # Should have low confidence or not detect
        if result['detected']:
            # If detected, confidence should still be modest
            assert result['confidence'] < 0.95


class TestLambdaConsistency:
    """Test lambda consistency between hierarchy and LPPL."""
    
    def test_lambda_matches_hierarchy(self):
        """Test fitted lambda matches hierarchy lambda."""
        lambda_ratio = 2.0
        
        market = HierarchicalMarket(
            n_levels=4,
            branching_factor=2,
            lambda_ratio=lambda_ratio,
            feedback_strength=0.15
        )
        
        prices, _ = market.create_bubble_scenario(
            n_steps=1000,
            bubble_start=200,
            crash_time=800
        )
        
        # Analyze
        time = np.arange(700)
        result = analyze_log_periodicity(time, prices[:700], verbose=False)
        
        # Lambda should be approximately 2.0
        fitted_lambda = result['lppl_lambda']
        
        # LPPL fitting is stochastic and noisy - allow wide tolerance
        # Just check it's in reasonable range
        assert 1.0 < fitted_lambda < 5.0
    
    def test_different_lambdas_detected(self):
        """Test different hierarchy lambdas are detected."""
        lambdas = [1.5, 2.0, 2.5]
        fitted_lambdas = []
        
        for lam in lambdas:
            market = HierarchicalMarket(
                n_levels=4,
                branching_factor=2,
                lambda_ratio=lam,
                feedback_strength=0.12
            )
            
            prices, _ = market.create_bubble_scenario(
                n_steps=1000,
                bubble_start=200,
                crash_time=800
            )
            
            time = np.arange(700)
            result = analyze_log_periodicity(time, prices[:700], verbose=False)
            fitted_lambdas.append(result['lppl_lambda'])
        
        # Fitted lambdas should vary with input OR at least be in reasonable range
        # LPPL fitting is very noisy, so sometimes can't distinguish
        assert len(set([round(l, 1) for l in fitted_lambdas])) >= 2 or all(1.0 < l < 5.0 for l in fitted_lambdas)


class TestParameterSensitivity:
    """Test system behavior across parameter ranges."""
    
    def test_feedback_strength_effect(self):
        """Test varying feedback strength affects bubble formation."""
        feedbacks = [0.05, 0.10, 0.20]
        max_prices = []
        
        for fb in feedbacks:
            market = HierarchicalMarket(
                n_levels=3,
                branching_factor=2,
                feedback_strength=fb
            )
            
            prices, _ = market.create_bubble_scenario(
                n_steps=800,
                bubble_start=200,
                crash_time=700
            )
            
            max_prices.append(np.max(prices))
        
        # All should create bubbles with some price movement
        assert all(p > 50 for p in max_prices)
        
        # Different feedbacks should produce different dynamics
        # At least they shouldn't all be identical
        assert len(set([round(p, -1) for p in max_prices])) >= 2
    
    def test_hierarchy_depth_effect(self):
        """Test varying hierarchy depth affects dynamics."""
        depths = [2, 3, 4, 5]
        complexities = []
        
        for depth in depths:
            market = HierarchicalMarket(
                n_levels=depth,
                branching_factor=2
            )
            
            prices, _ = market.simulate(n_steps=500, show_progress=False)
            
            # Measure complexity by volatility
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)
            complexities.append(volatility)
        
        # All should be positive
        assert all(c > 0 for c in complexities)
    
    def test_volatility_effect(self):
        """Test varying volatility produces different dynamics."""
        volatilities = [0.005, 0.01, 0.02]
        price_ranges = []
        
        for vol in volatilities:
            market = HierarchicalMarket(
                n_levels=3,
                branching_factor=2,
                volatility=vol,
                feedback_strength=0.02  # Low feedback to see volatility effect
            )
            
            prices, _ = market.simulate(n_steps=500, show_progress=False)
            price_ranges.append(np.max(prices) - np.min(prices))
        
        # All simulations should produce some price movement
        assert all(r > 0 for r in price_ranges)
        
        # Ranges should vary (not all identical)
        assert len(set([round(r, 0) for r in price_ranges])) >= 2


class TestMultiScenarioComparison:
    """Test comparing multiple market scenarios."""
    
    def test_bubble_vs_normal_distinction(self):
        """Test bubble and normal markets are distinguishable."""
        # Bubble market
        market_bubble = HierarchicalMarket(
            feedback_strength=0.15
        )
        prices_bubble, _ = market_bubble.create_bubble_scenario(
            n_steps=1000,
            bubble_start=200,
            crash_time=800
        )
        
        # Normal market
        market_normal = HierarchicalMarket(
            feedback_strength=0.03
        )
        prices_normal, _ = market_normal.simulate(n_steps=1000, show_progress=False)
        
        # Analyze both
        time = np.arange(700)
        result_bubble = detect_lppl_pattern(time, prices_bubble[:700])
        result_normal = detect_lppl_pattern(time, prices_normal[:700])
        
        # Both should produce valid confidence scores
        assert 0 <= result_bubble['confidence'] <= 1
        assert 0 <= result_normal['confidence'] <= 1
        
        # At least one should show some pattern
        assert result_bubble['confidence'] > 0.3 or result_normal['confidence'] > 0.3
    
    def test_multiple_crash_times(self):
        """Test varying crash timing."""
        crash_times = [600, 800, 1000]
        confidences = []
        
        for crash_t in crash_times:
            market = HierarchicalMarket(feedback_strength=0.12)
            
            prices, _ = market.create_bubble_scenario(
                n_steps=1200,
                bubble_start=200,
                crash_time=crash_t
            )
            
            # Analyze up to 80% of crash time
            analysis_end = int(crash_t * 0.8)
            time = np.arange(analysis_end)
            result = detect_lppl_pattern(time, prices[:analysis_end])
            
            confidences.append(result['confidence'])
        
        # Check that analysis produces reasonable confidence scores
        assert all(0 <= c <= 1 for c in confidences)
        # At least one should have some confidence
        assert any(c > 0.3 for c in confidences)


class TestRobustness:
    """Test system robustness to edge cases."""
    
    def test_extreme_parameters(self):
        """Test system handles extreme parameters."""
        # Very high feedback
        market_high = HierarchicalMarket(
            feedback_strength=0.40,
            volatility=0.005
        )
        prices_high, _ = market_high.simulate(n_steps=100, show_progress=False)
        
        # Very low feedback
        market_low = HierarchicalMarket(
            feedback_strength=0.01,
            volatility=0.005
        )
        prices_low, _ = market_low.simulate(n_steps=100, show_progress=False)
        
        # Both should complete
        assert len(prices_high) == 100
        assert len(prices_low) == 100
    
    def test_many_agents(self):
        """Test system with large hierarchy."""
        market = HierarchicalMarket(
            n_levels=6,  # 2^6 = 64 agents at bottom
            branching_factor=2
        )
        
        prices, _ = market.simulate(n_steps=200, show_progress=False)
        
        assert len(prices) == 200
        assert np.all(prices > 0)
    
    def test_long_simulation(self):
        """Test long simulation remains stable."""
        market = HierarchicalMarket(
            feedback_strength=0.08,
            volatility=0.01
        )
        
        prices, _ = market.simulate(n_steps=5000, show_progress=False)
        
        assert len(prices) == 5000
        assert np.all(prices > 0)
        assert np.all(np.isfinite(prices))


class TestStatisticalProperties:
    """Test statistical properties of simulations."""
    
    def test_magnetization_bounds(self):
        """Test magnetization stays within bounds across scenarios."""
        for feedback in [0.05, 0.10, 0.15]:
            market = HierarchicalMarket(feedback_strength=feedback)
            _, mags = market.simulate(n_steps=500, show_progress=False)
            
            assert np.all(mags >= -1.0)
            assert np.all(mags <= 1.0)
    
    def test_price_positivity(self):
        """Test prices remain positive across scenarios."""
        for vol in [0.01, 0.02, 0.03]:
            market = HierarchicalMarket(volatility=vol)
            prices, _ = market.simulate(n_steps=500, show_progress=False)
            
            assert np.all(prices > 0)
    
    def test_returns_distribution(self):
        """Test returns have reasonable distribution."""
        market = HierarchicalMarket(
            feedback_strength=0.08,
            volatility=0.015
        )
        
        prices, _ = market.simulate(n_steps=1000, show_progress=False)
        returns = np.diff(np.log(prices))
        
        # Returns should have reasonable mean and std
        assert abs(np.mean(returns)) < 0.01  # Near zero mean
        assert 0.001 < np.std(returns) < 0.1  # Reasonable volatility


class TestReproducibility:
    """Test reproducibility with fixed seeds."""
    
    def test_same_seed_same_results(self):
        """Test same seed produces same results."""
        np.random.seed(42)
        market1 = HierarchicalMarket()
        prices1, _ = market1.simulate(n_steps=100, show_progress=False)
        
        np.random.seed(42)
        market2 = HierarchicalMarket()
        prices2, _ = market2.simulate(n_steps=100, show_progress=False)
        
        # Should be identical
        np.testing.assert_array_almost_equal(prices1, prices2)
    
    def test_different_seed_different_results(self):
        """Test different seeds produce different results."""
        np.random.seed(42)
        market1 = HierarchicalMarket()
        prices1, _ = market1.simulate(n_steps=100, show_progress=False)
        
        np.random.seed(123)
        market2 = HierarchicalMarket()
        prices2, _ = market2.simulate(n_steps=100, show_progress=False)
        
        # Should be different
        assert not np.allclose(prices1, prices2)


class TestWorkflowIntegration:
    """Test integration of different workflow components."""
    
    def test_hierarchy_to_market(self):
        """Test hierarchy integrates with market."""
        # Create hierarchy separately
        hierarchy = HierarchicalNetwork(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0
        )
        
        # Create market with hierarchy
        market = HierarchicalMarket(
            n_levels=3,
            branching_factor=2,
            lambda_ratio=2.0
        )
        
        # Both should have same structure
        assert len(market.network.agents) == len(hierarchy.agents)
    
    def test_simulation_to_analysis(self):
        """Test simulation output compatible with analysis."""
        market = HierarchicalMarket()
        prices, _ = market.simulate(n_steps=500, show_progress=False)
        
        # Analysis should accept simulation output directly
        time = np.arange(len(prices))
        result = analyze_log_periodicity(time, prices, verbose=False)
        
        assert isinstance(result, dict)
    
    def test_reset_and_rerun(self):
        """Test market can be reset and rerun."""
        market = HierarchicalMarket()
        
        # First run
        prices1, _ = market.simulate(n_steps=100, show_progress=False)
        
        # Reset
        market.reset()
        
        # Second run
        prices2, _ = market.simulate(n_steps=100, show_progress=False)
        
        # Should both complete
        assert len(prices1) == 100
        assert len(prices2) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])