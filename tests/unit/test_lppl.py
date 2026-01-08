"""
Unit tests for LPPL analysis module.

Tests LPPL function, fitting algorithms, pattern detection,
scaling ratio computation, and analysis quality metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.analysis import (
    lppl_function, lppl_residuals, fit_lppl,
    detect_lppl_pattern, compute_scaling_ratios,
    analyze_log_periodicity
)


class TestLPPLFunction:
    """Test LPPL mathematical function."""
    
    def test_lppl_basic_evaluation(self):
        """Test LPPL function evaluates without errors."""
        t = np.linspace(0, 0.9, 100)
        tc = 1.0
        A, B, m = 100.0, 10.0, 0.3
        omega, C, phi = 10.0, 0.1, 0.0
        
        result = lppl_function(t, tc, A, B, m, omega, C, phi)
        
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_lppl_tc_boundary(self):
        """Test LPPL handles t approaching tc."""
        t = np.linspace(0, 0.99, 100)
        tc = 1.0
        
        result = lppl_function(t, tc, 100, 10, 0.3, 10, 0.1, 0)
        
        # Should not have infinities
        assert np.all(np.isfinite(result))
    
    def test_lppl_zero_oscillation(self):
        """Test LPPL with C=0 (no oscillation)."""
        t = np.linspace(0, 0.9, 100)
        tc = 1.0
        
        # C=0 means no oscillation, pure power law
        result = lppl_function(t, tc, 100, 10, 0.3, 10, 0.0, 0)
        
        # Should equal A + B*(tc-t)^m
        expected = 100 + 10 * ((tc - t) ** 0.3)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_lppl_symmetric_oscillation(self):
        """Test LPPL oscillation is bounded."""
        t = np.linspace(0, 0.9, 1000)
        tc = 1.0
        
        result = lppl_function(t, tc, 100, 10, 0.3, 10, 0.5, 0)
        
        # Oscillation should be bounded
        # Max should be less than A + B*(1+C)
        # Min should be more than A + B*(1-C)
        assert np.all(result < 130)  # Rough check
        assert np.all(result > 90)


class TestLPPLResiduals:
    """Test LPPL residual computation."""
    
    def test_residuals_perfect_fit(self):
        """Test residuals are zero for perfect fit."""
        t = np.linspace(0, 0.9, 100)
        tc, A, B, m, omega, C, phi = 1.0, 100, 10, 0.3, 10, 0.1, 0
        
        # Generate perfect LPPL data
        price = lppl_function(t, tc, A, B, m, omega, C, phi)
        
        params = [tc, A, B, m, omega, C, phi]
        residuals = lppl_residuals(params, t, price)
        
        # Should be very small (numerical errors only)
        assert residuals < 1e-10
    
    def test_residuals_increase_with_error(self):
        """Test residuals increase when fit is poor."""
        t = np.linspace(0, 0.9, 100)
        true_params = [1.0, 100, 10, 0.3, 10, 0.1, 0]
        wrong_params = [1.0, 100, 20, 0.3, 10, 0.1, 0]  # Wrong B
        
        price = lppl_function(t, *true_params)
        
        residuals_true = lppl_residuals(true_params, t, price)
        residuals_wrong = lppl_residuals(wrong_params, t, price)
        
        assert residuals_wrong > residuals_true
    
    def test_residuals_penalty_invalid_tc(self):
        """Test large penalty for tc <= t_max."""
        t = np.linspace(0, 0.9, 100)
        price = np.ones(100) * 100
        
        # tc before end of data
        params = [0.5, 100, 10, 0.3, 10, 0.1, 0]
        residuals = lppl_residuals(params, t, price)
        
        # Should be large penalty
        assert residuals >= 1e10
    
    def test_residuals_penalty_invalid_m(self):
        """Test penalty for m outside valid range."""
        t = np.linspace(0, 0.9, 100)
        price = np.ones(100) * 100
        
        # m >= 1
        params_high = [1.0, 100, 10, 1.5, 10, 0.1, 0]
        residuals = lppl_residuals(params_high, t, price)
        assert residuals >= 1e10
        
        # m <= 0
        params_low = [1.0, 100, 10, -0.1, 10, 0.1, 0]
        residuals = lppl_residuals(params_low, t, price)
        assert residuals >= 1e10


class TestFitLPPL:
    """Test LPPL fitting algorithm."""
    
    def test_fit_lppl_synthetic_data(self):
        """Test fitting on synthetic LPPL data."""
        # Generate synthetic LPPL data
        t = np.linspace(0, 0.8, 500)
        true_params = {
            'tc': 1.0, 'A': 4.6, 'B': 0.1, 'm': 0.3,
            'omega': 10.0, 'C': 0.1, 'phi': 0.0
        }
        
        # Generate data in log-space
        price = np.exp(lppl_function(t, **true_params))
        
        # Fit
        fitted_params, error = fit_lppl(t, price, method='differential_evolution')
        
        # Check fit quality
        assert error < 1.0  # Should fit well
        assert fitted_params['tc'] > t[-1]
        assert 0.1 < fitted_params['m'] < 0.9
    
    def test_fit_lppl_returns_dict(self):
        """Test fit_lppl returns parameter dictionary."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t) * (1 + 0.1 * np.sin(t/10))
        
        params, error = fit_lppl(t, price)
        
        assert isinstance(params, dict)
        assert 'tc' in params
        assert 'A' in params
        assert 'B' in params
        assert 'm' in params
        assert 'omega' in params
        assert 'C' in params
        assert 'phi' in params
        assert 'lambda' in params
    
    def test_fit_lppl_lambda_calculation(self):
        """Test lambda is calculated correctly."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        params, _ = fit_lppl(t, price)
        
        # Lambda should equal exp(2π/ω)
        expected_lambda = np.exp(2 * np.pi / params['omega'])
        assert abs(params['lambda'] - expected_lambda) < 1e-6


class TestDetectLPPLPattern:
    """Test LPPL pattern detection."""
    
    def test_detect_pattern_returns_dict(self):
        """Test detection returns complete results dict."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        result = detect_lppl_pattern(t, price)
        
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'confidence' in result
        assert 'params' in result
        assert 'predicted_crash_time' in result
    
    def test_detect_reasonable_parameters(self):
        """Test detection enforces reasonable parameter ranges."""
        # Create obviously non-LPPL data (linear)
        t = np.linspace(0, 100, 500)
        price = 100 + t
        
        result = detect_lppl_pattern(t, price)
        
        # Should either not detect or have low confidence
        if result['detected']:
            params = result['params']
            assert 0.1 < params['m'] < 0.9
            assert 5 < params['omega'] < 15
            assert abs(params['C']) < 1
    
    def test_detect_tc_in_future(self):
        """Test detected tc is after data."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t) * (1 + 0.1 * np.sin(t/5))
        
        result = detect_lppl_pattern(t, price)
        
        # tc should be beyond data
        assert result['predicted_crash_time'] > t[-1]
    
    def test_detect_confidence_bounds(self):
        """Test confidence (R²) is in [0, 1]."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        result = detect_lppl_pattern(t, price)
        
        assert 0.0 <= result['confidence'] <= 1.0


class TestComputeScalingRatios:
    """Test discrete scale invariance ratio computation."""
    
    def test_scaling_ratios_no_peaks(self):
        """Test handles data with no peaks."""
        t = np.linspace(0, 100, 500)
        price = np.ones(500) * 100  # Flat line
        
        ratios = compute_scaling_ratios(t, price)
        
        # Should return empty array
        assert len(ratios) == 0
    
    def test_scaling_ratios_few_peaks(self):
        """Test handles data with insufficient peaks."""
        t = np.linspace(0, 100, 100)
        price = 100 + 10 * np.sin(t)  # Only ~3 peaks
        
        ratios = compute_scaling_ratios(t, price, n_ratios=5)
        
        # Should return what it can find or empty
        assert isinstance(ratios, np.ndarray)
    
    def test_scaling_ratios_regular_peaks(self):
        """Test ratios with regularly spaced peaks."""
        t = np.linspace(0, 100, 1000)
        # Regular oscillation - peaks evenly spaced
        price = 100 + 10 * np.sin(2 * np.pi * t / 10)
        
        ratios = compute_scaling_ratios(t, price, n_ratios=3)
        
        if len(ratios) > 0:
            # Ratios should be approximately 1 (equal spacing)
            assert np.all(ratios > 0)
    
    def test_scaling_ratios_log_periodic(self):
        """Test ratios with log-periodic peaks."""
        t = np.linspace(1, 99, 1000)
        # Log-periodic: cos(ω·log(tc-t))
        tc = 100
        omega = 10
        price = 100 + 10 * np.cos(omega * np.log(tc - t))
        
        ratios = compute_scaling_ratios(t, price)
        
        # Ratios should be approximately constant (DSI signature)
        if len(ratios) >= 2:
            ratio_std = np.std(ratios)
            ratio_mean = np.mean(ratios)
            # Coefficient of variation should be small
            cv = ratio_std / ratio_mean if ratio_mean > 0 else np.inf
            assert cv < 1.0  # Allow some variation


class TestAnalyzeLogPeriodicity:
    """Test complete log-periodicity analysis."""
    
    def test_analysis_complete(self):
        """Test analysis returns all required fields."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t) * (1 + 0.1 * np.sin(t/10))
        
        result = analyze_log_periodicity(t, price, verbose=False)
        
        # Check all expected fields present
        assert 'lppl_detected' in result
        assert 'lppl_confidence' in result
        assert 'lppl_params' in result
        assert 'predicted_crash_time' in result
        assert 'time_to_crash' in result
        assert 'lppl_lambda' in result
        assert 'observed_ratios' in result
        assert 'mean_ratio' in result
        assert 'ratio_std' in result
        assert 'dsi_consistent' in result
    
    def test_analysis_detected_boolean(self):
        """Test lppl_detected is boolean."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        result = analyze_log_periodicity(t, price, verbose=False)
        
        assert isinstance(result['lppl_detected'], (bool, np.bool_))
    
    def test_analysis_time_to_crash(self):
        """Test time_to_crash is positive when detected."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t) * (1 + 0.1 * np.sin(t/5))
        
        result = analyze_log_periodicity(t, price, verbose=False)
        
        # Time to crash should be positive
        assert result['time_to_crash'] > 0
    
    def test_analysis_dsi_consistent_boolean(self):
        """Test dsi_consistent is boolean."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        result = analyze_log_periodicity(t, price, verbose=False)
        
        assert isinstance(result['dsi_consistent'], (bool, np.bool_))


class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_minimal_data_points(self):
        """Test with minimal data points."""
        t = np.linspace(0, 10, 50)  # Minimum viable
        price = np.exp(0.01 * t)
        
        # Should not crash
        result = detect_lppl_pattern(t, price)
        assert 'detected' in result
    
    def test_noisy_data(self):
        """Test with very noisy data."""
        t = np.linspace(0, 100, 500)
        signal = np.exp(0.01 * t)
        noise = 0.5 * np.random.randn(500)
        price = signal * np.exp(noise)
        
        # Should handle noise
        result = detect_lppl_pattern(t, price)
        assert isinstance(result, dict)
    
    def test_constant_price(self):
        """Test with constant price."""
        t = np.linspace(0, 100, 500)
        price = np.ones(500) * 100
        
        result = detect_lppl_pattern(t, price)
        
        # Should not detect pattern
        assert result['confidence'] < 0.5
    
    def test_explosive_growth(self):
        """Test with explosive growth."""
        t = np.linspace(0, 5, 500)
        price = np.exp(t)  # Exponential growth
        
        result = detect_lppl_pattern(t, price)
        
        # Should handle extreme values
        assert np.isfinite(result['confidence'])
    
    def test_price_crash_included(self):
        """Test with crash included in data."""
        t = np.linspace(0, 100, 1000)
        # Bubble then crash
        price = np.exp(0.05 * t)
        price[800:] = price[800] * 0.5  # Sudden crash
        
        # Should still attempt fit
        result = detect_lppl_pattern(t, price)
        assert isinstance(result, dict)
    
    def test_negative_prices(self):
        """Test rejects negative prices."""
        t = np.linspace(0, 100, 500)
        price = np.ones(500) * -100  # Invalid!
        
        # Should handle gracefully (likely low confidence)
        try:
            result = detect_lppl_pattern(t, price)
            # If it runs, confidence should be low
            assert result['confidence'] < 0.5
        except:
            # Or it may raise error - both acceptable
            pass
    
    def test_single_peak(self):
        """Test with single peak in data."""
        t = np.linspace(0, 10, 100)
        price = 100 + 10 * np.exp(-(t-5)**2)  # Gaussian bump
        
        ratios = compute_scaling_ratios(t, price)
        
        # Should return empty or very short array
        assert len(ratios) < 3
    
    def test_extreme_lambda(self):
        """Test lambda bounds are reasonable."""
        t = np.linspace(0, 100, 500)
        price = np.exp(0.01 * t)
        
        params, _ = fit_lppl(t, price)
        
        # Lambda should be in reasonable range
        assert 1.0 < params['lambda'] < 10.0
    
    def test_window_size_parameter(self):
        """Test window_size parameter in detection."""
        t = np.linspace(0, 200, 1000)
        price = np.exp(0.01 * t)
        
        # Use only last 500 points
        result = detect_lppl_pattern(t, price, window_size=500)
        
        assert isinstance(result, dict)


class TestNumericalStability:
    """Test numerical stability of algorithms."""
    
    def test_large_time_values(self):
        """Test with large time values."""
        t = np.linspace(1000, 2000, 500)
        price = 100 * np.exp(0.001 * (t - 1000))
        
        result = detect_lppl_pattern(t, price)
        
        assert np.all(np.isfinite([result['confidence']]))
    
    def test_small_price_changes(self):
        """Test with very small price changes."""
        t = np.linspace(0, 100, 500)
        price = 100 + 0.01 * t  # Very gradual
        
        result = detect_lppl_pattern(t, price)
        
        assert isinstance(result, dict)
    
    def test_large_price_values(self):
        """Test with large price values."""
        t = np.linspace(0, 100, 500)
        price = 1e6 * np.exp(0.01 * t)
        
        result = detect_lppl_pattern(t, price)
        
        # Should handle large values via log transform
        assert np.isfinite(result['confidence'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])