"""
Analysis tools for detecting log-periodic patterns and fitting LPPL model.

Implements:
- Log-Periodic Power Law (LPPL) model fitting
- Discrete scale invariance detection
- Critical time prediction
- Pattern visualization
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, Optional
import warnings


def lppl_function(t: np.ndarray, 
                  tc: float, 
                  A: float, 
                  B: float, 
                  m: float,
                  omega: float, 
                  C: float, 
                  phi: float) -> np.ndarray:
    """
    Log-Periodic Power Law (LPPL) function.
    
    p(t) = A + B(tc - t)^m [1 + C·cos(ω·log(tc - t) + φ)]
    
    Args:
        t: Time array
        tc: Critical time (predicted crash)
        A: Price level at crash
        B: Amplitude of power law
        m: Power law exponent (typically 0.2-0.5)
        omega: Log-periodic frequency
        C: Oscillation amplitude
        phi: Phase shift
        
    Returns:
        Predicted prices at time t
    """
    # Ensure tc > t to avoid log of negative
    valid = t < tc
    result = np.zeros_like(t)
    
    if np.any(valid):
        t_valid = t[valid]
        dt = tc - t_valid
        
        # Power law component
        power_law = B * (dt ** m)
        
        # Log-periodic oscillation
        log_periodic = 1 + C * np.cos(omega * np.log(dt) + phi)
        
        result[valid] = A + power_law * log_periodic
        
    return result


def lppl_residuals(params: np.ndarray, 
                   t: np.ndarray, 
                   price: np.ndarray) -> float:
    """
    Compute residuals for LPPL fitting.
    
    Args:
        params: [tc, A, B, m, omega, C, phi]
        t: Time array
        price: Observed prices
        
    Returns:
        Sum of squared residuals
    """
    tc, A, B, m, omega, C, phi = params
    
    # Bounds check
    if tc <= t[-1] or m <= 0 or m >= 1:
        return 1e10  # Large penalty
        
    predicted = lppl_function(t, tc, A, B, m, omega, C, phi)
    
    residuals = price - predicted
    return np.sum(residuals**2)


def fit_lppl(time: np.ndarray,
             price: np.ndarray,
             initial_guess: Optional[Dict] = None,
             method: str = 'differential_evolution') -> Tuple[Dict, float]:
    """
    Fit LPPL model to price data.
    
    Args:
        time: Time array
        price: Price array
        initial_guess: Initial parameter guesses (optional)
        method: 'differential_evolution' (global) or 'minimize' (local)
        
    Returns:
        (params_dict, residual_error)
    """
    # Use log prices for stability
    log_price = np.log(price)
    
    # Normalize time to [0, 1]
    t_min, t_max = time.min(), time.max()
    t_norm = (time - t_min) / (t_max - t_min)
    
    # Parameter bounds
    # tc: slightly beyond end of data
    # A, B: price-related
    # m: typically 0.1 to 0.9
    # omega: typically 5 to 15
    # C: oscillation amplitude -1 to 1
    # phi: phase 0 to 2π
    
    bounds = [
        (1.0, 1.5),              # tc (beyond end)
        (log_price.min(), log_price.max() * 1.2),  # A
        (-10, 10),                # B
        (0.1, 0.9),              # m
        (5, 15),                 # omega
        (-1, 1),                 # C
        (0, 2*np.pi),           # phi
    ]
    
    # Initial guess if not provided
    if initial_guess is None:
        initial_guess = {
            'tc': 1.1,
            'A': log_price[-1],
            'B': 0.1,
            'm': 0.3,
            'omega': 10,
            'C': 0.1,
            'phi': 0,
        }
    
    x0 = [initial_guess[k] for k in ['tc', 'A', 'B', 'm', 'omega', 'C', 'phi']]
    
    # Fit
    if method == 'differential_evolution':
        # Global optimization (better but slower)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                lppl_residuals,
                bounds,
                args=(t_norm, log_price),
                maxiter=1000,
                seed=42,
                atol=1e-6,
                tol=1e-6
            )
    else:
        # Local optimization (faster but may get stuck)
        result = minimize(
            lppl_residuals,
            x0,
            args=(t_norm, log_price),
            method='L-BFGS-B',
            bounds=bounds
        )
    
    # Extract parameters
    tc_norm, A, B, m, omega, C, phi = result.x
    
    # Denormalize tc
    tc_real = tc_norm * (t_max - t_min) + t_min
    
    params = {
        'tc': tc_real,
        'A': A,
        'B': B,
        'm': m,
        'omega': omega,
        'C': C,
        'phi': phi,
        'lambda': np.exp(2 * np.pi / omega)  # Derived scaling ratio
    }
    
    return params, result.fun


def detect_lppl_pattern(time: np.ndarray,
                       price: np.ndarray,
                       window_size: Optional[int] = None) -> Dict:
    """
    Detect if price data shows LPPL pattern.
    
    Args:
        time: Time array
        price: Price array
        window_size: Use last N points (None = all)
        
    Returns:
        Dictionary with detection results and parameters
    """
    if window_size is not None and len(price) > window_size:
        time = time[-window_size:]
        price = price[-window_size:]
    
    # Fit LPPL
    params, error = fit_lppl(time, price)
    
    # Quality metrics
    log_price = np.log(price)
    t_norm = (time - time.min()) / (time.max() - time.min())
    tc_norm = (params['tc'] - time.min()) / (time.max() - time.min())
    
    predicted_log = lppl_function(
        t_norm, tc_norm,
        params['A'], params['B'], params['m'],
        params['omega'], params['C'], params['phi']
    )
    
    # R-squared
    ss_res = np.sum((log_price - predicted_log)**2)
    ss_tot = np.sum((log_price - log_price.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Check if parameters are in reasonable ranges
    reasonable = (
        0.1 < params['m'] < 0.9 and
        5 < params['omega'] < 15 and
        abs(params['C']) < 1 and
        params['tc'] > time[-1]  # Crash in future
    )
    
    # Detect if pattern is present
    pattern_detected = reasonable and r_squared > 0.85
    
    return {
        'detected': pattern_detected,
        'confidence': r_squared,
        'params': params,
        'error': error,
        'predicted_crash_time': params['tc'],
        'time_to_crash': params['tc'] - time[-1],
        'scaling_ratio': params['lambda']
    }


def compute_scaling_ratios(time: np.ndarray,
                          price: np.ndarray,
                          n_ratios: int = 3) -> np.ndarray:
    """
    Compute discrete scale invariance ratios.
    
    Find local maxima/minima and check if their spacing follows
    geometric progression: Δt₁/Δt₂ ≈ Δt₂/Δt₃ ≈ λ
    
    Args:
        time: Time array
        price: Price array
        n_ratios: Number of ratios to compute
        
    Returns:
        Array of scaling ratios
    """
    from scipy.signal import find_peaks
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(price, distance=10)
    
    if len(peaks) < n_ratios + 1:
        return np.array([])
    
    # Take last n+1 peaks
    recent_peaks = peaks[-(n_ratios+1):]
    peak_times = time[recent_peaks]
    
    # Compute time differences
    dt = np.diff(peak_times)
    
    # Compute ratios
    if len(dt) >= 2:
        ratios = dt[:-1] / dt[1:]
        return ratios
    
    return np.array([])


def analyze_log_periodicity(time: np.ndarray,
                           price: np.ndarray,
                           verbose: bool = True) -> Dict:
    """
    Complete log-periodicity analysis.
    
    Args:
        time: Time array
        price: Price array
        verbose: Print results
        
    Returns:
        Analysis results dictionary
    """
    # Fit LPPL
    lppl_result = detect_lppl_pattern(time, price)
    
    # Compute scaling ratios
    ratios = compute_scaling_ratios(time, price)
    
    # Check if ratios are approximately constant (DSI signature)
    if len(ratios) > 0:
        ratio_mean = np.mean(ratios)
        ratio_std = np.std(ratios)
        ratio_consistent = ratio_std / ratio_mean < 0.3 if ratio_mean > 0 else False
    else:
        ratio_mean = np.nan
        ratio_std = np.nan
        ratio_consistent = False
    
    results = {
        'lppl_detected': lppl_result['detected'],
        'lppl_confidence': lppl_result['confidence'],
        'lppl_params': lppl_result['params'],
        'predicted_crash_time': lppl_result['predicted_crash_time'],
        'time_to_crash': lppl_result['time_to_crash'],
        'lppl_lambda': lppl_result['scaling_ratio'],
        'observed_ratios': ratios,
        'mean_ratio': ratio_mean,
        'ratio_std': ratio_std,
        'dsi_consistent': ratio_consistent,
    }
    
    if verbose:
        print("=== Log-Periodicity Analysis ===")
        print(f"LPPL Pattern Detected: {results['lppl_detected']}")
        print(f"Confidence (R²): {results['lppl_confidence']:.3f}")
        
        if results['lppl_detected']:
            print(f"\nPredicted crash time: t={results['predicted_crash_time']:.1f}")
            print(f"Time to crash: {results['time_to_crash']:.1f} time units")
            print(f"\nLPPL Parameters:")
            print(f"  m (power law exponent): {lppl_result['params']['m']:.3f}")
            print(f"  ω (frequency): {lppl_result['params']['omega']:.3f}")
            print(f"  λ (scaling ratio): {lppl_result['params']['lambda']:.3f}")
            print(f"  C (oscillation amplitude): {lppl_result['params']['C']:.3f}")
        
        print(f"\nDiscrete Scale Invariance:")
        if len(ratios) > 0:
            print(f"  Observed ratios: {ratios}")
            print(f"  Mean ratio: {ratio_mean:.3f}")
            print(f"  Std dev: {ratio_std:.3f}")
            print(f"  Consistent: {ratio_consistent}")
        else:
            print(f"  Insufficient peaks for ratio analysis")
    
    return results


def plot_lppl_fit(time: np.ndarray,
                 price: np.ndarray,
                 params: Dict,
                 ax=None):
    """
    Visualize LPPL fit.
    
    Args:
        time: Time array
        price: Price array
        params: LPPL parameters from fitting
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual prices
    ax.plot(time, price, 'b-', label='Actual Price', linewidth=2)
    
    # Plot LPPL fit
    t_norm = (time - time.min()) / (time.max() - time.min())
    tc_norm = (params['tc'] - time.min()) / (time.max() - time.min())
    
    fitted = np.exp(lppl_function(
        t_norm, tc_norm,
        params['A'], params['B'], params['m'],
        params['omega'], params['C'], params['phi']
    ))
    
    ax.plot(time, fitted, 'r--', label='LPPL Fit', linewidth=2, alpha=0.7)
    
    # Mark predicted crash
    if params['tc'] > time[-1]:
        ax.axvline(params['tc'], color='red', linestyle=':', 
                  label=f"Predicted Crash (tc={params['tc']:.1f})",
                  linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Log-Periodic Power Law Fit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add parameter box
    param_text = (f"m = {params['m']:.3f}\n"
                 f"ω = {params['omega']:.2f}\n"
                 f"λ = {params['lambda']:.2f}\n"
                 f"C = {params['C']:.3f}")
    ax.text(0.02, 0.98, param_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9,
           family='monospace')
    
    return ax
