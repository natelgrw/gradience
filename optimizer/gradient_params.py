#!/usr/bin/env python3
"""
gradient_params.py

Author: natelgrw
Last Edited: 11/15/2025

This module provides functions to convert between the 10D gradient
parameters and the 100-point gradient profile used by the ReTiNA_XGB1 
model to predict compound retention times.
"""

import numpy as np
from typing import List, Tuple


# ===== Functions For Gradient Parameterization ===== #


def params_to_gradient(params: np.ndarray, total_time: float) -> List[Tuple[float, float]]:
    """
    Converts 18D input parameters to an input gradient profile for
    ReTiNA_XGB1 retention time model.
    
    Parameterization uses 18 parameters:
    - b_0 to b_9: 10 solvent front B percentage values (0-100) at successive time points
    - t_1 to t_8: 8 time spacing parameters (0-1) controlling intervals between points
    
    Constraints:
    - t_0 is always 0 (start of method)
    - final point is at total_time (end of method)
    - 8 spacing parameters control the 9 intervals between 10 points
    
    Gradient shape: (0, b_0) -> (t1, b_1) -> ... -> (t9, b_9) -> (total_time, b_9)
    
    Args:
        params: array of 18 parameters (10 %B + 8 time spacings)
        total_time: total gradient duration in minutes
        
    Returns:
        list of (time_min, percent_B) tuples
    """
    # extract and clip first 10 parameters (%B values)
    b_values = np.clip(params[:10], 0, 100)
    
    # extract and clip last 8 parameters (time spacings)
    time_spacings = np.clip(params[10:18], 0.01, 1.0)
    
    # normalize spacings to distribute across total_time
    all_spacings = np.concatenate([[time_spacings[0]], time_spacings, [time_spacings[-1]]])
    spacings_norm = all_spacings / np.sum(all_spacings)
    
    # calculate cumulative time points
    time_points = [0.0]
    cumulative_time = 0.0
    
    for spacing_norm in spacings_norm[:-1]:
        cumulative_time += spacing_norm * total_time
        time_points.append(cumulative_time)
    
    time_points[-1] = total_time
    
    # build gradient profile
    gradient = []
    for i in range(10):
        gradient.append((time_points[i], b_values[i]))
    
    return gradient


def gradient_to_params(gradient: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
    """
    convert gradient profile to low-dimensional parameters
    
    Args:
        gradient: list of (time_min, percent_B) tuples
        
    Returns:
        params: array of 18 parameters (10 %B + 8 time spacings)
        total_time: total gradient duration in minutes
    """
    total_time = gradient[-1][0]
    
    # extract %B values
    if len(gradient) >= 10:
        b_values = np.array([gradient[i][1] for i in range(10)])
    else:
        b_values = np.zeros(10)
        n_points = len(gradient)
        
        if n_points >= 10:
            for i in range(10):
                b_values[i] = gradient[i][1]
        else:
            initial_b = gradient[0][1]
            final_b = gradient[-1][1]
            
            for i in range(10):
                frac = i / 9.0
                if i < n_points:
                    b_values[i] = gradient[i][1]
                else:
                    b_values[i] = initial_b + (final_b - initial_b) * frac
    
    # extract time spacings
    times = [gradient[i][0] for i in range(min(10, len(gradient)))]
    
    # calculate spacings between consecutive points
    if len(times) >= 10:
        time_diffs = [times[i+1] - times[i] for i in range(9)]
        # use middle 8 of the 9 intervals
        time_spacings = np.array(time_diffs[1:-1])
    else:
        # distribute evenly
        time_spacings = np.ones(8) / 8.0
    
    # ensure all positive
    time_spacings = np.maximum(time_spacings, 0.01)
    
    # normalize to 0-1 range while preserving relative proportions
    time_spacings = time_spacings / np.max(time_spacings)
    
    params = np.concatenate([b_values, time_spacings])
    
    return params, total_time


def get_bounds(dim: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    """
    get parameter bounds for optimization
    
    Args:
        dim: dimensionality (default: 18)
        
    Returns:
        lower_bounds: array of lower bounds
        upper_bounds: array of upper bounds
    """
    # bounds for %B parameters (first 10)
    b_lower = np.zeros(10)
    b_upper = np.ones(10) * 100.0
    
    # bounds for time spacing parameters (last 8)
    time_lower = np.ones(8) * 0.01
    time_upper = np.ones(8) * 1.0
    
    lower = np.concatenate([b_lower, time_lower])
    upper = np.concatenate([b_upper, time_upper])
    
    return lower[:dim], upper[:dim]


def sample_random_params(n_samples: int = 1, dim: int = 18) -> np.ndarray:
    """
    sample random gradient parameters within bounds
    
    Args:
        n_samples: number of samples to generate
        dim: dimensionality (default: 18)
        
    Returns:
        array of shape (n_samples, dim)
    """
    lower, upper = get_bounds(dim)
    samples = np.random.uniform(lower, upper, size=(n_samples, dim))
    
    return samples

