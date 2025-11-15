"""
objective.py

Author: natelgrw
Last Edited: 11/15/2025

Contains the objective function for gradient optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rt_pred.pred_rt import predict_retention_time_from_list
from optimizer.gradient_params import params_to_gradient


# ===== Functions For Objective Function ===== #


def compute_separation_score(retention_times: List[float], 
                             probabilities: List[float],
                             method_length: float) -> float:
    """
    Computes separation quality score from predicted retention times.
    
    The objective is to maximize spacing between compounds weighted by their
    probabilities. compounds with higher probability should be well separated
    
    Args:
        retention_times: predicted RTs in seconds
        probabilities: compound probabilities from askcos
        method_length: total method length in seconds
        
    Returns:
        separation score (higher is better)
    """
    if len(retention_times) < 2:
        return 0.0
    
    # convert to numpy arrays
    rts = np.array(retention_times)
    probs = np.array(probabilities)
    
    # normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones_like(probs) / len(probs)
    
    # sort by retention time
    sort_idx = np.argsort(rts)
    rts_sorted = rts[sort_idx]
    probs_sorted = probs[sort_idx]
    
    # compute pairwise separations weighted by probability product
    score = 0.0
    for i in range(len(rts_sorted)):
        for j in range(i + 1, len(rts_sorted)):
            separation = rts_sorted[j] - rts_sorted[i]
            weight = probs_sorted[i] * probs_sorted[j]
            
            if separation > 0:
                # more sensitive scoring: reward separations more strongly
                # use square root instead of log for better sensitivity
                sep_score = np.sqrt(separation / 5.0)
                score += weight * sep_score
    
    # penalty for compounds eluting too late or too early
    penalty = 0.0
    for i, rt in enumerate(rts):
        if rt > method_length:
            penalty += probs[i] * (rt - method_length) / method_length
        elif rt < 0:
            penalty += probs[i] * (60 - rt) / 60
    
    score -= penalty * 10.0
    
    # normalize by square root of pairs instead of pairs
    n_pairs = len(rts_sorted) * (len(rts_sorted) - 1) / 2
    if n_pairs > 0:
        score = score / np.sqrt(n_pairs)
    
    return score


def evaluate_gradient(params: np.ndarray,
                     compounds: List[Dict],
                     lcms_config: Dict) -> float:
    """
    Evaluates a gradient parameterization.
    
    Args:
        params: low-dimensional gradient parameters
        compounds: list of dicts with 'smiles' and 'probability' keys
        lcms_config: dict with 'solvents', 'column', 'flow_rate', 'temp', 'method_length'
        
    Returns:
        separation score (higher is better)
    """
    # convert params to gradient
    total_time = lcms_config.get('method_length', 15.0)
    gradient = params_to_gradient(params, total_time)
    
    # prepare predictions list
    predictions_list = []
    for compound in compounds:
        predictions_list.append({
            'compound_smiles': compound['smiles'],
            'solvents': lcms_config['solvents'],
            'gradient': gradient,
            'column': lcms_config['column'],
            'flow_rate': lcms_config['flow_rate'],
            'temp': lcms_config['temp']
        })
    
    # predict retention times
    retention_times = predict_retention_time_from_list(predictions_list)
    
    # filter out failed predictions
    valid_rts = []
    valid_probs = []
    for i, rt in enumerate(retention_times):
        if rt is not None and rt > 0:
            valid_rts.append(rt)
            valid_probs.append(compounds[i]['probability'])
    
    # compute separation score
    method_length_seconds = total_time * 60.0
    score = compute_separation_score(valid_rts, valid_probs, method_length_seconds)
    
    return score


def batch_evaluate_gradients(params_batch: np.ndarray,
                            compounds: List[Dict],
                            lcms_config: Dict) -> np.ndarray:
    """
    evaluate multiple gradient parameterizations in batch
    
    Args:
        params_batch: array of shape (n_samples, n_params)
        compounds: list of dicts with 'smiles' and 'probability' keys
        lcms_config: dict with 'solvents', 'column', 'flow_rate', 'temp', 'method_length'
        
    Returns:
        array of separation scores of shape (n_samples,)
    """
    scores = []
    for params in params_batch:
        score = evaluate_gradient(params, compounds, lcms_config)
        scores.append(score)
    
    return np.array(scores)

