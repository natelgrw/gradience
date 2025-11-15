#!/usr/bin/env python3
"""
run_pipeline.py

Author: natelgrw
Last Edited: 11/15/2025

Main pipeline for LC-MS gradient optimization.
"""

import asyncio
import argparse
import json
import sys
import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from product_pred.askcos_scraper import scrape_askcos
from optimizer.turbo import TuRBO
from optimizer.gradient_params import params_to_gradient, get_bounds
from optimizer.objective import evaluate_gradient
from optimizer.gradient_params import params_to_gradient



# ===== Default Configuration Variables ===== #


# default LC-MS configuration
DEFAULT_LCMS_CONFIG = {
    'solvents': {
        'A': [{'O': 95.0, 'CO': 5.0}, {'C(=O)O': 0.1}], 
        'B': [{'CC#N': 100.0}, {}] 
    },
    'column': ('RP', 4.6, 150, 5),
    'flow_rate': 1.0, 
    'temp': 40.0,
    'method_length': 15.0
}

# optimization settings
OPTIMIZER_CONFIG = {
    'n_init': 36, 
    'max_evals': 180, 
    'batch_size': 1, 
    'trust_region_init': 0.8, 
    'trust_region_min': 0.1, 
    'dim': 18,
    'verbose': True
}

# paths
MODEL_PATH = 'rt_pred/ReTiNA_XGB1/ReTINA_XGB1.json'


# ===== Functions For Pipeline ===== #


async def get_compounds(reactant_smiles: List[str], 
                       solvent_smiles: str) -> List[Dict]:
    """
    Fetches predicted compounds from askcos.
    
    Args:
        reactant_smiles: list of reactant smiles strings
        solvent_smiles: solvent smiles string
        
    Returns:
        list of dicts with 'smiles', 'probability', 'mol_weight' keys
    """
    results = await scrape_askcos(reactant_smiles, solvent_smiles)
    
    # convert to compounds list
    compounds = []
    for result in results:
        smiles = result['smiles']
        prob = float(result['probability']) if result['probability'] != 1 else 1.0
        compounds.append({
            'smiles': smiles,
            'probability': prob,
            'mol_weight': float(result['mol_weight'])
        })
    
    print(f"found {len(compounds)} compounds")
    
    return compounds


def optimize_gradient(compounds: List[Dict],
                     lcms_config: Dict,
                     optimizer_config: Dict) -> Dict:
    """
    Optimizes a gradient for compound separation.
    
    Args:
        compounds: list of compound dicts
        lcms_config: lcms configuration dict
        optimizer_config: optimizer configuration dict
        
    Returns:
        dict with 'params', 'gradient', 'score' keys
    """    
    # create objective function
    def objective(params):
        return evaluate_gradient(params, compounds, lcms_config)
    
    # get bounds
    bounds = get_bounds(dim=optimizer_config['dim'])
    
    # run turbo optimization
    optimizer = TuRBO(
        objective_fn=objective,
        dim=optimizer_config['dim'],
        bounds=bounds,
        n_init=optimizer_config['n_init'],
        max_evals=optimizer_config['max_evals'],
        batch_size=optimizer_config['batch_size'],
        trust_region_init=optimizer_config['trust_region_init'],
        trust_region_min=optimizer_config['trust_region_min'],
        verbose=optimizer_config['verbose']
    )
    
    best_params, best_score = optimizer.optimize()
    
    # convert to gradient
    gradient = params_to_gradient(best_params, lcms_config['method_length'])
    
    return {
        'params': best_params.tolist(),
        'gradient': gradient,
        'score': float(best_score)
    }


def format_gradient_output(gradient: List, lcms_config: Dict) -> str:
    """
    Formats a gradient for display.
    
    Args:
        gradient: list of (time_min, percent_b) tuples
        lcms_config: lcms configuration dict
        
    Returns:
        formatted string
    """
    lines = []
    lines.append("\nOptimized Gradient:")
    lines.append("-" * 40)
    for time, percent_b in gradient:
        lines.append(f"  {time:6.2f} min: {percent_b:5.1f}% B")
    lines.append("-" * 40)
    
    # method details
    lines.append("\nMethod Details:")
    lines.append(f"  Column: {lcms_config['column']}")
    lines.append(f"  Flow Rate: {lcms_config['flow_rate']} mL/min")
    lines.append(f"  Temperature: {lcms_config['temp']} C")
    lines.append(f"  Method Length: {lcms_config['method_length']} min")
    
    return "\n".join(lines)


async def run_pipeline(reactant_smiles: List[str],
                      solvent_smiles: str,
                      lcms_config: Optional[Dict] = None,
                      optimizer_config: Optional[Dict] = None,
                      output_file: Optional[str] = None):
    """
    main pipeline for gradient optimization
    
    Args:
        reactant_smiles: list of reactant smiles strings
        solvent_smiles: solvent smiles string
        lcms_config: lcms configuration (uses default if none)
        optimizer_config: optimizer configuration (uses default if none)
        output_file: optional output OptGradient .json file
    """
    # use defaults if not provided
    if lcms_config is None:
        lcms_config = DEFAULT_LCMS_CONFIG.copy()
    if optimizer_config is None:
        optimizer_config = OPTIMIZER_CONFIG.copy()
    
    print("=" * 60)
    print("LC-MS gradient optimization pipeline")
    print("=" * 60)
    
    # get compounds from askcos
    compounds = await get_compounds(reactant_smiles, solvent_smiles)
    
    if len(compounds) < 2:
        print("\nError: Need At Least 2 Compounds For Optimization")
        return
    
    # optimize gradient
    result = optimize_gradient(compounds, lcms_config, optimizer_config)
    
    # display results
    print(format_gradient_output(result['gradient'], lcms_config))
    print(f"\nSeparation Score: {result['score']:.4f}")
    
    # save results to OptGradient .json file if requested
    if output_file:
        output = {
            'reactants': reactant_smiles,
            'solvent': solvent_smiles,
            'predicted_products': compounds,
            'lcms_config': lcms_config,
            'optimized_gradient': result['gradient'],
            'gradient_params': result['params'],
            'separation_score': result['score']
        }

        with open(f"results/OptGradient_{output_file}", 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults Saved To results/OptGradient_{output_file}")
        
    
    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)


def plot_gradient(params, total_time=15.0, title="Gradient Profile", lcms_config=None):
    """
    plot a gradient profile from parameters
    
    Args:
        params: array of 18 gradient parameters (10 %B + 8 time spacings)
        total_time: total method duration in minutes
        title: plot title
        lcms_config: optional dict with column, flow_rate, temp info to display
    """
    gradient = params_to_gradient(params, total_time)
    
    times = [point[0] for point in gradient]
    percent_b = [point[1] for point in gradient]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, percent_b, 'b-', linewidth=2.5, marker='o', markersize=8, label='optimized gradient')
    plt.xlabel('time (min)', fontsize=12, fontweight='bold')
    plt.ylabel('%B', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(-0.5, max(times) + 0.5)
    plt.ylim(0, 105)
    
    # annotate points
    for t, b in zip(times, percent_b):
        plt.annotate(f'{b:.1f}%', xy=(t, b), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # add method info if provided
    if lcms_config:
        column = lcms_config.get('column', ('RP', 2.1, 100, 1.7))
        info_text = f"Column: {column[0]} {column[2]}×{column[1]}mm, {column[3]}μm\n"
        info_text += f"Flow: {lcms_config.get('flow_rate', 0.4)} mL/min, "
        info_text += f"Temp: {lcms_config.get('temp', 45.0)}°C\n"
        info_text += "Mobile phases: A=H2O/ACN/FA, B=ACN/FA"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    print(f"\nGradient Profile: {len(gradient)} Points Over {max(times):.1f} Minutes")
