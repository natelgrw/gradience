#!/usr/bin/env python3
"""
pred_rt.py

Author: natelgrw
Last Edited: 11/15/2025

This script predicts retention time values using the ReTiNA_XGB1 XGBoost
model. All retention time predictions are returned in seconds.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Union
from ast import literal_eval
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.interpolate import interp1d

try:
    from .calc_descriptors import calculate_156_descriptors
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from rt_pred.calc_descriptors import calculate_156_descriptors

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_model = None
_model_path = None


# ===== Constants for Method Encoding ===== #


SOLVENTS_ORDER = ['O', 'CC#N', 'CO', 'CC(O)C', 'CC(C)O', 'CC(=O)C']

ADDITIVES_ORDER = [
    'C(=O)(C(F)(F)F)O',
    'C(=O)C',
    'C(=O)O',
    'C(=O)O.[NH4+]',
    'C(=O)[O-].[NH4+]',
    'C(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O',
    'CC(=O)O',
    'CC(=O)[O-].[NH4+]'
]
COLUMN_TYPES = ['RP', 'HI']

SOLVENTS_FEATURE_NAMES = ['O', 'CC_N', 'CO', 'CC_O_C', 'CC_C_O', 'CC_O_C']

ADDITIVES_FEATURE_NAMES = [
    'C_O_C_F_F_F_O',
    'C_O_C',
    'C_O_O',
    'C_O_O_NH4',
    'C_O_O_NH4',
    'C_CN_CC_O_O_CC_O_O',
    'CC_O_O',
    'CC_O_O_NH4'
]


# ===== Helper Functions ===== #


def _get_model_path():
    """Get the path to the ReTiNA_XGB1 model file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ReTiNA_XGB1", "ReTINA_XGB1.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def _load_model():
    """Load the ReTiNA_XGB1 XGBoost model (lazy loading)."""
    global _model, _model_path
    if _model is None or _model_path != _get_model_path():
        _model_path = _get_model_path()
        logger.info(f"Loading ReTiNA_XGB1 model from {_model_path}")
        _model = xgb.Booster()
        _model.load_model(_model_path)
        logger.info("ReTiNA_XGB1 model loaded successfully")
    return _model


def _get_model_feature_names() -> List[str]:
    """
    Get the feature names expected by the loaded model.
    
    Returns
    -------
    List[str]
        List of feature names in the exact order expected by the model
    """
    model = _load_model()
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        return model.feature_names
    else:
        raise ValueError("Model does not have feature names. Cannot determine feature order.")

def encode_solvents(solvents: Union[str, Dict]) -> np.ndarray:
    """
    Encode solvents into 28 features.
    
    Parameters
    ----------
    solvents : str or Dict
        Either a string representation of solvents dict or the dict itself.
        Format: {'A': [{'O': 95.0, 'CO': 5.0}, {'C(=O)O': 0.1}], 
                'B': [{'CC#N': 100.0}, {}]}
        First dict in list is solvent percentages, second is additive molarities.
    
    Returns
    -------
    np.ndarray
        28 features: 12 solvents (6 solvents x 2 phases) + 16 additives (8 additives x 2 phases)
    """
    try:
        if isinstance(solvents, str):
            solvents_dict = literal_eval(solvents)
        else:
            solvents_dict = solvents
    except:
        return np.zeros(28)
    
    features = []
    
    for phase in ['A', 'B']:
        solv_percentages = np.zeros(6)
        additive_molarities = np.zeros(8)
        
        if phase in solvents_dict and isinstance(solvents_dict[phase], list):
            phase_data = solvents_dict[phase]
            
            # solvent percentages
            if len(phase_data) > 0 and isinstance(phase_data[0], dict):
                solvent_dict = phase_data[0]
                for solvent_smiles, percentage in solvent_dict.items():
                    if solvent_smiles in SOLVENTS_ORDER:
                        idx = SOLVENTS_ORDER.index(solvent_smiles)
                        solv_percentages[idx] = float(percentage)
            
            # additive molarities
            if len(phase_data) > 1 and isinstance(phase_data[1], dict):
                additive_dict = phase_data[1]
                for additive_smiles, molarity in additive_dict.items():
                    if additive_smiles in ADDITIVES_ORDER:
                        idx = ADDITIVES_ORDER.index(additive_smiles)
                        additive_molarities[idx] = float(molarity)
        
        # adding to features
        features.extend(solv_percentages)
        features.extend(additive_molarities)
    
    return np.array(features)


def normalize_gradient(gradient: Union[str, List[Tuple[float, float]]], n_points: int = 100) -> Tuple[np.ndarray, float]:
    """
    Encodes the gradient profile and duration (in seconds).
    
    Parameters
    ----------
    gradient : str or List[Tuple[float, float]]
        Either a string representation of gradient list or the list itself.
        Format: [(time_min, percent_B), ...]
    n_points : int
        Number of interpolation points (default: 100)
    
    Returns
    -------
    Tuple[np.ndarray, float]
        gradient_vector: 100-dimensional vector of %B values
        total_time_seconds: total method duration in seconds
    """
    try:
        if isinstance(gradient, str):
            gradient_list = literal_eval(gradient)
        else:
            gradient_list = gradient
    except Exception:
        return np.zeros(n_points), 0.0
    
    if not gradient_list or len(gradient_list) < 2:
        return np.zeros(n_points), 0.0
    
    # extracting times and % B values
    try:
        times = np.array([float(point[0]) for point in gradient_list], dtype=float)
        percent_b = np.array([float(point[1]) for point in gradient_list], dtype=float)
    except (TypeError, ValueError):
        return np.zeros(n_points), 0.0
    
    try:
        total_time_minutes = float(gradient_list[-1][0])
    except (TypeError, ValueError, IndexError):
        total_time_minutes = 0.0
    
    if total_time_minutes <= 0:
        return np.zeros(n_points), 0.0
    
    # normalizing times to [0, 1]
    times_normalized = times / total_time_minutes
    
    interp_func = interp1d(times_normalized, percent_b,
                           kind='linear',
                           bounds_error=False,
                           fill_value=(percent_b[0], percent_b[-1]))
    
    # sampling at 100 uniform points
    t_uniform = np.linspace(0, 1, n_points)
    gradient_vector = interp_func(t_uniform)
    
    total_time_seconds = total_time_minutes * 60.0
    
    return gradient_vector, total_time_seconds


def encode_column(column: Union[str, Tuple]) -> np.ndarray:
    """
    Encodes the column into 5 features.
    
    Parameters
    ----------
    column : str or Tuple
        Either a string representation of column tuple or the tuple itself.
        Format: (type, diameter_mm, length_mm, particle_size_um)
        Example: ('RP', 4.6, 150, 5)
    
    Returns
    -------
    np.ndarray
        5 features: [RP_onehot, HI_onehot, diameter_mm, length_mm, particle_size_um]
    """
    try:
        if isinstance(column, str):
            column_tuple = literal_eval(column)
        else:
            column_tuple = column
    except:
        return np.zeros(5)
    
    if not isinstance(column_tuple, tuple) or len(column_tuple) != 4:
        return np.zeros(5)
    
    features = []
    
    # one-hot encoding column type
    col_type = column_tuple[0]
    features.append(1.0 if col_type == 'RP' else 0.0)
    features.append(1.0 if col_type == 'HI' else 0.0)
    
    features.append(float(column_tuple[1]))
    features.append(float(column_tuple[2]))
    features.append(float(column_tuple[3]))
    
    return np.array(features)


def _extract_features(compound_smiles: str,
                     solvents: Union[str, Dict],
                     gradient: Union[str, List[Tuple[float, float]]],
                     column: Union[str, Tuple],
                     flow_rate: float,
                     temp: float,
                     feature_names: List[str]) -> Optional[np.ndarray]:
    """
    Extract all features in the order expected by the model.
    
    Parameters
    ----------
    compound_smiles : str
        SMILES string of the compound
    solvents : str or Dict
        Solvents encoding (see encode_solvents)
    gradient : str or List[Tuple[float, float]]
        Gradient encoding (see normalize_gradient)
    column : str or Tuple
        Column encoding (see encode_column)
    flow_rate : float
        Flow rate in mL/min
    temp : float
        Temperature in Celsius
    feature_names : List[str]
        List of feature names in the exact order expected by the model
    
    Returns
    -------
    np.ndarray or None
        Array of 292 features in the correct order, or None if extraction failed
    """
    try:
        # calculate compound descriptors (156 features)
        compound_descriptors = calculate_156_descriptors(compound_smiles)
        if compound_descriptors is None:
            logger.warning(f"Could not calculate descriptors for compound: {compound_smiles}")
            return None
        
        # encoding features
        solvent_features = encode_solvents(solvents)
        
        gradient_features, gradient_total_time = normalize_gradient(gradient)
        
        column_features = encode_column(column)
        
        feature_dict = {}
        
        for key, value in compound_descriptors.items():
            feature_dict[f"comp_{key}"] = value
        
        solvent_feature_names_A = [
            'solv_O_A_pct',
            'solv_CC_N_A_pct',
            'solv_CO_A_pct',
            'solv_CC_O_C_A_pct',
            'solv_CC_C_O_A_pct',
            'solv_CC_O_C_A_pct_2'
        ]
        for i, feature_name in enumerate(solvent_feature_names_A):
            feature_dict[feature_name] = solvent_features[i]
        
        additive_feature_names_A = [
            'add_C_O_C_F_F_F_O_A_M',
            'add_C_O_C_A_M',
            'add_C_O_O_A_M',
            'add_C_O_O_NH4_A_M',
            'add_C_O_O_NH4_A_M_2',
            'add_C_CN_CC_O_O_CC_O_O_A_M',
            'add_CC_O_O_A_M',
            'add_CC_O_O_NH4_A_M'
        ]
        for i, feature_name in enumerate(additive_feature_names_A):
            feature_dict[feature_name] = solvent_features[6 + i]
        
        solvent_feature_names_B = [
            'solv_O_B_pct',
            'solv_CC_N_B_pct',
            'solv_CO_B_pct',
            'solv_CC_O_C_B_pct',
            'solv_CC_C_O_B_pct', 
            'solv_CC_O_C_B_pct_2'
        ]
        for i, feature_name in enumerate(solvent_feature_names_B):
            feature_dict[feature_name] = solvent_features[12 + i]
        
        additive_feature_names_B = [
            'add_C_O_C_F_F_F_O_B_M',
            'add_C_O_C_B_M',
            'add_C_O_O_B_M',
            'add_C_O_O_NH4_B_M',
            'add_C_O_O_NH4_B_M_2',
            'add_C_CN_CC_O_O_CC_O_O_B_M',
            'add_CC_O_O_B_M',
            'add_CC_O_O_NH4_B_M'
        ]
        for i, feature_name in enumerate(additive_feature_names_B):
            feature_dict[feature_name] = solvent_features[18 + i]
        
        for i in range(100):
            feature_dict[f"grad_t{i:03d}"] = gradient_features[i]
        feature_dict["grad_total_time"] = gradient_total_time
        
        feature_dict['col_RP'] = column_features[0]
        feature_dict['col_HI'] = column_features[1]
        feature_dict['col_diam_mm'] = column_features[2]
        feature_dict['col_len_mm'] = column_features[3]
        feature_dict['col_part_um'] = column_features[4]
        
        feature_dict['flow_rate_mL_min'] = flow_rate
        feature_dict['temp_C'] = temp
        
        features = []
        for feature_name in feature_names:
            features.append(feature_dict.get(feature_name, 0.0))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None


def predict_retention_time(compound_smiles: str,
                          solvents: Union[str, Dict],
                          gradient: Union[str, List[Tuple[float, float]]],
                          column: Union[str, Tuple],
                          flow_rate: float,
                          temp: float) -> Optional[float]:
    """
    Predict retention time for a compound with given method parameters.
    
    Parameters
    ----------
    compound_smiles : str
        SMILES string of the compound
    solvents : str or Dict
        Solvents encoding. Format: {'A': [{'O': 95.0, 'CO': 5.0}, {'C(=O)O': 0.1}], 
                                    'B': [{'CC#N': 100.0}, {}]}
    gradient : str or List[Tuple[float, float]]
        Gradient profile. Format: [(time_min, percent_B), ...]
        Example: [(0, 5), (10, 95), (15, 95)]
    column : str or Tuple
        Column specification. Format: (type, diameter_mm, length_mm, particle_size_um)
        Example: ('RP', 4.6, 150, 5)
    flow_rate : float
        Flow rate in mL/min
    temp : float
        Temperature in Celsius
    
    Returns
    -------
    float or None
        Predicted retention time value in SECONDS, or None if prediction failed
    """
    results = predict_retention_time_from_list([{
        'compound_smiles': compound_smiles,
        'solvents': solvents,
        'gradient': gradient,
        'column': column,
        'flow_rate': flow_rate,
        'temp': temp
    }])
    
    if results:
        return results[0]
    return None


def predict_retention_time_from_list(predictions_list: List[Dict]) -> List[Optional[float]]:
    """
    Predict retention time values for multiple compounds with method parameters.
    
    Parameters
    ----------
    predictions_list : List[Dict]
        List of dictionaries, each containing:
        - compound_smiles: str
        - solvents: str or Dict
        - gradient: str or List[Tuple[float, float]]
        - column: str or Tuple
        - flow_rate: float
        - temp: float
    
    Returns
    -------
    List[Optional[float]]
        List of predicted retention time values in SECONDS (None for failed predictions)
    """
    model = _load_model()
    
    try:
        feature_names = _get_model_feature_names()
    except ValueError as e:
        logger.error(f"Could not get feature names from model: {e}")
        return [None] * len(predictions_list)
    
    features_list = []
    valid_indices = []
    
    for idx, pred_dict in enumerate(predictions_list):
        features = _extract_features(
            compound_smiles=pred_dict['compound_smiles'],
            solvents=pred_dict['solvents'],
            gradient=pred_dict['gradient'],
            column=pred_dict['column'],
            flow_rate=pred_dict['flow_rate'],
            temp=pred_dict['temp'],
            feature_names=feature_names
        )
        
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
        else:
            logger.warning(f"Could not extract features for prediction {idx}")
    
    if not features_list:
        logger.error("No valid predictions provided")
        return [None] * len(predictions_list)
    
    X_df = pd.DataFrame(features_list, columns=feature_names)
    dmatrix = xgb.DMatrix(X_df)
    
    logger.info(f"Making predictions for {len(valid_indices)} compounds")
    predictions = model.predict(dmatrix)
    
    results = [None] * len(predictions_list)
    for valid_idx, pred in zip(valid_indices, predictions):
        results[valid_idx] = float(pred)
    
    return results


def predict_retention_time_from_smiles(smiles_list: List[str]) -> Dict[str, float]:
    """
    Predict retention time values for a list of SMILES strings using default method parameters.
    
    This is a backward compatibility function that uses default method parameters.
    For more accurate predictions, use predict_retention_time with full method parameters.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to predict retention time for
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping SMILES to predicted retention time values in SECONDS
    """
    # default method parameters
    default_solvents = {
        'A': [{'O': 95.0, 'CO': 5.0}, {}],
        'B': [{'CC#N': 100.0}, {}]
    }
    default_gradient = [(0, 5), (10, 95), (15, 95)]
    default_column = ('RP', 4.6, 150, 5)
    default_flow_rate = 1.0
    default_temp = 40.0
    
    predictions_list = [{
        'compound_smiles': smiles,
        'solvents': default_solvents,
        'gradient': default_gradient,
        'column': default_column,
        'flow_rate': default_flow_rate,
        'temp': default_temp
    } for smiles in smiles_list]
    
    results_list = predict_retention_time_from_list(predictions_list)
    
    results = {}
    for smiles, rt in zip(smiles_list, results_list):
        if rt is not None:
            results[smiles] = rt
    
    return results


# ===== Main Function ===== #


if __name__ == "__main__":
    print("Testing retention time prediction with ReTiNA_XGB1...")
    print("Note: Retention times are predicted in SECONDS\n")
    
    print("1. Testing with full method parameters:")
    rt = predict_retention_time(
        compound_smiles="CCO",
        solvents={
            'A': [{'O': 95.0, 'CO': 5.0}, {}],
            'B': [{'CC#N': 100.0}, {}]
        },
        gradient=[(0, 5), (10, 95), (15, 95)],
        column=('RP', 4.6, 150, 5),
        flow_rate=1.0,
        temp=40.0
    )
    if rt:
        print(f"   CCO: {rt:.2f} s ({rt/60:.2f} min)")
    else:
        print("   CCO: Failed")
