"""
assign_compounds.py

Uses a compound matching score derived through match_utils.py combined with
data from Virtual Flask reaction prediction software to assign compound_id
specifications to each peak in an LC-MS summary dictionary.

In this way, the LC-MS summary dictionary can be utilized to accurately
maximize compound separation. The distance between peaks classified with 
the same compound_id will be minimized, and the distance between peaks
classified with different compound_id will be maximized. Peaks that do not
have a compound_id (compound_id = None) will be ignored.

LC-MS Complete Summary Dictionary Example:

Summary = {
    "reaction": "Fischer esterification - methyl salicylate synthesis",
    "reactants": [
        "O=C(O)c1ccccc1O",
        "CO"
    ],
    "solvents": [
        "CCO",              
        "CN(C)C=O"         
    ],
    "lambda_range": (220, 400),
    "baseline_method": "asls",
    "time_range": (0.0, 12.0),
    "min_distance": 0.45,
    "peaks": [
        {
            "time_range": (2.1, 2.7),
            "all_maxima": [2.4],
            "area": 13450.7,
            "lambda_max": [272],
            "mass_prediction": (149.3, 154.4),
            "compound_id": 1,
            "compound_smiles": "O=C(O)c1ccccc1O"
        },
        {
            "time_range": (3.4, 3.9),
            "all_maxima": [3.6],
            "area": 18500.2,
            "lambda_max": [258],
            "mass": (135.8, 136.8),
            "compound_id": 2,
            "compound_smiles": "CO"
        },
        {
            "time_range": (5.2, 5.9),
            "all_maxima": [5.6],
            "area": 24310.9,
            "lambda_max": [280],
            "mass": (164.8, 166.4),
            "compound_id": 3,
            "compound_smiles": "COC(=O)c1ccccc1O"
        },
        {
            "time_range": (6.1, 6.5),
            "all_maxima": [6.3],
            "area": 3200.4,
            "lambda_max": [242],
            "mass": (150.1, 155.8),
            "compound_id": None,
            "compound_smiles": None
        }
    ]
}

Author: natelgrw
Created: 06/01/2025
Last Edited: 6/01/2025
"""