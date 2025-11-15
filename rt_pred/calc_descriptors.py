#!/usr/bin/env python3
"""
calc_descriptors.py

Author: natelgrw
Last Edited: 11/10/2025

This module provides a function to calculate 156 molecular 
descriptors for a target molecule for model property prediction.
"""

from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, MolSurf, Lipinski, GraphDescriptors, Crippen, EState


# ===== Helper Functions For Functional Groups ===== #


def count_functional_groups(mol, smarts_pattern):
    """
    Helper function to count functional groups using SMARTS patterns.
    """
    try:
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is None:
            return 0
        return len(mol.GetSubstructMatches(pattern))
    except:
        return 0


def fr_Al_COO(mol):
    """
    Count of aliphatic carboxylic acid groups (-COOH).
    """
    return count_functional_groups(mol, '[C;!$(C=O)]C(=O)[O;H1]')


def fr_Al_OH(mol):
    """
    Count of aliphatic alcohol groups (-OH).
    """
    return count_functional_groups(mol, '[C;!$(C=O)]O[H]')


def fr_Al_OH_noTert(mol):
    """
    Count of non-tertiary aliphatic alcohol groups (exclude carbonyls).
    """
    all_alcohols = count_functional_groups(mol, "[C;!$(C=O)][OH]")
    tertiary_alcohols = count_functional_groups(mol, "[C;X4;$(C([#6])([#6])[#6])][OH]")
    return max(0, all_alcohols - tertiary_alcohols)


def fr_ArN(mol):
    """
    Count of aromatic amines (aniline-type: N directly bonded to an aromatic carbon).
    """
    return count_functional_groups(mol, "[c][NH1,NH2]")


def fr_Ar_COO(mol):
    """
    Count of aromatic carboxylic acid groups (-COOH or -COO⁻ attached to an aromatic carbon).
    """
    return count_functional_groups(mol, "[c]C(=O)[O;H1,-]")


def fr_Ar_N(mol):
    """
    Count of aromatic ring nitrogens (pyridine-like and pyrrole-like), excluding exocyclic amines.
    """
    return count_functional_groups(mol, "n")


def fr_Ar_NH(mol):
    """
    Count of aromatic amine groups (-NH₂ or -NH- attached to an aromatic carbon).
    """
    return count_functional_groups(mol, "[NX3;H2,H1][c]")


def fr_Ar_OH(mol):
    """
    Count of phenol groups (aromatic alcohols).
    """
    return count_functional_groups(mol, "[c][OH]")


def fr_C_O_noCOO(mol):
    """
    Count of carbonyls excluding carboxylic acids.
    """
    return count_functional_groups(mol, '[C;!$(C(=O)[O;H1])]=[O]')


def fr_Imine(mol):
    """
    Count of imine groups (C=NH or C=NR).
    """
    return count_functional_groups(mol, "[C]=[N;H0,H1]")


def fr_NH0(mol):
    """
    Count of nitrogen with zero attached hydrogens (e.g., quaternary).
    """
    return count_functional_groups(mol, '[N;H0]')


def fr_NH1(mol):
    """
    Count of nitrogen with one attached hydrogen.
    """
    return count_functional_groups(mol, '[N;H1]')


def fr_NH2(mol):
    """
    Count of nitrogen with two attached hydrogens.
    """
    return count_functional_groups(mol, '[N;H2]')


def fr_N_O(mol):
    """
    Count of nitrogen-oxygen bonds.
    """
    return count_functional_groups(mol, '[N]-[O]')


def fr_Nhpyrrole(mol):
    """
    Count of nitrogen atoms in pyrrole rings.
    """
    return count_functional_groups(mol, '[nH]1cccc1')


def fr_SH(mol):
    """
    Count of thiol groups (-SH).
    """
    return count_functional_groups(mol, '[S;H1][C]')


def fr_aldehyde(mol):
    """
    Count of aldehyde groups (-CHO).
    """
    return count_functional_groups(mol, '[C;H1](=O)[!#6]')


def fr_alkyl_carbamate(mol):
    """
    Count of alkyl carbamate groups.
    """
    return count_functional_groups(mol, '[C]OC(=O)N')


def fr_alkyl_halide(mol):
    """
    Count of alkyl halide groups.
    """
    return count_functional_groups(mol, '[C;!$(C=O)][F,Cl,Br,I]')


def fr_allylic_oxid(mol):
    """
    Count of allylic oxidation sites.
    """
    return count_functional_groups(mol, '[C;H2]=[C][C;H2]')


def fr_amide(mol):
    """
    Count of amide groups.
    """
    return count_functional_groups(mol, '[C](=O)[N;H0,H1,H2]')


def fr_amidine(mol):
    """
    Count of amidine groups.
    """
    return count_functional_groups(mol, '[C](=[N])[N;H0,H1,H2]')


def fr_aryl_methyl(mol):
    """
    Count of methyl groups attached to aromatic rings.
    """
    return count_functional_groups(mol, '[c][C;H3]')


def fr_azide(mol):
    """
    Count of azide groups (-N3).
    """
    return count_functional_groups(mol, '[N]=[N]=[N]')


def fr_azo(mol):
    """
    Count of azo groups (-N=N-).
    """
    return count_functional_groups(mol, '[N]=[N]')


def fr_barbitur(mol):
    """
    Count of barbituric acid-like groups.
    """
    return count_functional_groups(mol, '[C]1(=O)[N;H1]C(=O)[N;H1]C(=O)[N;H1]1')


def fr_benzene(mol):
    """
    Count of benzene rings.
    """
    return count_functional_groups(mol, 'c1ccccc1')


def fr_benzodiazepine(mol):
    """
    Count of benzodiazepine rings.
    """
    return count_functional_groups(mol, 'c1ccc2c(c1)nc(=O)cn2')


def fr_bicyclic(mol):
    """
    Count of bicyclic ring systems.
    """
    try:
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        return 1 if num_rings >= 2 else 0
    except:
        return 0


def fr_diazo(mol):
    """
    Count of diazo groups.
    """
    return count_functional_groups(mol, '[C][N]=[N]')


def fr_dihydropyridine(mol):
    """
    Count of dihydropyridine rings.
    """
    return count_functional_groups(mol, '[nH]1cccc[c]1')


def fr_epoxide(mol):
    """
    Count of epoxide rings.
    """
    return count_functional_groups(mol, '[C]1OC1')


def fr_ester(mol):
    """
    Count of ester groups.
    """
    return count_functional_groups(mol, '[C](=O)[O;H0]')


def fr_ether(mol):
    """
    Count of ether groups.
    """
    return count_functional_groups(mol, '[C,c;!$(C=O)][O;H0][C;!$(C=O)]')


def fr_furan(mol):
    """
    Count of furan rings.
    """
    return count_functional_groups(mol, '[o]1cccc1')


def fr_guanido(mol):
    """
    Count of guanidine groups.
    """
    return count_functional_groups(mol, '[N;H1,H0]C(=[N;H1,H0])[N;H1,H0]')


def fr_halogen(mol):
    """
    Count of halogen atoms (F, Cl, Br, I).
    """
    return count_functional_groups(mol, '[F,Cl,Br,I]')


def fr_hdrzine(mol):
    """
    Count of hydrazine groups.
    """
    return count_functional_groups(mol, '[N;H1][N;H1,H0]')


def fr_hdrzone(mol):
    """
    Count of hydrazone groups.
    """
    return count_functional_groups(mol, '[C]=[N][N;H1,H0]')


def fr_imidazole(mol):
    """
    Count of imidazole rings.
    """
    return count_functional_groups(mol, '[nH,n]1cncc1')


def fr_imid(mol):
    """
    Count of imide groups.
    """
    return count_functional_groups(mol, '[C](=O)[N;H0,H1]C(=O)')


def fr_isocyan(mol):
    """
    Count of isocyanate groups.
    """
    return count_functional_groups(mol, '[N]=[C]=[O]')


def fr_isothiocyan(mol):
    """
    Count of isothiocyanate groups.
    """
    return count_functional_groups(mol, '[N]=[C]=[S]')


def fr_ketone(mol):
    """
    Count of ketone groups.
    """
    return count_functional_groups(mol, '[C;!$(C=O)](=O)[C;!$(C=O)]')


def fr_lactam(mol):
    """
    Count of lactam rings.
    """
    return count_functional_groups(mol, '[C]1C(=O)N[C,C]C1')


def fr_lactone(mol):
    """
    Count of lactone rings.
    """
    return count_functional_groups(mol, '[C]1C(=O)O[C,C][C]1')


def fr_methoxy(mol):
    """
    Count of methoxy groups (-OCH3).
    """
    return count_functional_groups(mol, '[O;H0][C;H3]')


def fr_morpholine(mol):
    """
    Count of morpholine rings.
    """
    return count_functional_groups(mol, '[O]1CC[N;H0]CC1')


def fr_nitrile(mol):
    """
    Count of nitrile groups (-C≡N).
    """
    return count_functional_groups(mol, '[C]#[N]')


def fr_nitro(mol):
    """
    Count of nitro groups (-NO2).
    """
    return count_functional_groups(mol, '[N](=O)(=O)')


def fr_nitro_arom(mol):
    """
    Count of aromatic nitro groups.
    """
    return count_functional_groups(mol, '[c][N](=O)(=O)')


def fr_nitroso(mol):
    """
    Count of nitroso groups (-NO).
    """
    return count_functional_groups(mol, '[N]=[O]')


def fr_oxazole(mol):
    """
    Count of oxazole rings.
    """
    return count_functional_groups(mol, '[o]1[c,n][c,n][c,n]1')


def fr_oxime(mol):
    """
    Count of oxime groups (=NOH).
    """
    return count_functional_groups(mol, '[C;!$(C=O)]=[N][O;H1]')


def fr_aromatic_H(mol):
    """
    Count of aromatic carbons with H (possible hydroxylation sites).
    """
    return count_functional_groups(mol, '[cH]')


def fr_phenol(mol):
    """
    Count of phenol groups.
    """
    return count_functional_groups(mol, '[c]O[H]')


def fr_COO(mol):
    """
    Count of carboxylic acid groups (-COOH).
    """
    return count_functional_groups(mol, 'C(=O)[O;H1]')


def fr_C_O(mol):
    """
    Count of carbonyl groups (C=O).
    """
    return count_functional_groups(mol, '[C]=[O]')


def fr_C_S(mol):
    """
    Count of carbon-sulfur bonds.
    """
    return count_functional_groups(mol, '[C][S]')


def fr_HOCCN(mol):
    """
    Count of hydroxyl connected to carbon connected to nitrogen.
    """
    return count_functional_groups(mol, '[O;H1][C;!$(C=O)][N;H0,H1]')


def fr_phos_acid(mol):
    """
    Count of phosphoric acid groups.
    """
    return count_functional_groups(mol, '[P](=O)([O;H1])([O;H1])[O;H1]')


def fr_phos_ester(mol):
    """
    Count of phosphoric ester groups.
    """
    return count_functional_groups(mol, '[P](=O)([O;H0])([O;H0])[O;H0]')


def fr_priamide(mol):
    """
    Count of primary amides.
    """
    return count_functional_groups(mol, '[C](=O)[N;H2]')


def fr_prisulfonamd(mol):
    """
    Count of primary sulfonamides.
    """
    return count_functional_groups(mol, '[S](=O)(=O)[N;H2]')


def fr_pyridine(mol):
    """
    Count of pyridine rings.
    """
    return count_functional_groups(mol, '[n]1ccccc1')


def fr_quatN(mol):
    """
    Count of quaternary nitrogen atoms.
    """
    return count_functional_groups(mol, '[N+4]')


def fr_sulfide(mol):
    """
    Count of sulfide groups (thioethers).
    """
    return count_functional_groups(mol, '[C;!$(C=O)][S;H0][C;!$(C=O)]')


def fr_sulfonamd(mol):
    """
    Count of sulfonamide groups.
    """
    return count_functional_groups(mol, '[S](=O)(=O)[N;H0,H1]')


def fr_sulfone(mol):
    """
    Count of sulfone groups.
    """
    return count_functional_groups(mol, '[C][S](=O)(=O)[C]')


def fr_term_acetylene(mol):
    """
    Count of terminal acetylene groups.
    """
    return count_functional_groups(mol, '[C]#[C;H1]')


def fr_tetrazole(mol):
    """
    Count of tetrazole rings.
    """
    return count_functional_groups(mol, '[nH,n]1nncn1')


def fr_thiazole(mol):
    """
    Count of thiazole rings.
    """
    return count_functional_groups(mol, '[s]1[c,n][c,n][c,n]1')


def fr_thiocyan(mol):
    """
    Count of thiocyanate groups.
    """
    return count_functional_groups(mol, '[S][C]#[N]')


def fr_thiophene(mol):
    """
    Count of thiophene rings.
    """
    return count_functional_groups(mol, '[s]1[cH][cH][cH][cH]1')


def fr_urea(mol):
    """
    Count of urea groups.
    """
    return count_functional_groups(mol, '[N;H0,H1]C(=O)[N;H0,H1]')


# ===== Function To Calculate All 156 Descriptors ===== #


def calculate_156_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    Calculate all 156 molecular descriptors for a given SMILES string.

    Returns a dictionary containing all 156 calculated descriptors, or None if invalid SMILES.
    Keys are descriptor names, and values are numeric descriptor values.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {}
        
        # 1. BalabanJ - molecular complexity based on average distance connectivity
        descriptors['BalabanJ'] = GraphDescriptors.BalabanJ(mol)
        
        # 2. BertzCT - molecular complexity based on graph connectivity
        descriptors['BertzCT'] = GraphDescriptors.BertzCT(mol)
        
        # 3-5. Chi indices (0-2)
        descriptors['Chi0'] = GraphDescriptors.Chi0(mol)
        descriptors['Chi1'] = GraphDescriptors.Chi1(mol)
        
        # 6-10. Chi_n indices (0-4)
        descriptors['Chi0n'] = GraphDescriptors.Chi0n(mol)
        descriptors['Chi1n'] = GraphDescriptors.Chi1n(mol)
        descriptors['Chi2n'] = GraphDescriptors.Chi2n(mol)
        descriptors['Chi3n'] = GraphDescriptors.Chi3n(mol)
        descriptors['Chi4n'] = GraphDescriptors.Chi4n(mol)
        
        # 11-15. Chi_v indices (0-4)
        descriptors['Chi0v'] = GraphDescriptors.Chi0v(mol)
        descriptors['Chi1v'] = GraphDescriptors.Chi1v(mol)
        descriptors['Chi2v'] = GraphDescriptors.Chi2v(mol)
        descriptors['Chi3v'] = GraphDescriptors.Chi3v(mol)
        descriptors['Chi4v'] = GraphDescriptors.Chi4v(mol)
        
        # 18-20. Kappa indices (1-3)
        descriptors['Kappa1'] = GraphDescriptors.Kappa1(mol)
        descriptors['Kappa2'] = GraphDescriptors.Kappa2(mol)
        descriptors['Kappa3'] = GraphDescriptors.Kappa3(mol)
        
        # 21-24. electrotopological and electronic descriptors
        descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)

        # 25-26. electron counts
        descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
        descriptors['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
        
        # 27. HallKierAlpha - electrotopological descriptor
        descriptors['HallKierAlpha'] = GraphDescriptors.HallKierAlpha(mol)
        
        # 28-29. surface area descriptors
        descriptors['LabuteASA'] = MolSurf.LabuteASA(mol)
        descriptors['TPSA'] = MolSurf.TPSA(mol)
        
        # 30-38. EState_VSA descriptors (1-9)
        descriptors['EState_VSA1'] = EState.EState_VSA.EState_VSA1(mol)
        descriptors['EState_VSA2'] = EState.EState_VSA.EState_VSA2(mol)
        descriptors['EState_VSA3'] = EState.EState_VSA.EState_VSA3(mol)
        descriptors['EState_VSA4'] = EState.EState_VSA.EState_VSA4(mol)
        descriptors['EState_VSA5'] = EState.EState_VSA.EState_VSA5(mol)
        descriptors['EState_VSA6'] = EState.EState_VSA.EState_VSA6(mol)
        descriptors['EState_VSA7'] = EState.EState_VSA.EState_VSA7(mol)
        descriptors['EState_VSA8'] = EState.EState_VSA.EState_VSA8(mol)
        descriptors['EState_VSA9'] = EState.EState_VSA.EState_VSA9(mol)
        
        # 39-47. PEOE_VSA descriptors (1-9)
        descriptors['PEOE_VSA1'] = MolSurf.PEOE_VSA1(mol)
        descriptors['PEOE_VSA2'] = MolSurf.PEOE_VSA2(mol)
        descriptors['PEOE_VSA3'] = MolSurf.PEOE_VSA3(mol)
        descriptors['PEOE_VSA4'] = MolSurf.PEOE_VSA4(mol)
        descriptors['PEOE_VSA5'] = MolSurf.PEOE_VSA5(mol)
        descriptors['PEOE_VSA6'] = MolSurf.PEOE_VSA6(mol)
        descriptors['PEOE_VSA7'] = MolSurf.PEOE_VSA7(mol)
        descriptors['PEOE_VSA8'] = MolSurf.PEOE_VSA8(mol)
        descriptors['PEOE_VSA9'] = MolSurf.PEOE_VSA9(mol)
        
        # 48-56. SMR_VSA descriptors (1-9)
        descriptors['SMR_VSA1'] = MolSurf.SMR_VSA1(mol)
        descriptors['SMR_VSA2'] = MolSurf.SMR_VSA2(mol)
        descriptors['SMR_VSA3'] = MolSurf.SMR_VSA3(mol)
        descriptors['SMR_VSA4'] = MolSurf.SMR_VSA4(mol)
        descriptors['SMR_VSA5'] = MolSurf.SMR_VSA5(mol)
        descriptors['SMR_VSA6'] = MolSurf.SMR_VSA6(mol)
        descriptors['SMR_VSA7'] = MolSurf.SMR_VSA7(mol)
        descriptors['SMR_VSA8'] = MolSurf.SMR_VSA8(mol)
        descriptors['SMR_VSA9'] = MolSurf.SMR_VSA9(mol)
        
        # 57-68. SlogP_VSA descriptors (1-12)
        descriptors['SlogP_VSA1'] = MolSurf.SlogP_VSA1(mol)
        descriptors['SlogP_VSA2'] = MolSurf.SlogP_VSA2(mol)
        descriptors['SlogP_VSA3'] = MolSurf.SlogP_VSA3(mol)
        descriptors['SlogP_VSA4'] = MolSurf.SlogP_VSA4(mol)
        descriptors['SlogP_VSA5'] = MolSurf.SlogP_VSA5(mol)
        descriptors['SlogP_VSA6'] = MolSurf.SlogP_VSA6(mol)
        descriptors['SlogP_VSA7'] = MolSurf.SlogP_VSA7(mol)
        descriptors['SlogP_VSA8'] = MolSurf.SlogP_VSA8(mol)
        descriptors['SlogP_VSA9'] = MolSurf.SlogP_VSA9(mol)
        descriptors['SlogP_VSA10'] = MolSurf.SlogP_VSA10(mol)
        descriptors['SlogP_VSA11'] = MolSurf.SlogP_VSA11(mol)
        descriptors['SlogP_VSA12'] = MolSurf.SlogP_VSA12(mol)
        
        # 69-77. ring count descriptors
        descriptors['NumAliphaticCarbocycles'] = Lipinski.NumAliphaticCarbocycles(mol)
        descriptors['NumAliphaticHeterocycles'] = Lipinski.NumAliphaticHeterocycles(mol)
        descriptors['NumAliphaticRings'] = Lipinski.NumAliphaticRings(mol)
        descriptors['NumAromaticCarbocycles'] = Lipinski.NumAromaticCarbocycles(mol)
        descriptors['NumAromaticHeterocycles'] = Lipinski.NumAromaticHeterocycles(mol)
        descriptors['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
        descriptors['NumSaturatedCarbocycles'] = Lipinski.NumSaturatedCarbocycles(mol)
        descriptors['NumSaturatedHeterocycles'] = Lipinski.NumSaturatedHeterocycles(mol)
        descriptors['NumSaturatedRings'] = Lipinski.NumSaturatedRings(mol)
        
        # 78-81. molecular property descriptors
        descriptors['FractionCSP3'] = Lipinski.FractionCSP3(mol)
        descriptors['NumRotatableBonds'] = Lipinski.NumRotatableBonds(mol)
        descriptors['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
        descriptors['NumHDonors'] = Lipinski.NumHDonors(mol)
        
        # 82-149. functional group descriptors
        descriptors['fr_Al_COO'] = fr_Al_COO(mol)
        descriptors['fr_Al_OH'] = fr_Al_OH(mol)
        descriptors['fr_Al_OH_noTert'] = fr_Al_OH_noTert(mol)
        descriptors['fr_ArN'] = fr_ArN(mol)
        descriptors['fr_Ar_COO'] = fr_Ar_COO(mol)
        descriptors['fr_Ar_N'] = fr_Ar_N(mol)
        descriptors['fr_Ar_NH'] = fr_Ar_NH(mol)
        descriptors['fr_Ar_OH'] = fr_Ar_OH(mol)
        descriptors['fr_COO'] = fr_COO(mol)
        descriptors['fr_C_O'] = fr_C_O(mol)
        descriptors['fr_C_O_noCOO'] = fr_C_O_noCOO(mol)
        descriptors['fr_C_S'] = fr_C_S(mol)
        descriptors['fr_HOCCN'] = fr_HOCCN(mol)
        descriptors['fr_Imine'] = fr_Imine(mol)
        descriptors['fr_NH0'] = fr_NH0(mol)
        descriptors['fr_NH1'] = fr_NH1(mol)
        descriptors['fr_NH2'] = fr_NH2(mol)
        descriptors['fr_N_O'] = fr_N_O(mol)
        descriptors['fr_Nhpyrrole'] = fr_Nhpyrrole(mol)
        descriptors['fr_SH'] = fr_SH(mol)
        descriptors['fr_aldehyde'] = fr_aldehyde(mol)
        descriptors['fr_alkyl_carbamate'] = fr_alkyl_carbamate(mol)
        descriptors['fr_alkyl_halide'] = fr_alkyl_halide(mol)
        descriptors['fr_allylic_oxid'] = fr_allylic_oxid(mol)
        descriptors['fr_amide'] = fr_amide(mol)
        descriptors['fr_amidine'] = fr_amidine(mol)
        descriptors['fr_aryl_methyl'] = fr_aryl_methyl(mol)
        descriptors['fr_azide'] = fr_azide(mol)
        descriptors['fr_azo'] = fr_azo(mol)
        descriptors['fr_barbitur'] = fr_barbitur(mol)
        descriptors['fr_benzene'] = fr_benzene(mol)
        descriptors['fr_benzodiazepine'] = fr_benzodiazepine(mol)
        descriptors['fr_bicyclic'] = fr_bicyclic(mol)
        descriptors['fr_diazo'] = fr_diazo(mol)
        descriptors['fr_dihydropyridine'] = fr_dihydropyridine(mol)
        descriptors['fr_epoxide'] = fr_epoxide(mol)
        descriptors['fr_ester'] = fr_ester(mol)
        descriptors['fr_ether'] = fr_ether(mol)
        descriptors['fr_furan'] = fr_furan(mol)
        descriptors['fr_guanido'] = fr_guanido(mol)
        descriptors['fr_halogen'] = fr_halogen(mol)
        descriptors['fr_hdrzine'] = fr_hdrzine(mol)
        descriptors['fr_hdrzone'] = fr_hdrzone(mol)
        descriptors['fr_imidazole'] = fr_imidazole(mol)
        descriptors['fr_imid'] = fr_imid(mol)
        descriptors['fr_isocyan'] = fr_isocyan(mol)
        descriptors['fr_isothiocyan'] = fr_isothiocyan(mol)
        descriptors['fr_ketone'] = fr_ketone(mol)
        descriptors['fr_lactam'] = fr_lactam(mol)
        descriptors['fr_lactone'] = fr_lactone(mol)
        descriptors['fr_methoxy'] = fr_methoxy(mol)
        descriptors['fr_morpholine'] = fr_morpholine(mol)
        descriptors['fr_nitrile'] = fr_nitrile(mol)
        descriptors['fr_nitro'] = fr_nitro(mol)
        descriptors['fr_nitro_arom'] = fr_nitro_arom(mol)
        descriptors['fr_nitroso'] = fr_nitroso(mol)
        descriptors['fr_oxazole'] = fr_oxazole(mol)
        descriptors['fr_oxime'] = fr_oxime(mol)
        descriptors['fr_phenol'] = fr_phenol(mol)
        descriptors['fr_phos_acid'] = fr_phos_acid(mol)
        descriptors['fr_phos_ester'] = fr_phos_ester(mol)
        descriptors['fr_priamide'] = fr_priamide(mol)
        descriptors['fr_prisulfonamd'] = fr_prisulfonamd(mol)
        descriptors['fr_pyridine'] = fr_pyridine(mol)
        descriptors['fr_quatN'] = fr_quatN(mol)
        descriptors['fr_sulfide'] = fr_sulfide(mol)
        descriptors['fr_sulfonamd'] = fr_sulfonamd(mol)
        descriptors['fr_sulfone'] = fr_sulfone(mol)
        descriptors['fr_term_acetylene'] = fr_term_acetylene(mol)
        descriptors['fr_tetrazole'] = fr_tetrazole(mol)
        descriptors['fr_thiazole'] = fr_thiazole(mol)
        descriptors['fr_thiocyan'] = fr_thiocyan(mol)
        descriptors['fr_thiophene'] = fr_thiophene(mol)
        descriptors['fr_urea'] = fr_urea(mol)
        
        # 150-156. molecular weight and atom count descriptors
        descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
        descriptors['HeavyAtomCount'] = Lipinski.HeavyAtomCount(mol)
        descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['NumHeteroatoms'] = Lipinski.NumHeteroatoms(mol)
        descriptors['MolLogP'] = Crippen.MolLogP(mol)
        descriptors['MolMR'] = Crippen.MolMR(mol)

        return descriptors

    except Exception as e:
        return None
