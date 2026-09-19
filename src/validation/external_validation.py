# -*- coding: utf-8 -*-
"""
Unified external validation script for six MR-TADF prediction models.

Models handled in one workflow:
1) PL-peak expensive-feature model: final_model.pkl
2) PL-peak low-cost model: final_pl_peak_lowcost_qspr.pkl
3) PLQY expensive-feature model: plqy_rf_final_model.pkl
4) PLQY low-cost model: final_plqy_lowcost_rf_model.pkl
5) FWHM low-cost + expensive-feature model: final_fwhm_lowcost_plus_expensive_RF_model.pkl
6) FWHM low-cost model: final_fwhm_lowcost_RF_model.pkl

Outputs are saved into one folder:
external_validation_outputs/
    external_validation_predictions.xlsx
    external_validation_metrics.csv
    figures/*.png and *.pdf

Run:
    python external_validation_all_models.py

Put this script in the same folder as the model files and test molecules.xlsx,
or edit MODEL_DIR / TEST_FILE below.
"""

import os
import re
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys, PatternFingerprint, RDKFingerprint


# =========================================================
# 0. Path settings
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
TEST_FILE = PROJECT_ROOT / "data" / "test molecules.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "external_validation"
FIG_DIR = OUTPUT_DIR / "figures"

MODEL_FILES = {
    "pl_peak_expensive": MODEL_DIR / "final_pl_peak_expensive.pkl",
    "pl_peak_lowcost": MODEL_DIR / "final_pl_peak_lowcost.pkl",
    "plqy_expensive": MODEL_DIR / "final_plqy_rf_expensive.pkl",
    "plqy_lowcost": MODEL_DIR / "final_plqy_rf_lowcost.pkl",
    "fwhm_expensive": MODEL_DIR / "final_fwhm_rf_expensive.pkl",
    "fwhm_lowcost": MODEL_DIR / "final_fwhm_rf_lowcost.pkl",
}

SMILES_COL = "Smiles"
MOLECULE_COL = "Molecules"
ENV_COL = "Solution"
PL_PEAK_COL = "PL-peak"
PLQY_COL = "PLQY"
FWHM_COL = "FWHM"

PLQY_THRESHOLD = 80.0
FWHM_THRESHOLD = 30.0
DPI = 600


# =========================================================
# 1. Utilities
# =========================================================
def ensure_closed_axes(ax, lw=1.25):
    """Use closed rectangular axes frame and unified tick style."""
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(direction="out", length=4.5, width=1.0)
    return ax


def set_global_plot_style():
    """Set publication-sized fonts consistently for every generated figure."""
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "axes.titleweight": "normal",
        "axes.labelpad": 8,
        "axes.titlepad": 10,
        "axes.linewidth": 1.25,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
    })


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_descriptor(func, mol, default=0.0):
    try:
        v = func(mol)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return default
        return float(v)
    except Exception:
        return default


def bitvect_to_dict(prefix, bitvect, n_bits=None):
    if n_bits is None:
        n_bits = bitvect.GetNumBits()
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return {f"{prefix}_{i}": int(arr[i]) for i in range(n_bits)}


def get_positive_probability(model, X):
    """Return probability for class 1 when available."""
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        return np.asarray(pred, dtype=float)
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model", None), "classes_", None)
    if classes is not None and 1 in list(classes):
        idx = list(classes).index(1)
    else:
        idx = proba.shape[1] - 1
    return proba[:, idx]


def align_features(X_all, feature_names):
    """Add missing model columns as 0, remove extra columns, keep training order."""
    X = X_all.copy()
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, missing




def patch_sklearn_compatibility(estimator):
    """Patch sklearn 1.7 pickles loaded under newer sklearn versions."""
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        SimpleImputer = None

    def visit(obj):
        if obj is None:
            return
        if SimpleImputer is not None and isinstance(obj, SimpleImputer):
            if not hasattr(obj, "_fill_dtype"):
                if hasattr(obj, "_fit_dtype"):
                    obj._fill_dtype = obj._fit_dtype
                else:
                    obj._fill_dtype = None
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                visit(step)
        if hasattr(obj, "estimators_"):
            for child in getattr(obj, "estimators_", [])[:]:
                visit(child)
    visit(estimator)
    return estimator

def model_and_features(model_obj):
    """Extract estimator and feature order from either a dict bundle or a plain estimator."""
    if isinstance(model_obj, dict):
        model = model_obj["model"]
        if "selected_features" in model_obj and model_obj["selected_features"] is not None:
            features = list(model_obj["selected_features"])
        elif hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            features = list(model_obj.get("all_feature_columns", []))
        metadata = model_obj.get("metadata", {}) or {}
    else:
        model = model_obj
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            raise ValueError("Plain model object has no feature_names_in_; cannot align features safely.")
        metadata = {}
    return model, [str(f) for f in features], metadata


# =========================================================
# 2. Molecule descriptors and fingerprints
# =========================================================
def aromatic_ring_system_count(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    arom_rings = []
    for r in rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            arom_rings.append(r)
    if not arom_rings:
        return 0
    visited = [False] * len(arom_rings)
    n_systems = 0
    for i in range(len(arom_rings)):
        if visited[i]:
            continue
        n_systems += 1
        stack = [i]
        visited[i] = True
        while stack:
            j = stack.pop()
            for k in range(len(arom_rings)):
                if not visited[k] and len(arom_rings[j].intersection(arom_rings[k])) >= 1:
                    visited[k] = True
                    stack.append(k)
    return n_systems


def fused_ring_pair_count(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    count = 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i].intersection(rings[j])) >= 2:
                count += 1
    return count


def substructure_flag(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return int(mol.HasSubstructMatch(patt))


def featurize_one(smiles, core_smiles=None):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    features = {}
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    # Common RDKit descriptors
    descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "ExactMolWt": Descriptors.ExactMolWt,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "TPSA": Descriptors.TPSA,
        "MolLogP": Descriptors.MolLogP,
        "MolMR": Descriptors.MolMR,
        "BertzCT": Descriptors.BertzCT,
        "BalabanJ": Descriptors.BalabanJ,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumHDonors": Descriptors.NumHDonors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "RingCount": Descriptors.RingCount,
        "FractionCSP3": Descriptors.FractionCSP3,
        "Kappa1": Descriptors.Kappa1,
        "Kappa2": Descriptors.Kappa2,
        "Kappa3": Descriptors.Kappa3,
        "Chi0": Descriptors.Chi0,
        "Chi1": Descriptors.Chi1,
        "Chi0n": Descriptors.Chi0n,
        "Chi1n": Descriptors.Chi1n,
        "Chi2n": Descriptors.Chi2n,
        "Chi3n": Descriptors.Chi3n,
        "Chi4n": Descriptors.Chi4n,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
    }
    for name, func in descriptor_funcs.items():
        features[name] = safe_descriptor(func, mol)

    # Bond and ring descriptors
    features["NumConjugatedBonds"] = sum(int(b.GetIsConjugated()) for b in mol.GetBonds())
    features["NumAromaticBonds"] = sum(int(b.GetIsAromatic()) for b in mol.GetBonds())
    features["NumSingleBonds"] = sum(int(b.GetBondType() == Chem.rdchem.BondType.SINGLE) for b in mol.GetBonds())
    features["NumDoubleBonds"] = sum(int(b.GetBondType() == Chem.rdchem.BondType.DOUBLE) for b in mol.GetBonds())
    features["ConjugatedBondFrac"] = features["NumConjugatedBonds"] / (n_bonds + 1e-6)
    features["AromaticBondFrac"] = features["NumAromaticBonds"] / (n_bonds + 1e-6)
    features["RotatableBondFrac"] = features["NumRotatableBonds"] / (n_bonds + 1e-6)
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    features["LargestRingSize"] = max(ring_sizes) if ring_sizes else 0
    features["FusedRingPairCount"] = fused_ring_pair_count(mol)
    features["AromaticRingSystemCount"] = aromatic_ring_system_count(mol)
    features["BridgeheadAtomCount"] = safe_descriptor(rdMolDescriptors.CalcNumBridgeheadAtoms, mol)
    features["SpiroAtomCount"] = safe_descriptor(rdMolDescriptors.CalcNumSpiroAtoms, mol)

    # Aromatic/non-aromatic single-bond counts
    aa_single = 0
    an_single = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.rdchem.BondType.SINGLE:
            a1 = b.GetBeginAtom().GetIsAromatic()
            a2 = b.GetEndAtom().GetIsAromatic()
            if a1 and a2:
                aa_single += 1
            elif a1 != a2:
                an_single += 1
    features["AromaticAromaticSingleBondCount"] = aa_single
    features["AromaticNonAromaticSingleBondCount"] = an_single

    # Atom counts and fractions
    for sym in ["B", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]:
        features[f"Count_{sym}"] = atom_symbols.count(sym)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    sp2_atoms = sum(1 for a in mol.GetAtoms() if a.GetHybridization().name in ["SP2", "SP"])
    hetero_count = sum(1 for s in atom_symbols if s != "C" and s != "H")
    features["AromaticAtomFrac"] = aromatic_atoms / (n_atoms + 1e-6)
    features["SP2AtomFrac"] = sp2_atoms / (n_atoms + 1e-6)
    features["HeteroAtomFrac"] = hetero_count / (n_atoms + 1e-6)
    for sym in ["B", "N", "O", "S", "F"]:
        features[f"{sym}_frac"] = features[f"Count_{sym}"] / (n_atoms + 1e-6)
    features["has_B"] = int(features["Count_B"] > 0)
    features["has_BN_proxy"] = int(features["Count_B"] > 0 and features["Count_N"] > 0)
    features["has_BO_proxy"] = int(features["Count_B"] > 0 and features["Count_O"] > 0)
    features["has_BS_proxy"] = int(features["Count_B"] > 0 and features["Count_S"] > 0)
    features["has_BNO_proxy"] = int(features["Count_B"] > 0 and features["Count_N"] > 0 and features["Count_O"] > 0)
    features["has_BOS_proxy"] = int(features["Count_B"] > 0 and features["Count_O"] > 0 and features["Count_S"] > 0)
    features["has_BNS_proxy"] = int(features["Count_B"] > 0 and features["Count_N"] > 0 and features["Count_S"] > 0)

    # Functional-group flags: simple, robust SMARTS approximations
    fg_smarts = {
        "fg_methoxy_like": "[OX2H0][CH3]",
        "fg_alkoxy_aromatic": "[a][OX2][#6]",
        "fg_tertbutyl_like": "[C;X4]([CH3])([CH3])[CH3]",
        "fg_CF3": "[CX4](F)(F)F",
        "fg_cyano": "C#N",
        "fg_carbonyl": "[CX3]=[OX1]",
        "fg_sulfone_sulfoxide": "[SX3,SX4](=O)",
        "fg_diphenylamine_like": "cN(c)c",
        "fg_triphenylamine_like": "cN(c)c",
        "fg_carbazole_like": "c1ccc2c(c1)[nH,nX3]c1ccccc12",
        "fg_phenyl": "c1ccccc1",
        "fg_fluoro_aromatic": "[a][F]",
    }
    for name, smarts in fg_smarts.items():
        features[name] = substructure_flag(mol, smarts)

    # Core/motif features from model metadata
    if core_smiles:
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        group_sims = {"A": [], "B": [], "C": []}
        group_counts = {"A": 0, "B": 0, "C": 0}
        group_matches = {"A": 0, "B": 0, "C": 0}
        for label, smi in core_smiles.items():
            label = str(label)
            group = label[0] if label else "X"
            core_mol = Chem.MolFromSmiles(str(smi))
            match = 0
            count = 0
            sim = 0.0
            if core_mol is not None:
                try:
                    match = int(mol.HasSubstructMatch(core_mol))
                    count = len(mol.GetSubstructMatches(core_mol))
                except Exception:
                    match, count = 0, 0
                try:
                    core_fp = AllChem.GetMorganFingerprintAsBitVect(core_mol, radius=2, nBits=1024)
                    sim = float(DataStructs.TanimotoSimilarity(mol_fp, core_fp))
                except Exception:
                    sim = 0.0
            features[f"motif_match_{label}"] = match
            features[f"motif_count_{label}"] = count
            features[f"motif_sim_{label}"] = sim
            if group in group_sims:
                group_sims[group].append(sim)
                group_counts[group] += count
                group_matches[group] += match
        for group in ["A", "B", "C"]:
            sims = group_sims[group]
            features[f"motif_max_sim_{group}"] = max(sims) if sims else 0.0
            features[f"motif_any_{group}"] = int(group_matches[group] > 0)
            features[f"motif_total_count_{group}"] = group_counts[group]
            features[f"motif_n_labels_{group}"] = group_matches[group]

    # Fingerprints: names used by your low-cost models
    features.update(bitvect_to_dict("MORGAN2", AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), 1024))
    features.update(bitvect_to_dict("MORGAN3", AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024), 1024))
    features.update(bitvect_to_dict("RDKFP", RDKFingerprint(mol, fpSize=2048), 2048))
    features.update(bitvect_to_dict("PATTERN", PatternFingerprint(mol, fpSize=2048), 2048))
    maccs = MACCSkeys.GenMACCSKeys(mol)
    features.update(bitvect_to_dict("MACCS", maccs, maccs.GetNumBits()))

    # Old expensive PL-peak model used FP_0..FP_511 Morgan radius=2 bits
    features.update(bitvect_to_dict("FP", AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512), 512))
    return features


def build_feature_matrix(df, model_metadata_list=None):
    # Combine all core libraries from model metadata so motif features are available for every model.
    core_smiles = {}
    if model_metadata_list:
        for metadata in model_metadata_list:
            cs = metadata.get("core_fragment_smiles") if isinstance(metadata, dict) else None
            if isinstance(cs, dict):
                core_smiles.update(cs)
            elif isinstance(cs, list):
                # Accept list of tuples/dicts if future model metadata stores it this way.
                for item in cs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        core_smiles[str(item[0])] = str(item[1])
                    elif isinstance(item, dict) and "label" in item and "smiles" in item:
                        core_smiles[str(item["label"])] = str(item["smiles"])

    feature_rows = []
    valid_indices = []
    invalid_rows = []
    for idx, smi in enumerate(df[SMILES_COL].astype(str)):
        feat = featurize_one(smi, core_smiles=core_smiles)
        if feat is None:
            invalid_rows.append((idx, smi))
        else:
            feature_rows.append(feat)
            valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("No valid SMILES can be parsed by RDKit.")

    df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)
    X = pd.DataFrame(feature_rows).reset_index(drop=True)

    # Environment one-hot columns exactly like training metadata columns.
    if ENV_COL not in df_valid.columns:
        df_valid[ENV_COL] = "Unknown"
    env = df_valid[ENV_COL].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "NaN": "Unknown"})
    df_valid[ENV_COL] = env
    env_dum = pd.get_dummies(env, prefix=ENV_COL, dtype=int)
    X = pd.concat([X, env_dum.reset_index(drop=True)], axis=1)

    # Expensive / electronic features with aliases.
    def get_col_value(candidates):
        for c in candidates:
            if c in df_valid.columns:
                return pd.to_numeric(df_valid[c], errors="coerce")
        return pd.Series([np.nan] * len(df_valid))

    X["S1"] = get_col_value(["S1"])
    X["T1"] = get_col_value(["T1"])
    X["dEST"] = get_col_value(["dEST", "∆EST", "ΔEST", "DEst", "DeltaEST"])
    X["∆EST"] = X["dEST"]
    X["ΔEST"] = X["dEST"]
    X["f"] = get_col_value(["f", "oscillator strength", "Oscillator strength"])
    X["taup"] = get_col_value(["taup", "τp", "tp", "Taup"])
    X["τp"] = X["taup"]
    X["HOMO"] = get_col_value(["HOMO"])
    X["LUMO"] = get_col_value(["LUMO"])
    X["HL_gap"] = X["LUMO"] - X["HOMO"]
    X["S1_T1_gap"] = X["S1"] - X["T1"]
    X["orbital_center"] = (X["HOMO"] + X["LUMO"]) / 2.0
    X["f_div_dE"] = X["f"] / (X["dEST"] + 1e-6)
    X["f_times_dE"] = X["f"] * X["dEST"]
    X["f_div_dEST"] = X["f_div_dE"]
    X["f_times_dEST"] = X["f_times_dE"]
    X["log_taup"] = np.log10(X["taup"] + 1e-6)

    return df_valid, X, invalid_rows


# =========================================================
# 3. Prediction
# =========================================================
def load_inputs_and_models():
    missing = []
    if not TEST_FILE.exists():
        missing.append(str(TEST_FILE))
    for name, path in MODEL_FILES.items():
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    df = pd.read_excel(TEST_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    if SMILES_COL not in df.columns:
        raise ValueError(f"Missing required SMILES column: {SMILES_COL}")
    if MOLECULE_COL not in df.columns:
        df[MOLECULE_COL] = [f"M{i+1}" for i in range(len(df))]
    if ENV_COL not in df.columns:
        df[ENV_COL] = "Unknown"

    model_objects = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    # Compatibility patch for models pickled with a slightly different sklearn version.
    for _name, _obj in model_objects.items():
        if isinstance(_obj, dict) and "model" in _obj:
            _obj["model"] = patch_sklearn_compatibility(_obj["model"])
        else:
            model_objects[_name] = patch_sklearn_compatibility(_obj)
    metadata_list = []
    for obj in model_objects.values():
        if isinstance(obj, dict):
            metadata_list.append(obj.get("metadata", {}) or {})
    return df, model_objects, metadata_list


def predict_all(df_valid, X_all, model_objects):
    result = df_valid.copy()
    missing_feature_report = []

    # PL-peak expensive
    model, features, metadata = model_and_features(model_objects["pl_peak_expensive"])
    X, missing = align_features(X_all, features)
    result["Pred_PL_peak_expensive_nm"] = model.predict(X)
    missing_feature_report.append({"model": "PL-peak expensive", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # PL-peak low-cost
    model, features, metadata = model_and_features(model_objects["pl_peak_lowcost"])
    X, missing = align_features(X_all, features)
    result["Pred_PL_peak_lowcost_nm"] = model.predict(X)
    missing_feature_report.append({"model": "PL-peak low-cost", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # PLQY expensive probability of high PLQY
    model, features, metadata = model_and_features(model_objects["plqy_expensive"])
    X, missing = align_features(X_all, features)
    result["Pred_PLQY_expensive_high_prob"] = get_positive_probability(model, X)
    missing_feature_report.append({"model": "PLQY expensive", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # PLQY low-cost probability of high PLQY
    model, features, metadata = model_and_features(model_objects["plqy_lowcost"])
    X, missing = align_features(X_all, features)
    result["Pred_PLQY_lowcost_high_prob"] = get_positive_probability(model, X)
    missing_feature_report.append({"model": "PLQY low-cost", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # FWHM expensive probability of narrow FWHM
    model, features, metadata = model_and_features(model_objects["fwhm_expensive"])
    X, missing = align_features(X_all, features)
    result["Pred_FWHM_expensive_narrow_prob"] = get_positive_probability(model, X)
    missing_feature_report.append({"model": "FWHM low-cost + expensive", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # FWHM low-cost probability of narrow FWHM
    model, features, metadata = model_and_features(model_objects["fwhm_lowcost"])
    X, missing = align_features(X_all, features)
    result["Pred_FWHM_lowcost_narrow_prob"] = get_positive_probability(model, X)
    missing_feature_report.append({"model": "FWHM low-cost", "n_missing_features_added_as_0": len(missing), "missing_features": "; ".join(missing[:80])})

    # Derived errors and labels
    if PL_PEAK_COL in result.columns:
        result[PL_PEAK_COL] = pd.to_numeric(result[PL_PEAK_COL], errors="coerce")
        result["Error_PL_peak_expensive_nm"] = result["Pred_PL_peak_expensive_nm"] - result[PL_PEAK_COL]
        result["Error_PL_peak_lowcost_nm"] = result["Pred_PL_peak_lowcost_nm"] - result[PL_PEAK_COL]
    if PLQY_COL in result.columns:
        result[PLQY_COL] = pd.to_numeric(result[PLQY_COL], errors="coerce")
        result["True_PLQY_high_label"] = (result[PLQY_COL] >= PLQY_THRESHOLD).astype("Int64")
        result["Pred_PLQY_expensive_label_0p5"] = (result["Pred_PLQY_expensive_high_prob"] >= 0.5).astype(int)
        result["Pred_PLQY_lowcost_label_0p5"] = (result["Pred_PLQY_lowcost_high_prob"] >= 0.5).astype(int)
    if FWHM_COL in result.columns:
        # IMPORTANT:
        # Missing experimental FWHM values must NOT be converted to Broad FWHM.
        # In pandas, the expression (NaN <= 30) returns False. If we directly cast
        # that boolean series to int, missing FWHM would be incorrectly labeled as 0.
        # Therefore, generate the FWHM ground-truth label only for rows with observed
        # experimental FWHM values; keep missing rows as <NA>.
        result[FWHM_COL] = pd.to_numeric(result[FWHM_COL], errors="coerce")
        result["FWHM_observed"] = result[FWHM_COL].notna()
        result["FWHM_included_in_metrics"] = result["FWHM_observed"]

        true_fwhm_label = pd.Series(pd.NA, index=result.index, dtype="Int64")
        observed_mask = result["FWHM_observed"]
        true_fwhm_label.loc[observed_mask] = (
            result.loc[observed_mask, FWHM_COL] <= FWHM_THRESHOLD
        ).astype(int).values
        result["True_FWHM_narrow_label"] = true_fwhm_label

        # Prediction labels are still kept for all molecules because the models can
        # predict every valid SMILES. They are only evaluated where True_FWHM_narrow_label
        # is not missing.
        result["Pred_FWHM_expensive_label_0p5"] = (result["Pred_FWHM_expensive_narrow_prob"] >= 0.5).astype(int)
        result["Pred_FWHM_lowcost_label_0p5"] = (result["Pred_FWHM_lowcost_narrow_prob"] >= 0.5).astype(int)

    return result, pd.DataFrame(missing_feature_report)


# =========================================================
# 4. Metrics
# =========================================================
def regression_metrics(df, y_col, pred_col, model_label):
    tmp = df[[y_col, pred_col]].dropna()
    if tmp.empty:
        return {"model": model_label, "task": "PL-peak", "n": 0}
    y = tmp[y_col].values
    p = tmp[pred_col].values
    return {
        "model": model_label,
        "task": "PL-peak regression",
        "n": len(tmp),
        "MAE_nm": mean_absolute_error(y, p),
        "RMSE_nm": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p) if len(tmp) >= 2 else np.nan,
        "Mean_bias_nm": float(np.mean(p - y)),
    }


def classification_metrics(df, y_col, prob_col, model_label, threshold=0.5):
    tmp = df[[y_col, prob_col]].dropna()
    if tmp.empty:
        return {"model": model_label, "task": "classification", "n": 0}
    y = tmp[y_col].astype(int).values
    prob = tmp[prob_col].astype(float).values
    pred = (prob >= threshold).astype(int)
    out = {
        "model": model_label,
        "task": "classification",
        "n": len(tmp),
        "threshold": threshold,
        "Accuracy": accuracy_score(y, pred),
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }
    try:
        out["ROC_AUC"] = roc_auc_score(y, prob)
    except Exception:
        out["ROC_AUC"] = np.nan
    return out


def build_metrics_table(result):
    rows = []
    if PL_PEAK_COL in result.columns:
        rows.append(regression_metrics(result, PL_PEAK_COL, "Pred_PL_peak_expensive_nm", "PL-peak expensive"))
        rows.append(regression_metrics(result, PL_PEAK_COL, "Pred_PL_peak_lowcost_nm", "PL-peak low-cost"))
    if "True_PLQY_high_label" in result.columns:
        rows.append(classification_metrics(result, "True_PLQY_high_label", "Pred_PLQY_expensive_high_prob", "PLQY expensive"))
        rows.append(classification_metrics(result, "True_PLQY_high_label", "Pred_PLQY_lowcost_high_prob", "PLQY low-cost"))
    if "True_FWHM_narrow_label" in result.columns:
        # classification_metrics() drops rows with missing True_FWHM_narrow_label,
        # so FWHM metrics are calculated only on molecules with observed FWHM.
        rows.append(classification_metrics(result, "True_FWHM_narrow_label", "Pred_FWHM_expensive_narrow_prob", "FWHM low-cost + expensive"))
        rows.append(classification_metrics(result, "True_FWHM_narrow_label", "Pred_FWHM_lowcost_narrow_prob", "FWHM low-cost"))
    return pd.DataFrame(rows)


def build_fwhm_inclusion_table(result):
    """Record which molecules are included in FWHM external-validation metrics."""
    base_cols = [c for c in [
        MOLECULE_COL, SMILES_COL, FWHM_COL,
        "FWHM_observed", "FWHM_included_in_metrics", "True_FWHM_narrow_label",
        "Pred_FWHM_expensive_narrow_prob", "Pred_FWHM_expensive_label_0p5",
        "Pred_FWHM_lowcost_narrow_prob", "Pred_FWHM_lowcost_label_0p5",
    ] if c in result.columns]
    if not base_cols:
        return pd.DataFrame()
    out = result[base_cols].copy()
    if "FWHM_included_in_metrics" in out.columns:
        out["FWHM_exclusion_reason"] = np.where(
            out["FWHM_included_in_metrics"],
            "included",
            "missing experimental FWHM"
        )
    return out


# =========================================================
# 5. Plotting
# =========================================================
COLOR_EXPENSIVE = "#D55E5E"   # muted red, consistent with previous train/expensive style
COLOR_LOWCOST = "#2C7FB8"     # paper-style blue
COLOR_TRUE = "#4D4D4D"
COLOR_CORRECT = "#2E8B57"
COLOR_WRONG = "#B22222"
COLOR_GREY = "#5A5A5A"


def save_fig(fig, filename_base):
    png = FIG_DIR / f"{filename_base}.png"
    pdf = FIG_DIR / f"{filename_base}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=DPI)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def class_text(value, positive_name, negative_name):
    try:
        return positive_name if int(value) == 1 else negative_name
    except Exception:
        return "NA"


def correctness_text(true_value, pred_value):
    try:
        return "correct" if int(true_value) == int(pred_value) else "wrong"
    except Exception:
        return "NA"


def short_molecule_labels(values, max_len=14):
    labels = []
    for v in values:
        s = str(v)
        if max_len is None:
            labels.append(s)
        else:
            labels.append(s if len(s) <= max_len else s[:max_len - 1] + "…")
    return labels


def plot_pl_peak_scatter(df, pred_col, title, filename_base):
    tmp = df[[MOLECULE_COL, PL_PEAK_COL, pred_col]].dropna().copy()
    if tmp.empty:
        return
    x = tmp[PL_PEAK_COL].astype(float).values
    y = tmp[pred_col].astype(float).values
    min_v = min(x.min(), y.min()) - 8
    max_v = max(x.max(), y.max()) + 8

    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    r2 = r2_score(x, y) if len(x) >= 2 else np.nan
    bias = np.mean(y - x)

    color = COLOR_EXPENSIVE if "expensive" in pred_col.lower() else COLOR_LOWCOST
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(
        x, y,
        s=58, alpha=0.84, color=color,
        edgecolor="white", linewidth=0.55,
        label="External molecules"
    )
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color=COLOR_GREY, linewidth=1.35, label="Ideal fit")
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min_v, max_v, 200)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=1.9, color="black", label="Linear fit")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental PL-peak / nm")
    ax.set_ylabel("Predicted PL-peak / nm")
    ax.set_title(title, pad=9)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f} nm\nRMSE = {rmse:.2f} nm\nMean bias = {bias:.2f} nm",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=14, loc="lower right")
    ensure_closed_axes(ax)
    fig.tight_layout()
    save_fig(fig, filename_base)


def plot_probability_vs_experiment(
    df, exp_col, prob_col, true_col, title, xlabel, ylabel, filename_base,
    exp_threshold=None, prob_threshold=0.5, positive_text="Positive", negative_text="Negative"
):
    tmp = df[[MOLECULE_COL, exp_col, prob_col, true_col]].dropna().copy()
    if tmp.empty:
        return
    tmp[true_col] = tmp[true_col].astype(int)
    tmp["pred_label"] = (tmp[prob_col].astype(float) >= prob_threshold).astype(int)
    tmp["is_correct"] = tmp["pred_label"] == tmp[true_col]

    neg_correct = (tmp[true_col] == 0) & tmp["is_correct"]
    pos_correct = (tmp[true_col] == 1) & tmp["is_correct"]
    neg_wrong = (tmp[true_col] == 0) & (~tmp["is_correct"])
    pos_wrong = (tmp[true_col] == 1) & (~tmp["is_correct"])

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.scatter(tmp.loc[neg_correct, exp_col], tmp.loc[neg_correct, prob_col], s=58, marker="o",
               color=COLOR_LOWCOST, alpha=0.82, edgecolor="black", linewidth=0.65, label=f"Actual {negative_text}, correct")
    ax.scatter(tmp.loc[pos_correct, exp_col], tmp.loc[pos_correct, prob_col], s=66, marker="^",
               color=COLOR_EXPENSIVE, alpha=0.86, edgecolor="black", linewidth=0.65, label=f"Actual {positive_text}, correct")
    ax.scatter(tmp.loc[neg_wrong, exp_col], tmp.loc[neg_wrong, prob_col], s=82, marker="X",
               color=COLOR_LOWCOST, alpha=0.92, edgecolor=COLOR_WRONG, linewidth=1.0, label=f"Actual {negative_text}, wrong")
    ax.scatter(tmp.loc[pos_wrong, exp_col], tmp.loc[pos_wrong, prob_col], s=88, marker="X",
               color=COLOR_EXPENSIVE, alpha=0.92, edgecolor=COLOR_WRONG, linewidth=1.0, label=f"Actual {positive_text}, wrong")

    if exp_threshold is not None:
        ax.axvline(exp_threshold, linestyle="--", color="black", linewidth=1.2)
    ax.axhline(prob_threshold, linestyle="--", color="black", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=9)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.28)

    y = tmp[true_col].astype(int).values
    prob = tmp[prob_col].astype(float).values
    pred = (prob >= prob_threshold).astype(int)
    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, prob)
    except Exception:
        auc = np.nan
    ax.text(
        0.04, 0.95,
        f"Accuracy = {acc:.3f}\nBalanced acc. = {bal:.3f}\nROC-AUC = {auc:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=14, loc="lower right")
    ensure_closed_axes(ax)
    fig.tight_layout()
    save_fig(fig, filename_base)


def plot_pl_peak_comparison(df):
    tmp = df[[MOLECULE_COL, PL_PEAK_COL, "Pred_PL_peak_expensive_nm", "Pred_PL_peak_lowcost_nm"]].dropna().copy()
    if tmp.empty:
        return
    tmp = tmp.reset_index(drop=True)
    x = np.arange(len(tmp))

    y_true = tmp[PL_PEAK_COL].astype(float).values
    y_exp = tmp["Pred_PL_peak_expensive_nm"].astype(float).values
    y_low = tmp["Pred_PL_peak_lowcost_nm"].astype(float).values
    mae_exp = mean_absolute_error(y_true, y_exp)
    rmse_exp = np.sqrt(mean_squared_error(y_true, y_exp))
    mae_low = mean_absolute_error(y_true, y_low)
    rmse_low = np.sqrt(mean_squared_error(y_true, y_low))

    # Use a slightly wider/taller canvas so full molecule names can be shown without truncation.
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.plot(x, y_true, marker="o", linewidth=1.9, color=COLOR_TRUE, label="Experimental")
    ax.plot(x, y_exp, marker="s", linewidth=1.7, color=COLOR_EXPENSIVE, label="Augmented model")
    ax.plot(x, y_low, marker="^", linewidth=1.7, color=COLOR_LOWCOST, label="Low-cost model")

    ax.set_xticks(x)
    # PL-peak figure: show the full molecule names on the x-axis.
    ax.set_xticklabels(
        short_molecule_labels(tmp[MOLECULE_COL].astype(str), max_len=None),
        rotation=55,
        ha="right",
        fontsize=14,
    )
    ax.set_ylabel("PL-peak / nm")
    ax.set_xlabel("External molecules")
    ax.set_title("Comparison of experimental and predicted PL-peak", pad=9)
    ax.grid(axis="y", linestyle="--", alpha=0.28)

    metrics_text = (
        f"Augmented MAE = {mae_exp:.1f} nm\n"
        f"Augmented RMSE = {rmse_exp:.1f} nm\n"
        f"Low-cost MAE = {mae_low:.1f} nm\n"
        f"Low-cost RMSE = {rmse_low:.1f} nm"
    )
    ax.text(
        0.012, 0.035,
        metrics_text,
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="gray", alpha=0.90),
    )

    ax.legend(frameon=False, fontsize=14, ncol=3, loc="upper right")
    ensure_closed_axes(ax)
    # Reserve extra bottom space for long molecule names.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.35)
    save_fig(fig, "Compare_PL_peak_expensive_vs_lowcost")


def plot_probability_overlay_comparison(
    df, exp_col, true_col, prob_exp, prob_low, title, xlabel, ylabel, filename_base,
    exp_threshold=None, positive_name="Positive", negative_name="Negative", prob_threshold=0.5
):
    """Overlay expensive-feature and low-cost probabilities in one classification plot.

    Each external molecule is shown with a distinct marker shape. Model type is
    encoded by color; wrong predictions are highlighted by a red outline.
    """
    keep = [MOLECULE_COL, exp_col, true_col, prob_exp, prob_low]
    tmp = df[keep].dropna().copy()
    if tmp.empty:
        return
    tmp[true_col] = tmp[true_col].astype(int)
    tmp = tmp.reset_index(drop=True)

    marker_cycle = [
        "o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8", "H", "d", "1", "2", "3", "4"
    ]

    x_base = tmp[exp_col].astype(float).values
    x_range = float(np.nanmax(x_base) - np.nanmin(x_base)) if len(x_base) > 1 else 1.0
    x_offset = max(x_range * 0.006, 0.12)

    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    y_true = tmp[true_col].astype(int).values
    p_exp = tmp[prob_exp].astype(float).values
    p_low = tmp[prob_low].astype(float).values
    pred_exp = (p_exp >= prob_threshold).astype(int)
    pred_low = (p_low >= prob_threshold).astype(int)

    for i, (_, row) in enumerate(tmp.iterrows()):
        marker = marker_cycle[i % len(marker_cycle)]
        true_label = int(row[true_col])
        px_exp = float(row[exp_col]) - x_offset
        px_low = float(row[exp_col]) + x_offset
        py_exp = float(row[prob_exp])
        py_low = float(row[prob_low])

        ok_exp = int(py_exp >= prob_threshold) == true_label
        ok_low = int(py_low >= prob_threshold) == true_label

        ax.scatter(
            px_exp, py_exp,
            marker=marker, s=76,
            color=COLOR_EXPENSIVE, alpha=0.88,
            edgecolor=("black" if ok_exp else COLOR_WRONG),
            linewidth=(0.65 if ok_exp else 1.9),
            zorder=3,
        )
        ax.scatter(
            px_low, py_low,
            marker=marker, s=76,
            color=COLOR_LOWCOST, alpha=0.88,
            edgecolor=("black" if ok_low else COLOR_WRONG),
            linewidth=(0.65 if ok_low else 1.9),
            zorder=3,
        )

    if exp_threshold is not None:
        ax.axvline(exp_threshold, linestyle="--", color="black", linewidth=1.2, zorder=1)
    ax.axhline(prob_threshold, linestyle="--", color="black", linewidth=1.2, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=9)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.28)

    acc_exp = accuracy_score(y_true, pred_exp)
    acc_low = accuracy_score(y_true, pred_low)
    metrics_text = (
        f"Augmented acc. = {acc_exp:.3f}\n"
        f"Low-cost acc. = {acc_low:.3f}"
    )
    ax.text(
        0.04, 0.055,
        metrics_text,
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="gray", alpha=0.90),
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_EXPENSIVE,
               markeredgecolor="black", markersize=8, label="Augmented model"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_LOWCOST,
               markeredgecolor="black", markersize=8, label="Low-cost model"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Correct prediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLOR_WRONG, markeredgewidth=1.8, markersize=8, label="Wrong prediction"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=14,
        loc="lower right",
        labelspacing=0.55,
        handletextpad=0.55,
    )
    ensure_closed_axes(ax)
    fig.tight_layout()
    save_fig(fig, filename_base)


def plot_probability_pair_comparison(
    df, exp_col, true_col, prob_exp, prob_low, title, ylabel, filename_base,
    positive_name="High", negative_name="Low", prob_threshold=0.5
):
    keep = [MOLECULE_COL, exp_col, true_col, prob_exp, prob_low]
    tmp = df[keep].dropna().copy()
    if tmp.empty:
        return
    tmp[true_col] = tmp[true_col].astype(int)
    tmp["pred_exp"] = (tmp[prob_exp].astype(float) >= prob_threshold).astype(int)
    tmp["pred_low"] = (tmp[prob_low].astype(float) >= prob_threshold).astype(int)
    tmp["correct_exp"] = tmp["pred_exp"] == tmp[true_col]
    tmp["correct_low"] = tmp["pred_low"] == tmp[true_col]
    tmp = tmp.reset_index(drop=True)

    x = np.arange(len(tmp))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12.4, 7.0))

    ax.bar(
        x - width / 2,
        tmp[prob_exp].astype(float),
        width=width,
        color=COLOR_EXPENSIVE,
        edgecolor=[COLOR_CORRECT if ok else COLOR_WRONG for ok in tmp["correct_exp"]],
        linewidth=[1.25 if ok else 2.2 for ok in tmp["correct_exp"]],
        label="Augmented model",
    )
    ax.bar(
        x + width / 2,
        tmp[prob_low].astype(float),
        width=width,
        color=COLOR_LOWCOST,
        edgecolor=[COLOR_CORRECT if ok else COLOR_WRONG for ok in tmp["correct_low"]],
        linewidth=[1.25 if ok else 2.2 for ok in tmp["correct_low"]],
        label="Low-cost model",
    )

    ax.axhline(prob_threshold, linestyle="--", color="black", linewidth=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_molecule_labels(tmp[MOLECULE_COL].astype(str), max_len=18), rotation=55, ha="right", fontsize=14.0)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("External molecules")
    ax.set_title(title, pad=9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", alpha=0.28)

    y_true = tmp[true_col].astype(int).values
    acc_exp = accuracy_score(y_true, tmp["pred_exp"].astype(int).values)
    acc_low = accuracy_score(y_true, tmp["pred_low"].astype(int).values)
    ax.text(
        0.012, 0.98,
        f"Augmented acc. = {acc_exp:.2f}\nLow-cost acc. = {acc_low:.2f}\nGreen outline: correct\nRed outline: wrong",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="gray", alpha=0.90),
    )

    ax.legend(frameon=False, fontsize=14, ncol=2, loc="upper right")
    ensure_closed_axes(ax)
    fig.tight_layout()
    save_fig(fig, filename_base)

def make_all_plots(result):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only the combined PL-peak comparison line plot.
    # The individual PL-peak fitting/scatter plots are intentionally not generated.
    if PL_PEAK_COL in result.columns:
        plot_pl_peak_comparison(result)

    # Classification plots: merge expensive-feature and low-cost models into one
    # probability-vs-experiment plot for each task.
    # The previous individual probability plots and bar-style comparison plots are removed.
    if PLQY_COL in result.columns and "True_PLQY_high_label" in result.columns:
        plot_probability_overlay_comparison(
            result,
            exp_col=PLQY_COL,
            true_col="True_PLQY_high_label",
            prob_exp="Pred_PLQY_expensive_high_prob",
            prob_low="Pred_PLQY_lowcost_high_prob",
            title="External validation of PLQY models",
            xlabel="Experimental PLQY / %",
            ylabel="Predicted probability of high PLQY",
            filename_base="Compare_PLQY_expensive_vs_lowcost",
            exp_threshold=PLQY_THRESHOLD,
            positive_name=f"PLQY ≥ {PLQY_THRESHOLD:.0f}%",
            negative_name=f"PLQY < {PLQY_THRESHOLD:.0f}%",
            prob_threshold=0.5,
        )

    if FWHM_COL in result.columns and "True_FWHM_narrow_label" in result.columns:
        plot_probability_overlay_comparison(
            result,
            exp_col=FWHM_COL,
            true_col="True_FWHM_narrow_label",
            prob_exp="Pred_FWHM_expensive_narrow_prob",
            prob_low="Pred_FWHM_lowcost_narrow_prob",
            title="External validation of FWHM models",
            xlabel="Experimental FWHM / nm",
            ylabel="Predicted probability of narrow FWHM",
            filename_base="Compare_FWHM_expensive_vs_lowcost",
            exp_threshold=FWHM_THRESHOLD,
            positive_name=f"FWHM ≤ {FWHM_THRESHOLD:.0f} nm",
            negative_name=f"FWHM > {FWHM_THRESHOLD:.0f} nm",
            prob_threshold=0.5,
        )

# =========================================================
# 6. Main
# =========================================================
def main():
    set_global_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old figures so removed plots do not remain from previous runs.
    for old_fig in list(FIG_DIR.glob("*.png")) + list(FIG_DIR.glob("*.pdf")):
        try:
            old_fig.unlink()
        except Exception:
            pass

    df, model_objects, metadata_list = load_inputs_and_models()
    df_valid, X_all, invalid_rows = build_feature_matrix(df, metadata_list)
    result, missing_feature_report = predict_all(df_valid, X_all, model_objects)
    metrics = build_metrics_table(result)
    fwhm_inclusion = build_fwhm_inclusion_table(result)
    make_all_plots(result)

    # Save outputs
    out_xlsx = OUTPUT_DIR / "external_validation_predictions.xlsx"
    out_metrics = OUTPUT_DIR / "external_validation_metrics.csv"
    out_missing = OUTPUT_DIR / "feature_alignment_report.csv"
    out_invalid = OUTPUT_DIR / "invalid_smiles_report.csv"
    out_fwhm_inclusion = OUTPUT_DIR / "fwhm_eval_inclusion.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="predictions", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        missing_feature_report.to_excel(writer, sheet_name="feature_alignment", index=False)
        fwhm_inclusion.to_excel(writer, sheet_name="fwhm_eval_inclusion", index=False)
        pd.DataFrame(invalid_rows, columns=["row_index_0_based", "smiles"]).to_excel(writer, sheet_name="invalid_smiles", index=False)
    metrics.to_csv(out_metrics, index=False)
    missing_feature_report.to_csv(out_missing, index=False)
    fwhm_inclusion.to_csv(out_fwhm_inclusion, index=False)
    pd.DataFrame(invalid_rows, columns=["row_index_0_based", "smiles"]).to_csv(out_invalid, index=False)

    print("=" * 72)
    print("External validation finished")
    print("=" * 72)
    print(f"Input molecules: {len(df)}")
    print(f"Valid SMILES predicted: {len(result)}")
    print(f"Invalid SMILES: {len(invalid_rows)}")
    if "FWHM_observed" in result.columns:
        print(f"Observed experimental FWHM used in FWHM metrics: {int(result['FWHM_observed'].sum())}")
        print(f"Missing experimental FWHM excluded from FWHM metrics: {int((~result['FWHM_observed']).sum())}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nMetrics:")
    with pd.option_context("display.max_columns", 30, "display.width", 180):
        print(metrics)
    print("\nSaved figures:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" -", p.name)
    print("\nSaved files:")
    print(" -", out_xlsx.name)
    print(" -", out_metrics.name)
    print(" -", out_missing.name)
    print(" -", out_fwhm_inclusion.name)
    print(" -", out_invalid.name)


if __name__ == "__main__":
    main()
