#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screen generated TADF candidates with low-cost PL-peak / PLQY / FWHM models.

Default criteria:
    430 <= predicted PL-peak <= 490 nm
    predicted high-PLQY label/probability corresponds to PLQY > 80%
    predicted narrow-FWHM label/probability corresponds to FWHM < 30 nm
    SA score < 6

Important:
    The uploaded model files are dictionaries, not bare estimators.
    This script always loads dict['model'] and reindexes features to dict['selected_features'].

Example:
    python screen_generated_candidates_lowcost_models.py \
        --input scaffold_based_generated_molecules.csv \
        --plpeak_model final_pl_peak_lowcost_qspr.pkl \
        --plqy_model final_plqy_lowcost_rf_model.pkl \
        --fwhm_model final_fwhm_lowcost_RF_model.pkl \
        --output_dir screened_lowcost_candidates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDConfig, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors, rdmolops
from rdkit.Chem import MACCSkeys
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT

# SA score from RDKit Contrib
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
except Exception:
    sascorer = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Basic utilities
# -----------------------------


def patch_sklearn_compatibility(estimator):
    """Patch a few sklearn pickle-version compatibility attributes.

    Some user models were trained with sklearn 1.7.x. Newer sklearn versions may
    expect SimpleImputer._fill_dtype, while the pickle only has _fit_dtype.
    """
    try:
        if hasattr(estimator, "_fit_dtype") and not hasattr(estimator, "_fill_dtype"):
            estimator._fill_dtype = estimator._fit_dtype
    except Exception:
        pass
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            patch_sklearn_compatibility(step)
    if hasattr(estimator, "steps"):
        for _, step in getattr(estimator, "steps", []):
            patch_sklearn_compatibility(step)
    return estimator

def load_model_dict(path: str, name: str) -> Dict:
    """Load a model file that should be a dict containing at least 'model'."""
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise TypeError(f"{name} model file is not a dict: {path}; got {type(obj)}")
    if "model" not in obj:
        raise KeyError(f"{name} model dict does not contain key 'model'. keys={list(obj.keys())}")
    if "selected_features" not in obj:
        # fall back to estimator feature_names_in_ if available
        model = obj["model"]
        if hasattr(model, "feature_names_in_"):
            obj["selected_features"] = list(model.feature_names_in_)
        else:
            raise KeyError(f"{name} model dict does not contain key 'selected_features'.")
    obj["selected_features"] = list(obj["selected_features"])
    obj["model"] = patch_sklearn_compatibility(obj["model"])
    return obj


def find_smiles_column(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Specified SMILES column not found: {user_col}")
        return user_col
    candidates = [
        "canonical_smiles", "SMILES", "smiles", "Smiles", "canonical_SMILES",
        "Can_SMILES", "can_smiles", "mol_smiles"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find SMILES column. Existing columns: {list(df.columns)}")


def canonicalize_smiles(smi: str) -> Tuple[Optional[str], Optional[Chem.Mol], Optional[str]]:
    if pd.isna(smi):
        return None, None, "SMILES is NaN"
    smi = str(smi).strip()
    if not smi:
        return None, None, "SMILES is empty"
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, None, "RDKit MolFromSmiles failed"
        can = Chem.MolToSmiles(mol, canonical=True)
        return can, mol, None
    except Exception as e:
        return None, None, f"RDKit exception: {e}"


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) else 0.0


def count_atoms(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)


def count_bonds_by_type(mol: Chem.Mol, bond_type: Chem.BondType) -> int:
    return sum(1 for b in mol.GetBonds() if b.GetBondType() == bond_type)


def largest_ring_size(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


def fused_ring_pair_count(mol: Chem.Mol) -> int:
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    count = 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i].intersection(rings[j])) >= 2:
                count += 1
    return count


def aromatic_ring_system_count(mol: Chem.Mol) -> int:
    """Approximate number of fused aromatic ring systems."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    aromatic_rings = []
    for r in rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            aromatic_rings.append(r)
    if not aromatic_rings:
        return 0
    parent = list(range(len(aromatic_rings)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(aromatic_rings)):
        for j in range(i + 1, len(aromatic_rings)):
            if len(aromatic_rings[i].intersection(aromatic_rings[j])) >= 2:
                union(i, j)
    return len({find(i) for i in range(len(aromatic_rings))})


def aromatic_single_bond_counts(mol: Chem.Mol) -> Tuple[int, int]:
    aromatic_aromatic = 0
    aromatic_nonaromatic = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE:
            a1 = b.GetBeginAtom().GetIsAromatic()
            a2 = b.GetEndAtom().GetIsAromatic()
            if a1 and a2:
                aromatic_aromatic += 1
            elif a1 or a2:
                aromatic_nonaromatic += 1
    return aromatic_aromatic, aromatic_nonaromatic


def substruct_count(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt, uniquify=True))


def has_substruct(mol: Chem.Mol, smarts: str) -> int:
    return int(substruct_count(mol, smarts) > 0)


# -----------------------------
# Fingerprints and motifs
# -----------------------------

def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def bit_value(fp, idx: int) -> int:
    if idx < 0 or idx >= fp.GetNumBits():
        return 0
    return int(fp.GetBit(idx))


def parse_bit_feature(name: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"^(MORGAN2|MORGAN3|RDKFP|PATTERN)_(\d+)$", name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def prepare_motif_mols(core_fragment_smiles: Dict[str, str]) -> Dict[str, Tuple[Chem.Mol, object]]:
    """Return motif mols plus precomputed Morgan fingerprints for similarity."""
    motif_mols = {}
    for label, smi in (core_fragment_smiles or {}).items():
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                motif_mols[str(label)] = (mol, morgan_fp(mol, radius=2, n_bits=2048))
        except Exception:
            pass
    return motif_mols


def motif_features(mol: Chem.Mol, motif_mols: Dict[str, Tuple[Chem.Mol, object]], needed: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Calculate motif similarity/match features.

    For speed, expensive substructure matching is only done if motif_match_*,
    motif_count_* or motif_any_* features are actually requested.
    """
    out: Dict[str, float] = {}
    needed_set = set(needed or [])
    mol_fp = morgan_fp(mol, radius=2, n_bits=2048)

    need_any = any(f.startswith("motif_any_") for f in needed_set)
    need_match_or_count = need_any or any(
        f.startswith("motif_match_") or f.startswith("motif_count_") for f in needed_set
    )
    groups = {"A": [], "B": [], "C": []}

    for label, pair in motif_mols.items():
        motif, motif_fp = pair
        label = str(label)

        sim = 0.0
        try:
            sim = float(DataStructs.TanimotoSimilarity(mol_fp, motif_fp))
        except Exception:
            sim = 0.0
        out[f"motif_sim_{label}"] = sim

        match = 0
        count = 0
        if need_match_or_count:
            try:
                count = len(mol.GetSubstructMatches(motif, uniquify=True))
                match = int(count > 0)
            except Exception:
                count = 0
                match = 0
            out[f"motif_match_{label}"] = match
            out[f"motif_count_{label}"] = count

        if label and label[0] in groups:
            groups[label[0]].append((match, sim))

    for g, vals in groups.items():
        if f"motif_any_{g}" in needed_set:
            out[f"motif_any_{g}"] = int(any(v[0] for v in vals)) if vals else 0
        if f"motif_max_sim_{g}" in needed_set:
            out[f"motif_max_sim_{g}"] = max((v[1] for v in vals), default=0.0)
    return out


# -----------------------------
# Molecular descriptors
# -----------------------------

def calc_lowcost_features_for_mol(
    mol: Chem.Mol,
    needed_features: Iterable[str],
    motif_mols: Dict[str, Chem.Mol],
    default_solution: str = "Unknown",
) -> Dict[str, float]:
    needed = set(needed_features)
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol.GetNumHeavyAtoms()
    n_bonds = mol.GetNumBonds()
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    sp2_atoms = sum(1 for a in mol.GetAtoms() if str(a.GetHybridization()) == "SP2")

    num_conj = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())
    num_arom_bonds = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    num_single = count_bonds_by_type(mol, Chem.BondType.SINGLE)
    num_double = count_bonds_by_type(mol, Chem.BondType.DOUBLE)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    aa_single, an_single = aromatic_single_bond_counts(mol)

    desc: Dict[str, float] = {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "HeavyAtomCount": float(n_heavy),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "LogP": Crippen.MolLogP(mol),
        "MolMR": Crippen.MolMR(mol),
        "BertzCT": BertzCT(mol),
        "BalabanJ": BalabanJ(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumRotatableBonds": rot_bonds,
        "RotatableBonds": rot_bonds,
        "RingCount": ring_count,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "Chi0": Descriptors.Chi0(mol),
        "Chi1": Descriptors.Chi1(mol),
        "Chi0n": Descriptors.Chi0n(mol),
        "Chi1n": Descriptors.Chi1n(mol),
        "Chi2n": Descriptors.Chi2n(mol),
        "Chi3n": Descriptors.Chi3n(mol),
        "Chi4n": Descriptors.Chi4n(mol),
        "NumConjugatedBonds": num_conj,
        "NumAromaticBonds": num_arom_bonds,
        "NumSingleBonds": num_single,
        "NumDoubleBonds": num_double,
        "ConjugatedBondFrac": safe_div(num_conj, n_bonds),
        "AromaticBondFrac": safe_div(num_arom_bonds, n_bonds),
        "RotatableBondFrac": safe_div(rot_bonds, n_bonds),
        "LargestRingSize": largest_ring_size(mol),
        "FusedRingPairCount": fused_ring_pair_count(mol),
        "AromaticRingSystemCount": aromatic_ring_system_count(mol),
        "AromaticAromaticSingleBondCount": aa_single,
        "AromaticNonAromaticSingleBondCount": an_single,
        "BridgeheadAtomCount": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "SpiroAtomCount": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "NumAromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        "AromaticAtomFrac": safe_div(aromatic_atoms, n_heavy),
        "SP2AtomFrac": safe_div(sp2_atoms, n_heavy),
        "HeteroAtomFrac": safe_div(hetero, n_heavy),
    }

    for sym in ["B", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]:
        c = count_atoms(mol, sym)
        desc[f"Count_{sym}"] = c
        desc[f"{sym}_frac"] = safe_div(c, n_heavy)

    # Simple chemistry proxy flags
    desc["has_B"] = int(count_atoms(mol, "B") > 0)
    desc["has_BN_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "N") > 0)
    desc["has_BO_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "O") > 0)
    desc["has_BS_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "S") > 0)
    desc["has_BNO_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "N") > 0 and count_atoms(mol, "O") > 0)
    desc["has_BOS_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "O") > 0 and count_atoms(mol, "S") > 0)
    desc["has_BNS_proxy"] = int(count_atoms(mol, "B") > 0 and count_atoms(mol, "N") > 0 and count_atoms(mol, "S") > 0)

    # Functional group proxies. These are deliberately broad low-cost SMARTS.
    fg_smarts = {
        "fg_methoxy_like": "[OX2H0][CH3]",
        "fg_alkoxy_aromatic": "[a][OX2][#6]",
        "fg_tertbutyl_like": "[CX4]([CH3])([CH3])[CH3]",
        "fg_CF3": "[CX4](F)(F)F",
        "fg_cyano": "C#N",
        "fg_carbonyl": "[CX3]=[OX1]",
        "fg_sulfone_sulfoxide": "S(=O)",
        "fg_diphenylamine_like": "[NX3]([a])[a]",
        "fg_triphenylamine_like": "[NX3]([a])([a])[a]",
        "fg_carbazole_like": "n1c2ccccc2c2ccccc12",
        "fg_phenyl": "c1ccccc1",
        "fg_fluoro_aromatic": "[a]F",
    }
    for k, smarts in fg_smarts.items():
        desc[k] = has_substruct(mol, smarts)

    # Motif features
    desc.update(motif_features(mol, motif_mols, needed=needed))

    # Solution one-hot defaults. Generated molecules usually have no solvent/phase info.
    for f in needed:
        if f.startswith("Solution_"):
            desc[f] = int(f == f"Solution_{default_solution}")

    # Fingerprint bit features only when needed
    bit_groups = {"MORGAN2": [], "MORGAN3": [], "RDKFP": [], "PATTERN": []}
    for f in needed:
        parsed = parse_bit_feature(f)
        if parsed is not None:
            bit_groups[parsed[0]].append((f, parsed[1]))

    fps = {}
    if bit_groups["MORGAN2"]:
        fps["MORGAN2"] = morgan_fp(mol, radius=2, n_bits=1024)
    if bit_groups["MORGAN3"]:
        fps["MORGAN3"] = morgan_fp(mol, radius=3, n_bits=1024)
    if bit_groups["RDKFP"]:
        fps["RDKFP"] = Chem.RDKFingerprint(mol, fpSize=2048)
    if bit_groups["PATTERN"]:
        fps["PATTERN"] = Chem.PatternFingerprint(mol, fpSize=2048)

    for group, items in bit_groups.items():
        fp = fps.get(group)
        for fname, idx in items:
            desc[fname] = bit_value(fp, idx) if fp is not None else 0

    # Fill any unknown feature with zero. This avoids prediction crashes while preserving feature order.
    for f in needed:
        if f not in desc:
            desc[f] = 0.0
        val = desc[f]
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            desc[f] = 0.0
    return desc


def calc_sa_score(mol: Chem.Mol) -> float:
    if sascorer is None:
        raise RuntimeError(
            "RDKit Contrib sascorer is unavailable. Please install/use RDKit with Contrib/SA_Score."
        )
    return float(sascorer.calculateScore(mol))


# -----------------------------
# Prediction helpers
# -----------------------------

def predict_regression(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def positive_class_index(model) -> int:
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        final_model = model.named_steps["model"]
        if hasattr(final_model, "classes_"):
            classes = list(final_model.classes_)
    if classes is None:
        return 1
    if 1 in classes:
        return classes.index(1)
    # fallback: use the larger/last class as positive
    return len(classes) - 1


def predict_positive_probability(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        pos_idx = positive_class_index(model)
        pos = np.asarray(proba[:, pos_idx], dtype=float)
        label = np.asarray(model.predict(X), dtype=int)
        return pos, label
    # fallback if only decision/predict is available
    label = np.asarray(model.predict(X), dtype=int)
    return label.astype(float), label


# -----------------------------
# Main pipeline
# -----------------------------

def run_screening(args: argparse.Namespace) -> Dict:
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    smiles_col = find_smiles_column(df, args.smiles_col)

    plpeak_bundle = load_model_dict(args.plpeak_model, "PL-peak")
    plqy_bundle = load_model_dict(args.plqy_model, "PLQY")
    fwhm_bundle = load_model_dict(args.fwhm_model, "FWHM")

    plpeak_features = list(plpeak_bundle["selected_features"])
    plqy_features = list(plqy_bundle["selected_features"])
    fwhm_features = list(fwhm_bundle["selected_features"])
    all_needed_features = sorted(set(plpeak_features + plqy_features + fwhm_features))

    # Prefer the motif library stored in the PL-peak model, then PLQY/FWHM.
    core_fragment_smiles = {}
    for bundle in [plpeak_bundle, plqy_bundle, fwhm_bundle]:
        meta = bundle.get("metadata", {}) or {}
        if isinstance(meta.get("core_fragment_smiles"), dict) and meta.get("core_fragment_smiles"):
            core_fragment_smiles = meta["core_fragment_smiles"]
            break
    motif_mols = prepare_motif_mols(core_fragment_smiles)

    rows = []
    invalid_rows = []
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        can, mol, err = canonicalize_smiles(smi)
        if mol is None:
            bad = row.to_dict()
            bad["invalid_reason"] = err
            invalid_rows.append(bad)
            continue
        try:
            feats = calc_lowcost_features_for_mol(
                mol=mol,
                needed_features=all_needed_features,
                motif_mols=motif_mols,
                default_solution=args.default_solution,
            )
            sa = calc_sa_score(mol)
        except Exception as e:
            bad = row.to_dict()
            bad["canonical_smiles_calc"] = can
            bad["invalid_reason"] = f"feature/SA calculation failed: {e}"
            invalid_rows.append(bad)
            continue

        out = row.to_dict()
        out["canonical_smiles_calc"] = can
        out["SA_score"] = sa
        for f in all_needed_features:
            out[f] = feats.get(f, 0.0)
        rows.append(out)

    pred_df = pd.DataFrame(rows)
    invalid_df = pd.DataFrame(invalid_rows)

    if pred_df.empty:
        raise RuntimeError("No valid molecules available for prediction.")

    # Drop duplicate canonical structures after successful featurization.
    before_dedup = len(pred_df)
    pred_df = pred_df.drop_duplicates(subset=["canonical_smiles_calc"]).reset_index(drop=True)
    after_dedup = len(pred_df)

    X_peak = pred_df.reindex(columns=plpeak_features, fill_value=0.0).astype(float)
    X_plqy = pred_df.reindex(columns=plqy_features, fill_value=0.0).astype(float)
    X_fwhm = pred_df.reindex(columns=fwhm_features, fill_value=0.0).astype(float)

    pred_df["pred_PL_peak_nm"] = predict_regression(plpeak_bundle["model"], X_peak)
    pred_df["pred_PLQY_high_prob"] , pred_df["pred_PLQY_high_label"] = predict_positive_probability(plqy_bundle["model"], X_plqy)
    pred_df["pred_FWHM_lt30_prob"], pred_df["pred_FWHM_lt30_label"] = predict_positive_probability(fwhm_bundle["model"], X_fwhm)

    # Filtering. Because PLQY and FWHM are classification models, the hard property conditions
    # are represented by positive label plus configurable probability cutoffs.
    mask = (
        (pred_df["pred_PL_peak_nm"] >= args.plpeak_min)
        & (pred_df["pred_PL_peak_nm"] <= args.plpeak_max)
        & (pred_df["pred_PLQY_high_label"] == 1)
        & (pred_df["pred_PLQY_high_prob"] >= args.plqy_prob_min)
        & (pred_df["pred_FWHM_lt30_label"] == 1)
        & (pred_df["pred_FWHM_lt30_prob"] >= args.fwhm_prob_min)
        & (pred_df["SA_score"] < args.sa_max)
    )
    screened = pred_df.loc[mask].copy()

    # Ranking: prioritize PLQY/FWHM confidence, then blue wavelength closeness to the middle of window, then SA.
    target_mid = (args.plpeak_min + args.plpeak_max) / 2.0
    screened["rank_score"] = (
        screened["pred_PLQY_high_prob"]
        + screened["pred_FWHM_lt30_prob"]
        - 0.002 * (screened["pred_PL_peak_nm"] - target_mid).abs()
        - 0.03 * screened["SA_score"]
    )
    screened = screened.sort_values(
        by=["rank_score", "pred_PLQY_high_prob", "pred_FWHM_lt30_prob", "SA_score"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # Put important columns first.
    preferred = [
        "candidate_id", smiles_col, "canonical_smiles_calc", "scaffold_label",
        "pred_PL_peak_nm", "pred_PLQY_high_prob", "pred_PLQY_high_label",
        "pred_FWHM_lt30_prob", "pred_FWHM_lt30_label", "SA_score", "rank_score",
        "MW", "MolWt", "HeavyAtomCount", "TPSA", "MolLogP", "RingCount",
        "NumAromaticRings", "NumRotatableBonds", "is_novel_vs_training",
        "max_tanimoto_to_training_sample"
    ]
    preferred = [c for c in preferred if c in pred_df.columns]
    other_cols = [c for c in pred_df.columns if c not in preferred and c not in all_needed_features]
    feature_cols = [c for c in all_needed_features if c in pred_df.columns]

    all_predictions_path = os.path.join(args.output_dir, "all_predictions_with_features.csv")
    all_predictions_simple_path = os.path.join(args.output_dir, "all_predictions_summary.csv")
    screened_path = os.path.join(args.output_dir, "screened_candidates.csv")
    top_path = os.path.join(args.output_dir, f"screened_top{args.top_n}.csv")
    invalid_path = os.path.join(args.output_dir, "invalid_or_failed_molecules.csv")
    report_path = os.path.join(args.output_dir, "screening_report.json")

    pred_df[preferred + other_cols + feature_cols].to_csv(all_predictions_path, index=False)
    pred_df[preferred + other_cols].to_csv(all_predictions_simple_path, index=False)
    screened[preferred + [c for c in other_cols if c in screened.columns]].to_csv(screened_path, index=False)
    screened.head(args.top_n)[preferred + [c for c in other_cols if c in screened.columns]].to_csv(top_path, index=False)
    invalid_df.to_csv(invalid_path, index=False)

    report = {
        "input_file": args.input,
        "smiles_column": smiles_col,
        "n_input_rows": int(len(df)),
        "n_valid_featurized_rows_before_dedup": int(before_dedup),
        "n_valid_unique_after_dedup": int(after_dedup),
        "n_invalid_or_failed": int(len(invalid_df)),
        "n_duplicate_removed_after_featurization": int(before_dedup - after_dedup),
        "n_screened_pass": int(len(screened)),
        "criteria": {
            "plpeak_min_nm": args.plpeak_min,
            "plpeak_max_nm": args.plpeak_max,
            "plqy_high_label_required": 1,
            "plqy_prob_min": args.plqy_prob_min,
            "fwhm_lt30_label_required": 1,
            "fwhm_prob_min": args.fwhm_prob_min,
            "sa_score_max_strict_less_than": args.sa_max,
        },
        "model_keys": {
            "plpeak": list(plpeak_bundle.keys()),
            "plqy": list(plqy_bundle.keys()),
            "fwhm": list(fwhm_bundle.keys()),
        },
        "n_features": {
            "plpeak_selected_features": len(plpeak_features),
            "plqy_selected_features": len(plqy_features),
            "fwhm_selected_features": len(fwhm_features),
            "union_features_calculated": len(all_needed_features),
        },
        "outputs": {
            "all_predictions_with_features": all_predictions_path,
            "all_predictions_summary": all_predictions_simple_path,
            "screened_candidates": screened_path,
            "screened_top": top_path,
            "invalid_or_failed_molecules": invalid_path,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Screening finished ===")
    print(f"Input rows: {len(df)}")
    print(f"Valid unique molecules: {after_dedup}")
    print(f"Invalid/failed molecules: {len(invalid_df)}")
    print(f"Passed candidates: {len(screened)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top file: {top_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Screen generated TADF candidates using low-cost dict models.")
    p.add_argument("--input", required=True, help="Input CSV containing generated molecules.")
    p.add_argument("--smiles_col", default=None, help="SMILES column name. Auto-detected if omitted.")
    p.add_argument("--plpeak_model", required=True, help="PL-peak low-cost model pkl, dict format.")
    p.add_argument("--plqy_model", required=True, help="PLQY low-cost model pkl, dict format.")
    p.add_argument("--fwhm_model", required=True, help="FWHM low-cost model pkl, dict format.")
    p.add_argument("--output_dir", default="screened_lowcost_candidates", help="Output directory.")

    p.add_argument("--plpeak_min", type=float, default=430.0, help="Minimum predicted PL-peak wavelength, nm.")
    p.add_argument("--plpeak_max", type=float, default=490.0, help="Maximum predicted PL-peak wavelength, nm.")
    p.add_argument("--plqy_prob_min", type=float, default=0.50, help="Minimum probability for high PLQY class. Default 0.50.")
    p.add_argument("--fwhm_prob_min", type=float, default=0.50, help="Minimum probability for FWHM < 30 class. Default 0.50.")
    p.add_argument("--sa_max", type=float, default=6.0, help="Strict upper bound for SA score. Default: SA < 6.")
    p.add_argument("--default_solution", default="Unknown", help="Default solvent one-hot if Solution_* features are needed.")
    p.add_argument("--top_n", type=int, default=50, help="Number of top candidates to save separately.")
    return p


def main():
    """Parse command-line arguments and run candidate screening."""
    run_screening(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
