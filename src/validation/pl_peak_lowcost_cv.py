"""
MR-TADF PL-peak low-cost QSPR model
===================================
Purpose
-------
This script builds a generation-friendly PL-peak prediction model.
It removes expensive TDDFT / experimental excited-state features such as
S1, T1, dEST, f, HOMO, LUMO and taup, and only uses features that can be
calculated directly from SMILES plus optional environment / motif labels.

Recommended use
---------------
1. Use this model for first-round screening of VAE/CVAE-generated molecules.
2. Use the previous electronic-feature-enhanced model only after TDDFT
   validation of a small number of top candidates.

Output
------
plpeak_lowcost_qspr_outputs/
    final_pl_peak_lowcost_qspr.pkl
    selected_features_lowcost.csv
    all_feature_columns_lowcost.csv
    lowcost_model_metrics.csv
    five_fold_random_cv_folds_lowcost.csv
    five_fold_random_cv_summary_lowcost.csv
    five_fold_random_cv_oof_predictions_lowcost.csv
    train_test_diagnostics_lowcost.png
    feature_importance_top20_lowcost.png
    feature_group_importance_lowcost.png

Notes
-----
This version reads core-structure info.xlsx and creates explicit motif features
for A1-A4, B1-B7 and C0-C5, including exact substructure matches and
Morgan-similarity features.
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors, Crippen

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 0. Basic settings
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(PROJECT_ROOT / "data" / "tadf_dataset.xlsx")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "plpeak_lowcost_cv")

SMILES_COL = "Smiles"
TARGET_COL = "PL-peak"
ENV_COL = "Solution"        # e.g. toluene / film / solution; if absent, set Unknown
MOLECULE_NAME_COL = "Molecules"

PL_MIN = 400.0
PL_MAX = 600.0

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_N_FEATURES = 80          # Reduced feature number to mitigate overfitting in the generation-screening model
MIN_FEATURE_IMPORTANCE = 1      # Remove zero-importance features during feature selection

# Fingerprint settings
MORGAN_BITS = 1024
RDK_BITS = 2048
PATTERN_BITS = 2048
FP_MIN_FREQ = 0.03           # remove almost-always-off bits
FP_MAX_FREQ = 0.97           # remove almost-always-on bits

# Optional: if your dataset already contains motif/core label columns, they will be used directly.
# These columns are only optional; the script can also detect A/B/C motifs from CORE_INFO_PATH.
OPTIONAL_CATEGORICAL_COLS = [
    "core_A_label", "core_B_label", "fragment_C_label",
    "primary_motif", "group_motif", "core_label",
    "Core", "core", "Motif", "motif",
]

# Core / fragment structure file.
# The uploaded file should contain two columns: "core-type" and "smiles".
# If this file is not found, the embedded DEFAULT_CORE_SMILES below will be used.
CORE_INFO_PATH = str(PROJECT_ROOT / "data" / "core-structure info.xlsx")
CORE_TYPE_COL = "core-type"
CORE_SMILES_COL = "smiles"

# Fallback copied from core-structure info.xlsx, making the script runnable even when the file path is not set yet.
DEFAULT_CORE_SMILES = {
    "A1": "C1(N(C2=CC=CC=C2)C3=C4B5C6=C(C=CC=C6)N(C7=CC=CC=C7)C4=CC=C3)=C5C=CC=C1",
    "A2": "C12=CC=CC(N3C4=C5C=CC=C4C6=C3C=CC=C6)=C1B5C7=C8N2C9=C(C=CC=C9)C8=CC=C7",
    "A3": "C1(N(C2=CC=CC=C2)C3=CC=CC=C3)=CC4=C5C(N(C6=CC=CC=C6)C(C=C(N(C7=CC=CC=C7)C8=C9B%10C%11=C(C=CC=C%11)N(C%12=CC=CC=C%12)C9=CC(N(C%13=CC=CC=C%13)C%14=CC=CC=C%14)=C8)C%10=C%15)=C%15B5C(C=CC=C%16)=C%16N4C%17=CC=CC=C%17)=C1",
    "A4": "C12=CC=CC(N(C3=CC=CC=C3)C4=C5C=CC=C4)=C1B5C6=C7N2C8=C(C=CC=C8)C7=CC=C6",
    "B1": "C1(OC2=C3B4C5=C(C6=CC=C5)N(C7=C6C=CC=C7)C3=CC=C2)=C4C=CC=C1",
    "B2": "C1(OC2=C3B4C5=C(C=CC=C5)OC3=CC=C2)=C4C=CC=C1",
    "B3": "C1(SC2=C3B4C5=C(C=CC=C5)SC3=CC=C2)=C4C=CC=C1",
    "B4": "C1(SC2=C3B4C5=C(C=CC=C5)OC3=CC=C2)=C4C=CC=C1",
    "B5": "C1(OC2=C3B4C5=C(C=CC=C5)N(C6=CC=CC=C6)C3=CC=C2)=C4C=CC=C1",
    "B6": "O=C(C1=C(C2=CC=C1)N3C4=C(C=CC=C4)C2=O)C5=C3C=CC=C5",
    "B7": "CB(C1=C2N3C4=C(C=CC=C4)B(C)C2=CC=C1)C5=C3C=CC=C5",
    "C0": "C12=CC=CC=C1B(C3=CC=CC=C3)C4=C(C=CC=C4)N2C5=CC=CC=C5",
    "C1": "C12=CC=CC=C1OC3=C(C=CC=C3)N2C4=CC=CC=C4",
    "C2": "C12=CC=CC=C1SC3=C(C=CC=C3)N2C4=CC=CC=C4",
    "C3": "C12=CC=CC=C1OC3=C(C=CC=C3)B2C4=CC=CC=C4",
    "C4": "C12=CC=CC=C1SC3=C(C=CC=C3)B2C4=CC=CC=C4",
    "C5": "O=C1C2=C(C=CC=C2)OC3=CC=CC=C31",
}

# Loaded in initialize_core_queries() before feature construction.
CORE_QUERY_INFO = {}


# =========================================================
# 1. Utility functions
# =========================================================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def count_substruct(mol, smarts: str) -> int:
    if not smarts:
        return 0
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def bitvect_to_dict(bitvect, prefix: str) -> dict:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return {f"{prefix}_{i}": int(v) for i, v in enumerate(arr)}


def get_largest_ring_size(mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return 0
    return max(len(r) for r in rings)


def get_fused_ring_pair_count(mol) -> int:
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    count = 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i].intersection(rings[j])) >= 2:
                count += 1
    return count


def get_aromatic_ring_system_count(mol) -> int:
    """Approximate number of connected aromatic ring systems."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    aromatic_rings = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(ring)
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
            if len(aromatic_rings[i].intersection(aromatic_rings[j])) >= 1:
                union(i, j)

    return len({find(i) for i in range(len(aromatic_rings))})


def count_single_bonds_between_aromatic_atoms(mol) -> int:
    count = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE:
            a1 = b.GetBeginAtom()
            a2 = b.GetEndAtom()
            if a1.GetIsAromatic() and a2.GetIsAromatic():
                count += 1
    return count


def count_aromatic_to_nonaromatic_single_bonds(mol) -> int:
    count = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE:
            a1 = b.GetBeginAtom()
            a2 = b.GetEndAtom()
            if a1.GetIsAromatic() != a2.GetIsAromatic():
                count += 1
    return count


def safe_descriptor(func, mol, default=np.nan):
    try:
        return safe_float(func(mol), default=default)
    except Exception:
        return default



# =========================================================
# 2. Core / fragment query loading
# =========================================================
def _try_existing_core_path(path: str) -> str | None:
    """Find the core info file from the configured path or common local locations."""
    candidates = []
    if path:
        candidates.append(path)
        candidates.append(os.path.basename(path))
    candidates.extend([
        "core-structure info.xlsx",
        "core_structure_info.xlsx",
        os.path.join(os.getcwd(), "core-structure info.xlsx"),
        os.path.join(os.getcwd(), "core_structure_info.xlsx"),
    ])
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def load_core_smiles(core_info_path: str = CORE_INFO_PATH) -> dict:
    """
    Load A/B/C core and fragment SMILES from an Excel file.

    Required columns:
        core-type: A1, A2, ..., B1, ..., C0, ...
        smiles: corresponding structure SMILES

    If the file cannot be found, DEFAULT_CORE_SMILES is used.
    """
    existing_path = _try_existing_core_path(core_info_path)
    if existing_path is None:
        print("Core structure file not found; using embedded DEFAULT_CORE_SMILES.")
        return DEFAULT_CORE_SMILES.copy()

    core_df = pd.read_excel(existing_path)
    core_df.columns = [str(c).strip() for c in core_df.columns]
    if CORE_TYPE_COL not in core_df.columns or CORE_SMILES_COL not in core_df.columns:
        raise ValueError(
            f"Core file must contain columns '{CORE_TYPE_COL}' and '{CORE_SMILES_COL}'. "
            f"Current columns: {list(core_df.columns)}"
        )

    core_smiles = {}
    for _, row in core_df.iterrows():
        label = str(row[CORE_TYPE_COL]).strip()
        smi = str(row[CORE_SMILES_COL]).strip()
        if not label or label.lower() == "nan" or not smi or smi.lower() == "nan":
            continue
        mol = mol_from_smiles(smi)
        if mol is None:
            print(f"Warning: invalid core SMILES skipped: {label} -> {smi}")
            continue
        core_smiles[label] = Chem.MolToSmiles(mol, canonical=True)

    if not core_smiles:
        raise ValueError("No valid core/fragment SMILES were loaded from the core info file.")

    print(f"Loaded {len(core_smiles)} core/fragment structures from: {existing_path}")
    return core_smiles


def initialize_core_queries(core_info_path: str = CORE_INFO_PATH) -> dict:
    """
    Compile core/fragment SMILES into RDKit query molecules.

    Matching strategy:
    1. exact substructure match from the core/fragment SMILES;
    2. Morgan similarity to each reference motif as a soft low-cost feature.

    This is more specific than broad B/N/O/S proxy features and can distinguish A1-A4,
    B1-B7, and C0-C5 according to the user's defined motif system.
    """
    global CORE_QUERY_INFO
    if CORE_QUERY_INFO:
        return CORE_QUERY_INFO

    core_smiles = load_core_smiles(core_info_path)
    query_info = {}
    for label, smi in core_smiles.items():
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        group = label[0].upper() if label else "Unknown"
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
        query_info[label] = {
            "label": label,
            "group": group,
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "mol": mol,
            "fp": fp,
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        }

    CORE_QUERY_INFO = query_info
    labels_by_group = defaultdict(list)
    for label, info in query_info.items():
        labels_by_group[info["group"]].append(label)
    print("Core/fragment query labels:", {g: sorted(v) for g, v in labels_by_group.items()})
    return CORE_QUERY_INFO


def get_core_fragment_features(mol) -> dict:
    """Generate exact match, count, group and similarity features for A/B/C motifs."""
    query_info = initialize_core_queries()
    features = {}

    mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
    group_match_count = defaultdict(int)
    group_total_count = defaultdict(int)
    group_max_similarity = defaultdict(float)
    total_matched_labels = 0

    for label in sorted(query_info.keys()):
        info = query_info[label]
        group = info["group"]
        qmol = info["mol"]

        try:
            matches = mol.GetSubstructMatches(qmol, uniquify=True, maxMatches=20)
            n_matches = len(matches)
        except Exception:
            n_matches = 0

        try:
            sim = float(DataStructs.TanimotoSimilarity(mol_fp, info["fp"]))
        except Exception:
            sim = 0.0

        features[f"motif_match_{label}"] = int(n_matches > 0)
        features[f"motif_count_{label}"] = int(n_matches)
        features[f"motif_sim_{label}"] = sim

        if n_matches > 0:
            total_matched_labels += 1
            group_match_count[group] += 1
            group_total_count[group] += n_matches
        group_max_similarity[group] = max(group_max_similarity[group], sim)

    for group in ["A", "B", "C"]:
        features[f"motif_any_{group}"] = int(group_match_count[group] > 0)
        features[f"motif_n_labels_{group}"] = int(group_match_count[group])
        features[f"motif_total_count_{group}"] = int(group_total_count[group])
        features[f"motif_max_sim_{group}"] = float(group_max_similarity[group])

    features["motif_n_matched_labels_total"] = int(total_matched_labels)
    features["motif_any_target_core"] = int(group_match_count["A"] > 0 or group_match_count["B"] > 0)
    features["motif_any_fragment_C"] = int(group_match_count["C"] > 0)
    return features


# =========================================================
# 3. Low-cost feature engineering from SMILES
# =========================================================
def featurize_lowcost(smiles: str) -> dict | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    features = {}
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol.GetNumHeavyAtoms()
    n_bonds = mol.GetNumBonds()

    # ---------- Basic RDKit 2D descriptors ----------
    basic_descriptors = {
        "MolWt": Descriptors.MolWt,
        "ExactMolWt": Descriptors.ExactMolWt,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "TPSA": Descriptors.TPSA,
        "MolLogP": Crippen.MolLogP,
        "MolMR": Crippen.MolMR,
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
    }
    for name, func in basic_descriptors.items():
        features[name] = safe_descriptor(func, mol)

    # ---------- Ring and rigidity proxies ----------
    features["NumConjugatedBonds"] = sum(int(b.GetIsConjugated()) for b in mol.GetBonds())
    features["NumAromaticBonds"] = sum(int(b.GetIsAromatic()) for b in mol.GetBonds())
    features["NumSingleBonds"] = sum(int(b.GetBondType() == Chem.BondType.SINGLE) for b in mol.GetBonds())
    features["NumDoubleBonds"] = sum(int(b.GetBondType() == Chem.BondType.DOUBLE) for b in mol.GetBonds())
    features["ConjugatedBondFrac"] = features["NumConjugatedBonds"] / (n_bonds + 1e-6)
    features["AromaticBondFrac"] = features["NumAromaticBonds"] / (n_bonds + 1e-6)
    features["RotatableBondFrac"] = features["NumRotatableBonds"] / (n_bonds + 1e-6)
    features["LargestRingSize"] = get_largest_ring_size(mol)
    features["FusedRingPairCount"] = get_fused_ring_pair_count(mol)
    features["AromaticRingSystemCount"] = get_aromatic_ring_system_count(mol)
    features["AromaticAromaticSingleBondCount"] = count_single_bonds_between_aromatic_atoms(mol)
    features["AromaticNonAromaticSingleBondCount"] = count_aromatic_to_nonaromatic_single_bonds(mol)
    features["BridgeheadAtomCount"] = safe_descriptor(rdMolDescriptors.CalcNumBridgeheadAtoms, mol, default=0)
    features["SpiroAtomCount"] = safe_descriptor(rdMolDescriptors.CalcNumSpiroAtoms, mol, default=0)
    features["NumAromaticHeterocycles"] = safe_descriptor(rdMolDescriptors.CalcNumAromaticHeterocycles, mol, default=0)
    features["NumAromaticCarbocycles"] = safe_descriptor(rdMolDescriptors.CalcNumAromaticCarbocycles, mol, default=0)

    # ---------- Atom counts and fractions ----------
    elements = ["B", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]
    for e in elements:
        features[f"Count_{e}"] = atom_symbols.count(e)

    hetero_count = sum(1 for s in atom_symbols if s not in ["C", "H"])
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    sp2_like_atoms = sum(
        1 for a in mol.GetAtoms()
        if a.GetHybridization() in [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP]
    )

    features["AromaticAtomFrac"] = aromatic_atoms / (n_atoms + 1e-6)
    features["SP2AtomFrac"] = sp2_like_atoms / (n_atoms + 1e-6)
    features["HeteroAtomFrac"] = hetero_count / (n_atoms + 1e-6)
    features["B_frac"] = features["Count_B"] / (n_atoms + 1e-6)
    features["N_frac"] = features["Count_N"] / (n_atoms + 1e-6)
    features["O_frac"] = features["Count_O"] / (n_atoms + 1e-6)
    features["S_frac"] = features["Count_S"] / (n_atoms + 1e-6)
    features["F_frac"] = features["Count_F"] / (n_atoms + 1e-6)

    # ---------- MR-TADF broad core proxies ----------
    # These are not exact core labels. They are low-cost structural proxies useful for generated molecules.
    features["has_B"] = int(features["Count_B"] >= 1)
    features["has_BN_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1)
    features["has_BO_proxy"] = int(features["Count_B"] >= 1 and features["Count_O"] >= 1)
    features["has_BS_proxy"] = int(features["Count_B"] >= 1 and features["Count_S"] >= 1)
    features["has_BNO_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1 and features["Count_O"] >= 1)
    features["has_BOS_proxy"] = int(features["Count_B"] >= 1 and features["Count_O"] >= 1 and features["Count_S"] >= 1)
    features["has_BNS_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1 and features["Count_S"] >= 1)

    # ---------- Functional group / substituent features ----------
    # SMARTS are approximate and designed for statistical screening, not final chemical assignment.
    fg_smarts = {
        "fg_methoxy_like": "[OX2;!R]-[CH3]",
        "fg_alkoxy_aromatic": "c-[OX2]-[CX4]",
        "fg_tertbutyl_like": "C(C)(C)C",
        "fg_CF3": "C(F)(F)F",
        "fg_cyano": "C#N",
        "fg_carbonyl": "C=O",
        "fg_sulfone_sulfoxide": "S(=O)",
        "fg_diphenylamine_like": "N(c1ccccc1)c1ccccc1",
        "fg_triphenylamine_like": "N(c1ccccc1)(c1ccccc1)c1ccccc1",
        "fg_carbazole_like": "n1c2ccccc2c2ccccc12",
        "fg_phenyl": "c1ccccc1",
        "fg_fluoro_aromatic": "cF",
    }
    for name, smarts in fg_smarts.items():
        features[name] = count_substruct(mol, smarts)

    # ---------- User-defined A/B/C core and fragment features ----------
    # These come from core-structure info.xlsx and are used to distinguish
    # A1-A4, B1-B7 and C0-C5 instead of only broad B/N/O/S proxies.
    features.update(get_core_fragment_features(mol))

    # ---------- Fingerprints ----------
    fp_morgan_r2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
    fp_morgan_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=MORGAN_BITS)
    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    fp_rdk = Chem.RDKFingerprint(mol, fpSize=RDK_BITS)
    fp_pattern = Chem.PatternFingerprint(mol, fpSize=PATTERN_BITS)

    features.update(bitvect_to_dict(fp_morgan_r2, "MORGAN2"))
    features.update(bitvect_to_dict(fp_morgan_r3, "MORGAN3"))
    features.update(bitvect_to_dict(fp_maccs, "MACCS"))
    features.update(bitvect_to_dict(fp_rdk, "RDKFP"))
    features.update(bitvect_to_dict(fp_pattern, "PATTERN"))

    return features


def remove_sparse_fp_bits(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    fp_prefixes = ("MORGAN2_", "MORGAN3_", "MACCS_", "RDKFP_", "PATTERN_")
    fp_cols = [c for c in X.columns if c.startswith(fp_prefixes)]
    non_fp_cols = [c for c in X.columns if c not in fp_cols]

    if not fp_cols:
        return X.copy(), [], non_fp_cols

    fp_freq = X[fp_cols].mean(axis=0)
    keep_fp_cols = fp_freq[(fp_freq >= FP_MIN_FREQ) & (fp_freq <= FP_MAX_FREQ)].index.tolist()

    X_filtered = pd.concat([X[non_fp_cols].copy(), X[keep_fp_cols].copy()], axis=1)
    return X_filtered, keep_fp_cols, non_fp_cols


def add_optional_categorical_features(df_valid: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OPTIONAL_CATEGORICAL_COLS if c in df_valid.columns]
    if not cols:
        return pd.DataFrame(index=df_valid.index)

    cat_df = df_valid[cols].copy()
    for c in cols:
        cat_df[c] = cat_df[c].fillna("Unknown").astype(str).str.strip()

    return pd.get_dummies(cat_df, columns=cols, drop_first=False)


def build_feature_matrix(df_valid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feature_rows = []
    valid_rows = []

    for idx, smi in enumerate(df_valid[SMILES_COL]):
        feat = featurize_lowcost(smi)
        if feat is not None:
            feature_rows.append(feat)
            valid_rows.append(idx)

    df_valid2 = df_valid.iloc[valid_rows].reset_index(drop=True)
    X_struct = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0)

    X_struct, keep_fp_cols, non_fp_cols = remove_sparse_fp_bits(X_struct)

    # Environment feature. For generation screening, set the same environment value as used here.
    if ENV_COL in df_valid2.columns:
        env_raw = df_valid2[[ENV_COL]].copy()
        env_raw[ENV_COL] = env_raw[ENV_COL].fillna("Unknown").astype(str).str.strip()
    else:
        env_raw = pd.DataFrame({ENV_COL: ["Unknown"] * len(df_valid2)})
    X_env = pd.get_dummies(env_raw, columns=[ENV_COL], prefix=[ENV_COL], drop_first=False)

    # Optional motif/core labels already present in the source table.
    X_cat = add_optional_categorical_features(df_valid2)

    X_all = pd.concat(
        [
            X_struct.reset_index(drop=True),
            X_env.reset_index(drop=True),
            X_cat.reset_index(drop=True),
        ],
        axis=1,
    )

    metadata = {
        "kept_fingerprint_columns": keep_fp_cols,
        "non_fingerprint_columns": non_fp_cols,
        "environment_columns": X_env.columns.tolist(),
        "optional_categorical_columns_encoded": X_cat.columns.tolist(),
        "all_feature_columns": X_all.columns.tolist(),
        "core_fragment_labels": sorted(CORE_QUERY_INFO.keys()),
        "core_fragment_smiles": {k: v["smiles"] for k, v in CORE_QUERY_INFO.items()},
    }
    return df_valid2, X_all, metadata


# =========================================================
# 3. Feature selection and model fitting
# =========================================================
def make_lgbm_for_feature_selection(seed=RANDOM_STATE):
    # Conservative feature-selection model. The aim is not to maximize training fit,
    # but to rank robust low-cost descriptors for generated-molecule screening.
    return LGBMRegressor(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=12,
        max_depth=4,
        min_child_samples=25,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.2,
        reg_lambda=3.0,
        random_state=seed,
        verbosity=-1,
    )


def make_lgbm_final(seed=RANDOM_STATE):
    # More regularized final model to reduce the train-test gap observed with 80 features.
    return LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=16,
        max_depth=4,
        min_child_samples=15,
        subsample=0.60,
        subsample_freq=1,
        colsample_bytree=1.00,
        reg_alpha=8.0,
        reg_lambda=4.0,
        min_split_gain=0.0,
        random_state=seed,
        verbosity=-1
    )


def select_top_features(X_train: pd.DataFrame, y_train: np.ndarray, top_n: int = TOP_N_FEATURES) -> tuple[list[str], pd.DataFrame]:
    model_fs = make_lgbm_for_feature_selection()
    model_fs.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_fs.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Keep only informative features. This avoids passing zero-importance
    # descriptors/fingerprint bits into the final generation-screening model.
    informative_df = importance_df[importance_df["importance"] >= MIN_FEATURE_IMPORTANCE].copy()
    if len(informative_df) == 0:
        informative_df = importance_df.copy()

    selected = informative_df.head(top_n)["feature"].tolist()
    return selected, importance_df


def run_five_fold_random_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    smiles: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random five-fold CV with feature selection repeated inside every fold."""
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    smiles = pd.Series(smiles).reset_index(drop=True)

    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    fold_rows = []
    selected_feature_rows = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_fit = X.iloc[fit_idx]
        X_val = X.iloc[val_idx]
        y_fit = y[fit_idx]
        y_val = y[val_idx]

        fold_features, _ = select_top_features(
            X_fit,
            y_fit,
            top_n=TOP_N_FEATURES,
        )
        for rank, feature in enumerate(fold_features, start=1):
            selected_feature_rows.append({
                "fold": fold,
                "rank": rank,
                "feature": feature,
            })

        fold_model = make_lgbm_final(seed=RANDOM_STATE + fold)
        fold_model.fit(X_fit[fold_features], y_fit)
        pred_fit = fold_model.predict(X_fit[fold_features])
        pred_val = fold_model.predict(X_val[fold_features])
        oof_pred[val_idx] = pred_val
        oof_fold[val_idx] = fold

        fold_rows.append({
            "fold": fold,
            "n_train": len(fit_idx),
            "n_validation": len(val_idx),
            "n_selected_features": len(fold_features),
            "train_MAE_nm": mean_absolute_error(y_fit, pred_fit),
            "train_RMSE_nm": rmse(y_fit, pred_fit),
            "train_R2": r2_score(y_fit, pred_fit),
            "validation_MAE_nm": mean_absolute_error(y_val, pred_val),
            "validation_RMSE_nm": rmse(y_val, pred_val),
            "validation_R2": r2_score(y_val, pred_val),
        })

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame([
        {
            "metric": "MAE_nm",
            "mean": fold_df["validation_MAE_nm"].mean(),
            "std": fold_df["validation_MAE_nm"].std(ddof=1),
            "overall_OOF": mean_absolute_error(y, oof_pred),
        },
        {
            "metric": "RMSE_nm",
            "mean": fold_df["validation_RMSE_nm"].mean(),
            "std": fold_df["validation_RMSE_nm"].std(ddof=1),
            "overall_OOF": rmse(y, oof_pred),
        },
        {
            "metric": "R2",
            "mean": fold_df["validation_R2"].mean(),
            "std": fold_df["validation_R2"].std(ddof=1),
            "overall_OOF": r2_score(y, oof_pred),
        },
    ])
    oof_df = pd.DataFrame({
        "training_row": np.arange(len(y)),
        "fold": oof_fold,
        "Smiles": smiles,
        "actual_PL_peak": y,
        "oof_predicted_PL_peak": oof_pred,
        "residual": oof_pred - y,
    })
    selected_feature_df = pd.DataFrame(selected_feature_rows)
    return fold_df, summary_df, oof_df, selected_feature_df


def evaluate_predictions(y_train, pred_train, y_test, pred_test) -> pd.DataFrame:
    rows = [
        {
            "split": "train",
            "MAE_nm": mean_absolute_error(y_train, pred_train),
            "RMSE_nm": rmse(y_train, pred_train),
            "R2": r2_score(y_train, pred_train),
            "n": len(y_train),
        },
        {
            "split": "test",
            "MAE_nm": mean_absolute_error(y_test, pred_test),
            "RMSE_nm": rmse(y_test, pred_test),
            "R2": r2_score(y_test, pred_test),
            "n": len(y_test),
        },
    ]
    return pd.DataFrame(rows)



# =========================================================
# 4. Unified publication plotting style
# =========================================================
DPI = 300
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "ytick.direction": "out",
    "xtick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_TRAIN = "#D55E5E"   # muted red
COLOR_TEST = "#2C7FB8"    # paper-style blue
COLOR_BLUE = "#2C7FB8"
COLOR_BLUE_DARK = "#1F5E8C"
COLOR_GREY = "#5A5A5A"

# Larger fonts used specifically for SHAP and feature-importance figures.
INTERPRET_TICK_SIZE = 14
INTERPRET_LABEL_SIZE = 16
INTERPRET_TITLE_SIZE = 17
INTERPRET_COLORBAR_LABEL_SIZE = 15


def prettify_axis(ax, grid_axis=None, labelsize=11):
    """Use a closed box frame for publication-style consistency."""
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.tick_params(width=1.0, length=4, labelsize=labelsize)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.30)
    else:
        ax.grid(False)


def enlarge_shap_fonts(fig, ax, xlabel):
    """Increase beeswarm feature names, axis text, and colorbar text."""
    ax.set_xlabel(xlabel, fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.tick_params(axis="both", labelsize=INTERPRET_TICK_SIZE)
    for extra_ax in fig.axes:
        if extra_ax is ax:
            continue
        extra_ax.tick_params(labelsize=INTERPRET_TICK_SIZE)
        extra_ax.xaxis.label.set_size(INTERPRET_COLORBAR_LABEL_SIZE)
        extra_ax.yaxis.label.set_size(INTERPRET_COLORBAR_LABEL_SIZE)


def format_feature_label(feature: str) -> str:
    """Format feature names with consistent case, Greek symbols and subscripts."""
    mapping = {
        "S1": r"$S_1$",
        "T1": r"$T_1$",
        "dEST": r"$\Delta E_{\mathrm{ST}}$",
        "∆EST": r"$\Delta E_{\mathrm{ST}}$",
        "ΔEST": r"$\Delta E_{\mathrm{ST}}$",
        "taup": r"$\tau_{\mathrm{p}}$",
        "τp": r"$\tau_{\mathrm{p}}$",
        "log_taup": r"log$_{10}(\tau_{\mathrm{p}})$",
        "f": r"$f$",
        "HOMO": "HOMO",
        "LUMO": "LUMO",
        "HL_gap": r"$E_{\mathrm{LUMO}}-E_{\mathrm{HOMO}}$",
        "S1_T1_gap": r"$S_1-T_1$",
        "orbital_center": "Orbital center",
        "f_div_dE": r"$f/\Delta E_{\mathrm{ST}}$",
        "f_times_dE": r"$f\times\Delta E_{\mathrm{ST}}$",
        "PLQY": "PLQY",
        "MolWt": "Molecular weight",
        "ExactMolWt": "Exact molecular weight",
        "TPSA": "TPSA",
        "MolLogP": "MolLogP",
        "MolMR": "MolMR",
        "BertzCT": "BertzCT",
        "NumAromaticRings": "Number of aromatic rings",
        "NumAliphaticRings": "Number of aliphatic rings",
        "NumSaturatedRings": "Number of saturated rings",
        "NumHAcceptors": "Number of H acceptors",
        "NumHDonors": "Number of H donors",
        "NumRotatableBonds": "Number of rotatable bonds",
        "RingCount": "Ring count",
        "FractionCSP3": r"Fraction Csp$^3$",
        "NumConjugatedBonds": "Number of conjugated bonds",
        "NumAromaticBonds": "Number of aromatic bonds",
        "ConjugatedBondFrac": "Conjugated bond fraction",
        "AromaticBondFrac": "Aromatic bond fraction",
        "RotatableBondFrac": "Rotatable bond fraction",
        "AromaticAtomFrac": "Aromatic atom fraction",
        "SP2AtomFrac": r"sp$^2$ atom fraction",
        "HeteroAtomFrac": "Heteroatom fraction",
        "FusedRingPairCount": "Fused ring pair count",
        "AromaticRingSystemCount": "Aromatic ring system count",
        "BridgeheadAtomCount": "Bridgehead atom count",
        "SpiroAtomCount": "Spiro atom count",
        "NumAromaticHeterocycles": "Number of aromatic heterocycles",
        "NumAromaticCarbocycles": "Number of aromatic carbocycles",
    }
    if feature in mapping:
        return mapping[feature]
    if feature.startswith("Count_"):
        return "Count " + feature.replace("Count_", "")
    if feature.endswith("_frac") and len(feature) > 5:
        return feature.replace("_frac", " fraction")
    if feature.startswith("fg_"):
        return feature.replace("fg_", "FG: ").replace("_", " ")
    if feature.startswith("motif_match_"):
        return "Motif match " + feature.replace("motif_match_", "")
    if feature.startswith("motif_count_"):
        return "Motif count " + feature.replace("motif_count_", "")
    if feature.startswith("motif_sim_"):
        return "Motif similarity " + feature.replace("motif_sim_", "")
    if feature.startswith("motif_any_"):
        return "Any motif " + feature.replace("motif_any_", "")
    if feature.startswith("motif_max_sim_"):
        return "Max motif similarity " + feature.replace("motif_max_sim_", "")
    if feature.startswith("has_"):
        return feature.replace("has_", "Has ").replace("_", " ")
    return feature.replace("_", " ")


def save_figure(fig, save_path: str):
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    pdf_path = str(save_path).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_diagnostics(
    y_train, pred_train, y_test, pred_test,
    metrics_df: pd.DataFrame,
    save_path: str,
):
    """Simple train/test fitting plot: no marginal distribution or residual panels."""
    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([pred_train, pred_test])
    min_v = float(min(all_true.min(), all_pred.min()) - 8)
    max_v = float(max(all_true.max(), all_pred.max()) + 8)

    train_row = metrics_df[metrics_df["split"] == "train"].iloc[0]
    test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(
        y_train, pred_train,
        s=42, alpha=0.78, color=COLOR_TRAIN,
        edgecolor="white", linewidth=0.35,
        label=f"Training set ($R^2$={train_row['R2']:.2f})",
    )
    ax.scatter(
        y_test, pred_test,
        s=48, alpha=0.86, color=COLOR_TEST,
        edgecolor="white", linewidth=0.35,
        label=f"Test set ($R^2$={test_row['R2']:.2f})",
    )
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5, color=COLOR_GREY, label="Ideal fit")

    coef = np.polyfit(all_true, all_pred, 1)
    fit_x = np.linspace(min_v, max_v, 200)
    ax.plot(fit_x, coef[0] * fit_x + coef[1], linewidth=1.8, color="black", label="Linear fit")

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental PL-peak / nm", fontsize=15)
    ax.set_ylabel("Predicted PL-peak / nm", fontsize=15)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    prettify_axis(ax)
    save_figure(fig, save_path)


def plot_feature_importance(importance_sel: pd.DataFrame, save_path: str, top_n=20):
    top_imp = importance_sel.sort_values("importance", ascending=True).tail(top_n).copy()
    top_imp["feature_label"] = top_imp["feature"].apply(format_feature_label)

    fig, ax = plt.subplots(figsize=(8.8, max(6.2, 0.36 * len(top_imp) + 1.6)))
    ax.barh(top_imp["feature_label"], top_imp["importance"], color=COLOR_BLUE, edgecolor=COLOR_BLUE_DARK, linewidth=0.4)
    ax.set_xlabel("Feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title(f"Top {len(top_imp)} feature importances", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
    save_figure(fig, save_path)


def feature_group_name(feature: str) -> str:
    if feature.startswith("MORGAN"):
        return "Morgan fingerprint"
    if feature.startswith("MACCS"):
        return "MACCS keys"
    if feature.startswith("RDKFP"):
        return "RDKit fingerprint"
    if feature.startswith("PATTERN"):
        return "Pattern fingerprint"
    if feature.startswith(f"{ENV_COL}_"):
        return "Environment"
    if feature.startswith("fg_"):
        return "Functional group"
    if feature.startswith("motif_match_") or feature.startswith("motif_count_"):
        return "Exact A/B/C motif match"
    if feature.startswith("motif_sim_") or feature.startswith("motif_max_sim_"):
        return "A/B/C motif similarity"
    if feature.startswith("motif_any_") or feature.startswith("motif_n_") or feature.startswith("motif_total_count_"):
        return "A/B/C motif summary"
    if feature.startswith("has_B"):
        return "MR core proxy"
    if feature in ["S1", "T1", "dEST", "f", "HOMO", "LUMO", "taup", "log_taup", "HL_gap", "S1_T1_gap", "orbital_center", "f_div_dE", "f_times_dE"]:
        return "Electronic descriptor"
    if feature.startswith("core_") or feature.startswith("Core_") or feature.startswith("Motif_") or feature.startswith("motif_"):
        return "Core/motif label"
    return "2D descriptor"


def plot_feature_group_importance(importance_sel: pd.DataFrame, save_path: str):
    df = importance_sel.copy()
    df["group"] = df["feature"].apply(feature_group_name)
    group_imp = df.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, max(5.6, 0.44 * len(group_imp) + 1.6)))
    ax.barh(group_imp["group"], group_imp["importance"], color=COLOR_BLUE, edgecolor=COLOR_BLUE_DARK, linewidth=0.4)
    ax.set_xlabel("Summed feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature group", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title("Feature group importance", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
    save_figure(fig, save_path)


def plot_shap_summary(model, X_for_shap: pd.DataFrame, save_path: str, max_display=20):
    """Create a SHAP beeswarm summary with the same closed-frame style."""
    try:
        import shap
    except Exception as exc:
        print(f"SHAP is not installed or cannot be imported. Skipped SHAP plot: {exc}")
        return

    X_plot = X_for_shap.copy()
    if len(X_plot) > 300:
        X_plot = X_plot.sample(300, random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_plot)
        display_names = [format_feature_label(c) for c in X_plot.columns]
        X_display = X_plot.copy()
        X_display.columns = display_names

        plt.figure(figsize=(8.8, 7.2), dpi=DPI)
        shap.summary_plot(shap_values, X_display, max_display=max_display, show=False)
        ax = plt.gca()
        fig = plt.gcf()
        enlarge_shap_fonts(fig, ax, "SHAP value / nm")
        prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)
        fig.tight_layout()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        fig.savefig(str(save_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"SHAP plot failed and was skipped: {exc}")


# =========================================================
# 5. Main workflow
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    print("===== Load data =====")
    df = pd.read_excel(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if SMILES_COL not in df.columns:
        raise ValueError(f"SMILES column '{SMILES_COL}' not found. Current columns: {list(df.columns)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Current columns: {list(df.columns)}")

    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()

    print("Original data size:", len(df))
    df = df[(df[TARGET_COL] >= PL_MIN) & (df[TARGET_COL] <= PL_MAX)].copy()
    print(f"{PL_MIN:.0f}-{PL_MAX:.0f} nm data size:", len(df))

    print("\n===== Load A/B/C core and fragment structures =====")
    initialize_core_queries(CORE_INFO_PATH)

    print("\n===== Build low-cost features from SMILES =====")
    df_valid, X_all, metadata = build_feature_matrix(df)
    y = df_valid[TARGET_COL].values.astype(float)

    print("Valid molecules:", len(df_valid))
    print("Low-cost feature matrix shape:", X_all.shape)
    print("Kept fingerprint bits:", len(metadata["kept_fingerprint_columns"]))
    print("Environment features:", metadata["environment_columns"])

    # Save feature metadata
    pd.DataFrame({"feature": X_all.columns}).to_csv(
        os.path.join(OUTPUT_DIR, "all_feature_columns_lowcost.csv"), index=False
    )

    # Use all valid molecules; no OOF-based sample removal is applied.
    df_clean = df_valid.reset_index(drop=True)
    X_clean = X_all.reset_index(drop=True)
    y_clean = y

    print("Original valid size:", len(df_valid))
    print("Cleaned size:", len(df_clean))

    print("\n===== Independent train/test split =====")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X_clean,
        y_clean,
        df_clean[SMILES_COL].astype(str),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    print("\n===== Random five-fold cross-validation on training set =====")
    cv_folds_df, cv_summary_df, cv_oof_df, cv_features_df = run_five_fold_random_cv(
        X_train,
        y_train,
        smiles_train,
    )
    print(cv_folds_df.to_string(index=False))
    print("\nCross-validation summary (mean +/- SD):")
    print(cv_summary_df.to_string(index=False))

    cv_folds_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_folds_lowcost.csv"), index=False
    )
    cv_summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_summary_lowcost.csv"), index=False
    )
    cv_oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_oof_predictions_lowcost.csv"), index=False
    )
    cv_features_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_random_cv_selected_features_lowcost.csv"), index=False
    )

    print("\n===== Feature selection on the complete training set =====")
    selected_features, importance_all = select_top_features(X_train, y_train, top_n=TOP_N_FEATURES)
    print("Top selected features:")
    print(selected_features)

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(OUTPUT_DIR, "selected_features_lowcost.csv"), index=False
    )
    importance_all.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_all_lowcost.csv"), index=False)

    print("\n===== Final model =====")
    model = make_lgbm_final()
    model.fit(X_train[selected_features], y_train)

    pred_train = model.predict(X_train[selected_features])
    pred_test = model.predict(X_test[selected_features])

    metrics_df = evaluate_predictions(y_train, pred_train, y_test, pred_test)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "lowcost_model_metrics.csv"), index=False)

    importance_sel = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_sel.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_selected_lowcost.csv"), index=False)

    # Save figures using previous diagnostic standard
    diag_path = os.path.join(OUTPUT_DIR, "train_test_diagnostics_lowcost.png")
    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_top20_lowcost.png")
    group_imp_path = os.path.join(OUTPUT_DIR, "feature_group_importance_lowcost.png")

    plot_prediction_diagnostics(y_train, pred_train, y_test, pred_test, metrics_df, diag_path)
    plot_feature_importance(importance_sel, imp_path, top_n=20)
    plot_feature_group_importance(importance_sel, group_imp_path)
    plot_shap_summary(model, X_train[selected_features], os.path.join(OUTPUT_DIR, "shap_summary_lowcost.png"), max_display=20)

    # Save prediction table for diagnosis
    train_pred_df = pd.DataFrame({
        "split": "train",
        "actual_PL_peak": y_train,
        "predicted_PL_peak": pred_train,
        "residual": pred_train - y_train,
    })
    test_pred_df = pd.DataFrame({
        "split": "test",
        "actual_PL_peak": y_test,
        "predicted_PL_peak": pred_test,
        "residual": pred_test - y_test,
    })
    pd.concat([train_pred_df, test_pred_df], axis=0).to_csv(
        os.path.join(OUTPUT_DIR, "train_test_predictions_lowcost.csv"), index=False
    )

    # Save deployable package for generated-molecule screening
    package = {
        "model": model,
        "selected_features": selected_features,
        "all_feature_columns": X_all.columns.tolist(),
        "metadata": metadata,
        "settings": {
            "SMILES_COL": SMILES_COL,
            "TARGET_COL": TARGET_COL,
            "ENV_COL": ENV_COL,
            "PL_MIN": PL_MIN,
            "PL_MAX": PL_MAX,
            "FP_MIN_FREQ": FP_MIN_FREQ,
            "FP_MAX_FREQ": FP_MAX_FREQ,
            "MORGAN_BITS": MORGAN_BITS,
            "RDK_BITS": RDK_BITS,
            "PATTERN_BITS": PATTERN_BITS,
            "TOP_N_FEATURES": TOP_N_FEATURES,
            "MIN_FEATURE_IMPORTANCE": MIN_FEATURE_IMPORTANCE,
            "CORE_INFO_PATH": CORE_INFO_PATH,
            "CORE_TYPE_COL": CORE_TYPE_COL,
            "CORE_SMILES_COL": CORE_SMILES_COL,
            "validation": "80:20 random holdout plus shuffled five-fold KFold on the training set",
            "cv_n_splits": 5,
        },
    }
    model_path = os.path.join(OUTPUT_DIR, "final_pl_peak_lowcost_qspr.pkl")
    joblib.dump(package, model_path)

    # Also save JSON metadata that is easy to inspect manually.
    json_meta = {
        "selected_features": selected_features,
        "all_feature_columns": X_all.columns.tolist(),
        "metadata": metadata,
        "settings": package["settings"],
    }
    with open(os.path.join(OUTPUT_DIR, "lowcost_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(json_meta, f, ensure_ascii=False, indent=2)

    print("\n===== Saved outputs =====")
    print(model_path)
    print(diag_path)
    print(imp_path)
    print(group_imp_path)
    print(os.path.join(OUTPUT_DIR, "lowcost_model_metrics.csv"))


if __name__ == "__main__":
    main()
