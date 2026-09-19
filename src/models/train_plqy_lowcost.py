"""
MR-TADF PLQY low-cost QSPR classification model
================================================
Purpose
-------
This script rebuilds the PLQY prediction model for VAE/CVAE-generated
molecules. It removes expensive TDDFT / experimental excited-state features
such as S1, T1, dEST, f, HOMO, LUMO and taup, and only uses low-cost features
that can be calculated directly from SMILES plus user-defined A/B/C core and
fragment structures.

Main output
-----------
plqy_lowcost_qspr_outputs_feature50/
    final_plqy_lowcost_rf_model.pkl
    selected_features_plqy_lowcost.csv
    all_feature_columns_plqy_lowcost.csv
    all_model_results_summary_plqy_lowcost.csv
    Table_RF_scaffold_metrics_plqy_lowcost.csv
    Figure_model_comparison_random_scaffold_plqy_lowcost.png
    Figure_ROC_curve_RF_scaffold_plqy_lowcost.png
    Figure_confusion_matrix_RF_scaffold_plqy_lowcost.png
    Figure_feature_importance_RF_lowcost_top20.png
    Figure_feature_group_importance_RF_lowcost.png

Notes
-----
1. This is a generation-screening model, not a final mechanistic model.
2. The model is core-aware: it reads A1-A4, B1-B7, C0-C5 from
   core-structure info.xlsx and creates motif match/count/similarity features.
3. The default final model is RF under scaffold OOF evaluation, consistent with
   the previous PLQY workflow, but using only low-cost features.
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# =========================================================
# 0. Basic settings
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = str(PROJECT_ROOT / "data" / "tadf_dataset.xlsx")
CORE_INFO_PATH = str(PROJECT_ROOT / "data" / "core-structure info.xlsx")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "plqy_lowcost")

PLQY_THRESHOLD = 80
BEST_THRESHOLD = 0.51
FEATURE_CONFIG = "base+MORGAN2"
SMILES_COL_CANDIDATES = ["Smiles", "SMILES", "smiles"]
TARGET_COL = "PLQY"
ENV_COL = "Solution"        # if absent, an Unknown environment column is used
MOLECULE_NAME_COL = "Molecules"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_SCAFFOLD = 5
DPI = 300

# Feature selection
TOP_N_FEATURES = 30
MIN_FEATURE_IMPORTANCE = 1      # remove zero-importance features

# Fingerprint settings
MORGAN_BITS = 1024
RDK_BITS = 1024
PATTERN_BITS = 1024
FP_MIN_FREQ = 0.03
FP_MAX_FREQ = 0.97

# Optional source-table categorical columns. These are not required.
OPTIONAL_CATEGORICAL_COLS = [
    "core_A_label", "core_B_label", "fragment_C_label",
    "primary_motif", "group_motif", "core_label",
    "Core", "core", "Motif", "motif",
]

CORE_TYPE_COL = "core-type"
CORE_SMILES_COL = "smiles"

# Fallback copied from the uploaded core-structure info.xlsx.
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

CORE_QUERY_INFO = {}


# =========================================================
# 1. Plot style
# =========================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLOR_LOGREG = "#4C72B0"
COLOR_RF = "#DD8452"
COLOR_LGBM = "#55A868"
MAIN_BLUE = "#2C7FB8"
MAIN_BLUE_DARK = "#1F5E8C"

# Larger fonts used specifically for SHAP and feature-importance figures.
INTERPRET_TICK_SIZE = 14
INTERPRET_LABEL_SIZE = 16
INTERPRET_TITLE_SIZE = 17
INTERPRET_COLORBAR_LABEL_SIZE = 15


def prettify_axis(ax, grid_axis=None, labelsize=11):
    """Closed-box publication style, consistent with PL-peak figures."""
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


# =========================================================
# 2. Utility functions
# =========================================================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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
    try:
        return len(mol.GetSubstructMatches(patt))
    except Exception:
        return 0


def bitvect_to_dict(bitvect, prefix: str) -> dict:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return {f"{prefix}_{i}": int(v) for i, v in enumerate(arr)}


def safe_descriptor(func, mol, default=np.nan):
    try:
        return safe_float(func(mol), default=default)
    except Exception:
        return default


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


def get_scaffold(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "unknown"
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaf if scaf else "no_scaffold"
    except Exception:
        return "unknown"


def find_existing_path(path: str, fallback_names: list[str]) -> str | None:
    candidates = []
    if path:
        candidates.append(path)
        candidates.append(os.path.basename(path))
    candidates.extend(fallback_names)
    candidates.extend([os.path.join(os.getcwd(), name) for name in fallback_names])
    candidates.extend([os.path.join("/mnt/data", name) for name in fallback_names])
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


# =========================================================
# 3. Core / fragment query loading
# =========================================================
def load_core_smiles(core_info_path: str = CORE_INFO_PATH) -> dict:
    existing_path = find_existing_path(
        core_info_path,
        ["core-structure info.xlsx", "core_structure_info.xlsx"]
    )
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
# 4. Low-cost feature engineering
# =========================================================
def featurize_lowcost(smiles: str) -> dict | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    features = {}
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # Basic 2D descriptors
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

    # Ring / rigidity / planarity proxies
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

    # Atom counts and fractions
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

    # Broad MR-TADF proxies
    features["has_B"] = int(features["Count_B"] >= 1)
    features["has_BN_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1)
    features["has_BO_proxy"] = int(features["Count_B"] >= 1 and features["Count_O"] >= 1)
    features["has_BS_proxy"] = int(features["Count_B"] >= 1 and features["Count_S"] >= 1)
    features["has_BNO_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1 and features["Count_O"] >= 1)
    features["has_BOS_proxy"] = int(features["Count_B"] >= 1 and features["Count_O"] >= 1 and features["Count_S"] >= 1)
    features["has_BNS_proxy"] = int(features["Count_B"] >= 1 and features["Count_N"] >= 1 and features["Count_S"] >= 1)

    # Functional group / substituent features
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

    # A/B/C motif features
    features.update(get_core_fragment_features(mol))

    # Fingerprints
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


def remove_sparse_fp_bits(X: pd.DataFrame):
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


def build_feature_matrix(df_raw: pd.DataFrame, smiles_col: str):
    feature_rows = []
    valid_rows = []

    for idx, smi in enumerate(df_raw[smiles_col]):
        feat = featurize_lowcost(smi)
        if feat is not None:
            feature_rows.append(feat)
            valid_rows.append(idx)

    df_valid = df_raw.iloc[valid_rows].reset_index(drop=True)
    X_struct = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_struct, keep_fp_cols, non_fp_cols = remove_sparse_fp_bits(X_struct)

    # Optimized external-validation feature setting:
    # keep only non-fingerprint low-cost descriptors/motif features + MORGAN2 bits.
    # Explicitly exclude MORGAN3, MACCS, RDKFP, PATTERN, and all expensive columns.
    fp_prefixes_to_exclude = ("MORGAN3_", "MACCS_", "RDKFP_", "PATTERN_")
    keep_cols = [
        c for c in X_struct.columns
        if not c.startswith(fp_prefixes_to_exclude)
    ]
    X_struct = X_struct[keep_cols].copy()
    keep_fp_cols = [c for c in keep_fp_cols if c.startswith("MORGAN2_")]

    if ENV_COL in df_valid.columns:
        env_raw = df_valid[[ENV_COL]].copy()
        env_raw[ENV_COL] = env_raw[ENV_COL].fillna("Unknown").astype(str).str.strip()
    else:
        env_raw = pd.DataFrame({ENV_COL: ["Unknown"] * len(df_valid)})
    X_env = pd.get_dummies(env_raw, columns=[ENV_COL], prefix=[ENV_COL], drop_first=False)

    X_cat = add_optional_categorical_features(df_valid)

    X_all = pd.concat(
        [X_struct.reset_index(drop=True), X_env.reset_index(drop=True), X_cat.reset_index(drop=True)],
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
    return df_valid, X_all, metadata


# =========================================================
# 5. Models, selection and evaluation
# =========================================================
def make_feature_selector(seed=RANDOM_STATE):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def select_top_features(X_train: pd.DataFrame, y_train: pd.Series, top_n=TOP_N_FEATURES):
    selector = make_feature_selector()
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    selector.fit(X_train_clean, y_train)
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": selector.feature_importances_,
    }).sort_values("importance", ascending=False)

    if MIN_FEATURE_IMPORTANCE > 0:
        importances = importances[importances["importance"] >= MIN_FEATURE_IMPORTANCE * 1e-12]

    selected = importances.head(top_n)["feature"].tolist()
    if not selected:
        selected = X_train.columns.tolist()[:min(top_n, X_train.shape[1])]
    return selected, importances


def build_models():
    return {
        "LogReg": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=10,
            max_depth=4,
            min_child_samples=25,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=1.5,
            reg_lambda=4.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def build_pipeline(model):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if isinstance(model, LogisticRegression):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def safe_binary_metrics(y_true, y_pred, y_prob):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        out["PR_AUC"] = average_precision_score(y_true, y_prob)
    else:
        out["ROC_AUC"] = np.nan
        out["PR_AUC"] = np.nan
    return out


def get_model_feature_importance(pipe, feature_names):
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).ravel()
    return np.zeros(len(feature_names), dtype=float)


def evaluate_random_split(X_all, y):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    selected_features, imp_df = select_top_features(X_train, y_train)

    for model_name, model in build_models().items():
        pipe = build_pipeline(model)
        pipe.fit(X_train[selected_features], y_train)
        y_pred = pipe.predict(X_test[selected_features])
        y_prob = pipe.predict_proba(X_test[selected_features])[:, 1]
        results.append({
            "feature_set": f"lowcost_top{len(selected_features)}",
            "model": model_name,
            "split": "random",
            **safe_binary_metrics(y_test, y_pred, y_prob),
        })
    return results, selected_features, imp_df


def evaluate_scaffold_oof(X_all, y, groups):
    results = []
    gkf = GroupKFold(n_splits=N_SPLITS_SCAFFOLD)

    final_rf_true, final_rf_pred, final_rf_prob = [], [], []
    final_rf_fold_importance_rows = []
    final_rf_selected_counter = Counter()
    fold_selected_features = []

    for model_name, base_model in build_models().items():
        fold_scores = []
        if model_name == "RF":
            final_rf_true, final_rf_pred, final_rf_prob = [], [], []
            final_rf_fold_importance_rows = []
            final_rf_selected_counter = Counter()
            fold_selected_features = []

        for fold, (tr, te) in enumerate(gkf.split(X_all, y, groups), 1):
            X_tr, X_te = X_all.iloc[tr].reset_index(drop=True), X_all.iloc[te].reset_index(drop=True)
            y_tr, y_te = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)

            selected_features, _ = select_top_features(X_tr, y_tr)
            X_tr_sel = X_tr[selected_features]
            X_te_sel = X_te[selected_features]

            pipe = build_pipeline(base_model)
            pipe.fit(X_tr_sel, y_tr)
            y_pred = pipe.predict(X_te_sel)
            y_prob = pipe.predict_proba(X_te_sel)[:, 1]
            fold_scores.append(safe_binary_metrics(y_te, y_pred, y_prob))

            if model_name == "RF":
                final_rf_true.extend(y_te.tolist())
                final_rf_pred.extend(y_pred.tolist())
                final_rf_prob.extend(y_prob.tolist())
                final_rf_selected_counter.update(selected_features)
                fold_selected_features.append(selected_features)

                fi = get_model_feature_importance(pipe, selected_features)
                final_rf_fold_importance_rows.append(
                    pd.DataFrame({
                        "fold": fold,
                        "feature": selected_features,
                        "importance": fi,
                    })
                )

        avg = pd.DataFrame(fold_scores).mean(numeric_only=True).to_dict()
        results.append({
            "feature_set": f"lowcost_top{TOP_N_FEATURES}_foldwise",
            "model": model_name,
            "split": "scaffold",
            **avg,
        })

    if final_rf_fold_importance_rows:
        rf_imp_long = pd.concat(final_rf_fold_importance_rows, axis=0, ignore_index=True)
        rf_imp_avg = (
            rf_imp_long.groupby("feature", as_index=False)["importance"].mean()
            .sort_values("importance", ascending=False)
        )
        rf_imp_avg["selection_frequency"] = rf_imp_avg["feature"].map(final_rf_selected_counter) / N_SPLITS_SCAFFOLD
    else:
        rf_imp_avg = pd.DataFrame(columns=["feature", "importance", "selection_frequency"])

    final_data = {
        "rf_oof_true": np.asarray(final_rf_true),
        "rf_oof_pred": np.asarray(final_rf_pred),
        "rf_oof_prob": np.asarray(final_rf_prob),
        "rf_oof_importance": rf_imp_avg,
        "rf_fold_selected_features": fold_selected_features,
    }
    return results, final_data


# =========================================================
# 6. Plotting functions
# =========================================================
def plot_grouped_bar(ax, data_dict, ylabel, subtitle, ylim):
    labels = ["Low-cost"]
    x = np.arange(len(labels))
    width = 0.22
    colors = [COLOR_LOGREG, COLOR_RF, COLOR_LGBM]

    for i, (model, color) in enumerate(zip(list(data_dict.keys()), colors)):
        values = data_dict[model]
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, v in zip(bars, values):
            if pd.isna(v):
                label = "NA"
                ypos = 0.02
            else:
                label = f"{v:.2f}"
                ypos = v + 0.008
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(subtitle, fontsize=14, pad=10)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    prettify_axis(ax, grid_axis="y")


def plot_model_comparison(df_res, output_path):
    def extract(metric, split):
        d = df_res[df_res["split"] == split]
        out = {}
        for model in ["LogReg", "RF", "LGBM"]:
            vals = d[d["model"] == model][metric].values
            out[model] = [vals[0] if len(vals) else np.nan]
        return out

    roc_random = extract("ROC_AUC", "random")
    f1_random = extract("F1", "random")
    roc_scaffold = extract("ROC_AUC", "scaffold")
    f1_scaffold = extract("F1", "scaffold")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    plot_grouped_bar(axes[0, 0], roc_random, "ROC-AUC", "(a) ROC-AUC under random split", (0.50, 1.00))
    plot_grouped_bar(axes[0, 1], f1_random, "F1 score", "(b) F1 score under random split", (0.50, 1.00))
    plot_grouped_bar(axes[1, 0], roc_scaffold, "ROC-AUC", "(c) ROC-AUC under scaffold split", (0.50, 1.00))
    plot_grouped_bar(axes[1, 1], f1_scaffold, "F1 score", "(d) F1 score under scaffold split", (0.50, 1.00))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def feature_to_label(feature: str) -> str:
    label_map = {
        "MolWt": r"$M_w$",
        "ExactMolWt": r"Exact $M_w$",
        "TPSA": "TPSA",
        "MolLogP": r"$\log P$",
        "MolMR": "MolMR",
        "NumHDonors": r"$N_{\mathrm{HD}}$",
        "NumHAcceptors": r"$N_{\mathrm{HA}}$",
        "NumRotatableBonds": r"$N_{\mathrm{rot}}$",
        "RingCount": r"$N_{\mathrm{ring}}$",
        "HeavyAtomCount": r"$N_{\mathrm{heavy}}$",
        "NumAromaticRings": r"$N_{\mathrm{ArRing}}$",
        "FractionCSP3": r"$f_{\mathrm{CSP3}}$",
        "SP2AtomFrac": r"sp$^2$ atom fraction",
        "AromaticAtomFrac": "Aromatic atom fraction",
        "HeteroAtomFrac": "Heteroatom fraction",
        "AromaticBondFrac": "Aromatic bond fraction",
        "ConjugatedBondFrac": "Conjugated bond fraction",
        "RotatableBondFrac": "Rotatable bond fraction",
        "NumConjugatedBonds": "Number of conjugated bonds",
        "NumAromaticBonds": "Number of aromatic bonds",
        "FusedRingPairCount": "Fused ring pair count",
        "AromaticRingSystemCount": "Aromatic ring system count",
        "BridgeheadAtomCount": "Bridgehead atom count",
        "SpiroAtomCount": "Spiro atom count",
    }
    if feature in label_map:
        return label_map[feature]
    if feature.startswith("Count_"):
        return "Count " + feature.replace("Count_", "")
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
    return feature.replace("_", " ")


def categorize_feature(feature: str) -> str:
    if feature.startswith("RDKFP_"):
        return "RDKit fingerprint"
    if feature.startswith("MORGAN"):
        return "Morgan fingerprint"
    if feature.startswith("MACCS_"):
        return "MACCS fingerprint"
    if feature.startswith("PATTERN_"):
        return "Pattern fingerprint"
    if feature.startswith("motif_sim_") or feature.startswith("motif_max_sim_"):
        return "A/B/C motif similarity"
    if feature.startswith("motif_match_") or feature.startswith("motif_count_"):
        return "A/B/C motif match/count"
    if feature.startswith("motif_any_") or feature.startswith("motif_n_") or feature.startswith("motif_total_"):
        return "A/B/C motif summary"
    if feature.startswith("fg_"):
        return "Functional group"
    if feature.startswith(f"{ENV_COL}_"):
        return "Environment"
    if any(feature.startswith(c + "_") for c in OPTIONAL_CATEGORICAL_COLS):
        return "Optional categorical label"
    return "2D descriptor"


def plot_feature_importance(fi_df, output_dir):
    fi_df = fi_df.copy().sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_dir, "Table_RF_lowcost_feature_importance.csv"), index=False, encoding="utf-8-sig")

    top_n = 20
    fi_top = fi_df.head(top_n).copy()
    fi_top["Feature_label"] = fi_top["feature"].map(feature_to_label)
    fi_top = fi_top.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    ax.barh(fi_top["Feature_label"], fi_top["importance"], color=MAIN_BLUE, edgecolor=MAIN_BLUE_DARK, linewidth=0.4)
    ax.set_xlabel("Feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title("Top 20 feature importances", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_feature_importance_RF_lowcost_top20.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    fi_df["feature_group"] = fi_df["feature"].map(categorize_feature)
    group_imp = (
        fi_df.groupby("feature_group", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=True)
    )
    group_imp.to_csv(os.path.join(output_dir, "Table_feature_group_importance_plqy_lowcost.csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.barh(group_imp["feature_group"], group_imp["importance"], color=MAIN_BLUE, edgecolor=MAIN_BLUE_DARK, linewidth=0.4)
    ax.set_xlabel("Summed feature importance", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature group", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title("Feature group importance", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_feature_group_importance_RF_lowcost.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    return fi_df, group_imp


def get_positive_class_shap(shap_obj):
    """
    Convert SHAP output to a 2D array with shape (n_samples, n_features),
    using the positive class (High PLQY) for binary classification.
    Compatible with both older and newer SHAP versions.
    """
    # Older SHAP: list of arrays, usually [class0, class1]
    if isinstance(shap_obj, list):
        if len(shap_obj) == 2:
            return np.asarray(shap_obj[1])
        return np.asarray(shap_obj[-1])

    # Newer SHAP may return an Explanation object
    if hasattr(shap_obj, "values"):
        vals = np.asarray(shap_obj.values)
    else:
        vals = np.asarray(shap_obj)

    if vals.ndim == 2:
        return vals

    # Common binary-classification shapes:
    # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    if vals.ndim == 3:
        if vals.shape[2] == 2:
            return vals[:, :, 1]
        if vals.shape[0] == 2:
            return vals[1, :, :]

    raise ValueError(f"Unsupported SHAP output shape: {vals.shape}")


def plot_shap_analysis(model_package, X_all, output_dir):
    """
    SHAP analysis for the final deployment RF model.

    This follows the same logic as the expensive-version script:
    1. Use the final RF model refitted on the full dataset for interpretation.
    2. Apply the pipeline imputer before TreeExplainer.
    3. Export a SHAP beeswarm summary, a blue publication-style SHAP bar plot,
       and a CSV table of mean absolute SHAP values.
    """
    try:
        import shap
    except ImportError:
        print("\nSHAP is not installed. Please install it with: pip install shap")
        print("SHAP analysis skipped.")
        return None

    selected_features = list(model_package["selected_features"])
    deploy_pipe = model_package["model"]

    X_shap = X_all[selected_features].copy().replace([np.inf, -np.inf], np.nan)

    imputer = deploy_pipe.named_steps["imputer"]
    rf_model_for_shap = deploy_pipe.named_steps["model"]

    X_shap_imp = pd.DataFrame(
        imputer.transform(X_shap),
        columns=selected_features,
        index=X_shap.index,
    )

    shap_feature_names = [feature_to_label(c) for c in selected_features]

    explainer = shap.TreeExplainer(rf_model_for_shap)
    shap_raw = explainer.shap_values(X_shap_imp)
    shap_values_pos = get_positive_class_shap(shap_raw)

    print("SHAP shape:", shap_values_pos.shape)

    # -------- 1. SHAP summary beeswarm plot --------
    plt.figure(figsize=(8.8, 7.4))
    shap.summary_plot(
        shap_values_pos,
        X_shap_imp,
        feature_names=shap_feature_names,
        show=False,
        max_display=20,
    )
    fig = plt.gcf()
    ax = plt.gca()
    plt.title("SHAP summary plot of RF", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    enlarge_shap_fonts(fig, ax, "SHAP value")
    prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_SHAP_summary_RF_lowcost.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    # -------- 2. SHAP importance bar plot --------
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    shap_bar_df = pd.DataFrame({
        "Feature": selected_features,
        "Feature_label": shap_feature_names,
        "MeanAbsSHAP": mean_abs_shap,
    }).sort_values("MeanAbsSHAP", ascending=False)

    top_n = 20
    shap_bar_top = shap_bar_df.head(top_n).sort_values("MeanAbsSHAP", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    ax.barh(
        shap_bar_top["Feature_label"],
        shap_bar_top["MeanAbsSHAP"],
        color=MAIN_BLUE,
        edgecolor=MAIN_BLUE_DARK,
        linewidth=0.4,
    )
    ax.set_xlabel("Mean(|SHAP value|)", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title("Top 20 SHAP feature importances", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    prettify_axis(ax, grid_axis="x", labelsize=INTERPRET_TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_SHAP_bar_RF_lowcost.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    # -------- 3. Export table --------
    shap_bar_df.to_csv(
        os.path.join(output_dir, "Table_SHAP_importance_RF_lowcost.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSHAP results saved:")
    print(" - Figure_SHAP_summary_RF_lowcost.png")
    print(" - Figure_SHAP_bar_RF_lowcost.png")
    print(" - Table_SHAP_importance_RF_lowcost.csv")
    print("\nTop 20 SHAP features:")
    print(shap_bar_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return shap_bar_df


def plot_roc_and_confusion(y_true, y_pred, y_prob, output_dir):
    # ROC
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_value = auc(fpr, tpr)
    else:
        fpr, tpr, roc_value = [0, 1], [0, 1], np.nan

    plt.figure(figsize=(5.8, 5.2))
    plt.plot(fpr, tpr, lw=2.2, label=f"RF (AUC = {roc_value:.3f})", color=MAIN_BLUE)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False positive rate", fontsize=13)
    plt.ylabel("True positive rate", fontsize=13)
    plt.title("ROC curve of RF under scaffold split", fontsize=14, pad=10)
    plt.legend(frameon=True, fontsize=11, loc="lower right")
    ax = plt.gca()
    prettify_axis(ax)
    ax.grid(linestyle="--", linewidth=0.6, alpha=0.30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_ROC_curve_RF_scaffold_plqy_lowcost.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low PLQY", "High PLQY"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion matrix of RF (scaffold OOF)", fontsize=14, pad=10)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_confusion_matrix_RF_scaffold_plqy_lowcost.png"), dpi=DPI, bbox_inches="tight")
    plt.close()


# =========================================================
# 7. Final deployment model
# =========================================================
def fit_deployment_model(X_all, y, metadata, output_dir):
    selected_features, selector_importance = select_top_features(X_all, y)
    selector_importance.to_csv(os.path.join(output_dir, "feature_selection_importance_full_data.csv"), index=False, encoding="utf-8-sig")

    rf_model = build_models()["RF"]
    deploy_pipe = build_pipeline(rf_model)
    deploy_pipe.fit(X_all[selected_features], y)

    # Final RF feature importance on full-data selected features
    fi = get_model_feature_importance(deploy_pipe, selected_features)
    deploy_fi = pd.DataFrame({
        "feature": selected_features,
        "importance": fi,
    }).sort_values("importance", ascending=False)
    deploy_fi["importance_norm"] = deploy_fi["importance"] / (deploy_fi["importance"].sum() + 1e-12)

    model_package = {
        "model": deploy_pipe,
        "selected_features": selected_features,
        "all_feature_columns": metadata["all_feature_columns"],
        "metadata": metadata,
        "plqy_threshold": PLQY_THRESHOLD,
        "best_threshold": BEST_THRESHOLD,
        "feature_config": FEATURE_CONFIG,
        "feature_generator": "MR_TADF_PLQY_lowcost_QSPR_coreaware_feature50.py",
        "note": "Use the same low-cost featurization, reindex to selected_features, and classify with probability >= BEST_THRESHOLD.",
    }
    joblib.dump(model_package, os.path.join(output_dir, "final_plqy_lowcost_rf_model_external_optimized.pkl"))

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(output_dir, "selected_features_plqy_lowcost.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"feature": metadata["all_feature_columns"]}).to_csv(
        os.path.join(output_dir, "all_feature_columns_plqy_lowcost.csv"), index=False, encoding="utf-8-sig"
    )

    meta_to_save = dict(metadata)
    meta_to_save.update({
        "TOP_N_FEATURES": TOP_N_FEATURES,
        "PLQY_THRESHOLD": PLQY_THRESHOLD,
        "RANDOM_STATE": RANDOM_STATE,
        "final_model": "RandomForestClassifier",
        "best_threshold": BEST_THRESHOLD,
        "feature_config": FEATURE_CONFIG,
        "optimized_rf_params": {"n_estimators": 80, "max_depth": 5, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": "sqrt", "class_weight": None},
        "expensive_features_removed": ["S1", "T1", "∆EST", "ΔEST", "dEST", "f", "τp", "tp", "HOMO", "LUMO"],
    })
    with open(os.path.join(output_dir, "lowcost_plqy_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_to_save, f, ensure_ascii=False, indent=2)

    return model_package, deploy_fi


# =========================================================
# 8. Main
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    data_path = find_existing_path(FILE_PATH, ["tadf_dataset_PLQY.xlsx"])
    if data_path is None:
        raise FileNotFoundError(
            "Cannot find PLQY data file. Please set FILE_PATH to your tadf_dataset_PLQY.xlsx path."
        )

    df = pd.read_excel(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    smiles_col = None
    for c in SMILES_COL_CANDIDATES:
        if c in df.columns:
            smiles_col = c
            break
    if smiles_col is None:
        raise ValueError("No SMILES column found. Expected one of: Smiles / SMILES / smiles")
    if TARGET_COL not in df.columns:
        raise ValueError("No PLQY column found. Please check the source table.")

    df = df[df[TARGET_COL].notna() & df[smiles_col].notna()].reset_index(drop=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    df["y"] = (df[TARGET_COL] >= PLQY_THRESHOLD).astype(int)

    print("Original data size:", len(df))
    print("Class distribution, 0=Low PLQY, 1=High PLQY:")
    print(df["y"].value_counts())

    # Explicitly warn if expensive columns exist; they are not used.
    expensive_cols = [c for c in ["S1", "T1", "∆EST", "ΔEST", "dEST", "f", "τp", "tp", "HOMO", "LUMO"] if c in df.columns]
    print("Expensive columns found but intentionally excluded:", expensive_cols)

    initialize_core_queries(CORE_INFO_PATH)
    df_valid, X_all, metadata = build_feature_matrix(df, smiles_col)
    y = df_valid["y"].reset_index(drop=True)
    groups = df_valid[smiles_col].apply(get_scaffold).reset_index(drop=True)

    print("Valid molecules:", len(df_valid))
    print("Low-cost feature matrix shape before selection:", X_all.shape)

    # Save source feature matrix columns and valid index info
    pd.DataFrame({"feature": X_all.columns.tolist()}).to_csv(
        os.path.join(OUTPUT_DIR, "all_feature_columns_before_selection.csv"), index=False, encoding="utf-8-sig"
    )

    # Random split and scaffold OOF evaluation
    random_results, random_selected_features, random_imp = evaluate_random_split(X_all, y)
    scaffold_results, final_data = evaluate_scaffold_oof(X_all, y, groups)
    df_res = pd.DataFrame(random_results + scaffold_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "all_model_results_summary_plqy_lowcost.csv"), index=False, encoding="utf-8-sig")

    print("\n===== Model comparison summary =====")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_model_comparison(df_res, os.path.join(OUTPUT_DIR, "Figure_model_comparison_random_scaffold_plqy_lowcost.png"))

    rf_oof_true = final_data["rf_oof_true"]
    rf_oof_pred = final_data["rf_oof_pred"]
    rf_oof_prob = final_data["rf_oof_prob"]

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1 score", "ROC-AUC", "PR-AUC"],
        "Value": [
            accuracy_score(rf_oof_true, rf_oof_pred),
            balanced_accuracy_score(rf_oof_true, rf_oof_pred),
            precision_score(rf_oof_true, rf_oof_pred, zero_division=0),
            recall_score(rf_oof_true, rf_oof_pred, zero_division=0),
            f1_score(rf_oof_true, rf_oof_pred, zero_division=0),
            roc_auc_score(rf_oof_true, rf_oof_prob) if len(np.unique(rf_oof_true)) == 2 else np.nan,
            average_precision_score(rf_oof_true, rf_oof_prob) if len(np.unique(rf_oof_true)) == 2 else np.nan,
        ]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "Table_RF_scaffold_metrics_plqy_lowcost.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("Final OOF model: RF + scaffold split + low-cost selected features")
    print("=" * 72)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClassification report:")
    print(classification_report(rf_oof_true, rf_oof_pred, digits=4))

    plot_roc_and_confusion(rf_oof_true, rf_oof_pred, rf_oof_prob, OUTPUT_DIR)

    # Fold-wise RF importance table
    final_data["rf_oof_importance"].to_csv(
        os.path.join(OUTPUT_DIR, "Table_RF_scaffold_OOF_feature_importance_foldwise.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Deployment model and final full-data importance
    model_package, deploy_fi = fit_deployment_model(X_all, y, metadata, OUTPUT_DIR)
    plot_feature_importance(deploy_fi, OUTPUT_DIR)
    plot_shap_analysis(model_package, X_all, OUTPUT_DIR)

    print("\nSelected deployment features:", len(model_package["selected_features"]))
    print(model_package["selected_features"])
    print("\nTop 20 deployment RF features:")
    print(deploy_fi.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved files in:", OUTPUT_DIR)
    print(" - final_plqy_lowcost_rf_model_external_optimized.pkl")
    print(" - selected_features_plqy_lowcost.csv")
    print(" - all_model_results_summary_plqy_lowcost.csv")
    print(" - Table_RF_scaffold_metrics_plqy_lowcost.csv")
    print(" - Figure_model_comparison_random_scaffold_plqy_lowcost.png")
    print(" - Figure_ROC_curve_RF_scaffold_plqy_lowcost.png")
    print(" - Figure_confusion_matrix_RF_scaffold_plqy_lowcost.png")
    print(" - Figure_feature_importance_RF_lowcost_top20.png")
    print(" - Figure_feature_group_importance_RF_lowcost.png")
    print(" - Figure_SHAP_summary_RF_lowcost.png")
    print(" - Figure_SHAP_bar_RF_lowcost.png")
    print(" - Table_SHAP_importance_RF_lowcost.csv")


if __name__ == "__main__":
    main()
