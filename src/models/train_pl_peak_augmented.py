import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def main():
    warnings.filterwarnings("ignore")
    project_root = Path(__file__).resolve().parents[2]
    # =========================================================
    # =========================================================
    USE_PLQY = False


    # =========================================================
    # =========================================================
    file_path = project_root / "data" / "tadf_dataset.xlsx"
    df = pd.read_excel(file_path)
    df.columns = [c.strip() for c in df.columns]

    OUTPUT_DIR = project_root / "results" / "plpeak_augmented"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SMILES_COL = "Smiles"
    TARGET_COL = "PL-peak"
    ENV_COL = "Solution"

    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()

    if ENV_COL in df.columns:
        df[ENV_COL] = df[ENV_COL].fillna("Unknown").astype(str).str.strip()
    else:
        df[ENV_COL] = "Unknown"

    print("Original data size:", len(df))

    df = df[(df[TARGET_COL] >= 400) & (df[TARGET_COL] <= 600)].copy()
    print("400–600 nm data size:", len(df))


    # =========================================================
    # =========================================================
    def featurize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = {}

        features["MolWt"] = Descriptors.MolWt(mol)
        features["TPSA"] = Descriptors.TPSA(mol)
        features["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
        features["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        features["NumHDonors"] = Descriptors.NumHDonors(mol)
        features["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        features["RingCount"] = Descriptors.RingCount(mol)
        features["FractionCSP3"] = Descriptors.FractionCSP3(mol)

        features["NumConjugatedBonds"] = sum(int(b.GetIsConjugated()) for b in mol.GetBonds())

        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        features["Count_B"] = atom_symbols.count("B")
        features["Count_N"] = atom_symbols.count("N")
        features["Count_O"] = atom_symbols.count("O")
        features["Count_S"] = atom_symbols.count("S")
        features["Count_F"] = atom_symbols.count("F")

        n_atoms = mol.GetNumAtoms()
        aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        sp2_like_atoms = sum(
            1 for a in mol.GetAtoms()
            if a.GetHybridization().name in ["SP2", "SP"]
        )

        features["AromaticAtomFrac"] = aromatic_atoms / (n_atoms + 1e-6)
        features["SP2AtomFrac"] = sp2_like_atoms / (n_atoms + 1e-6)
        features["HeteroAtomFrac"] = (
            (atom_symbols.count("B") + atom_symbols.count("N") +
             atom_symbols.count("O") + atom_symbols.count("S") +
             atom_symbols.count("F")) / (n_atoms + 1e-6)
        )

        # ---------- Morgan FP ----------
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        fp_array = np.array(list(fp), dtype=int)
        for i, bit in enumerate(fp_array):
            features[f"FP_{i}"] = bit

        return features


    feature_rows = []
    valid_idx = []

    for idx, smi in enumerate(df[SMILES_COL]):
        feat = featurize(smi)
        if feat is not None:
            feature_rows.append(feat)
            valid_idx.append(idx)

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    X_struct = pd.DataFrame(feature_rows)

    print("Valid molecules:", len(df_valid))
    print("Structural feature shape before FP filtering:", X_struct.shape)


    # =========================================================
    # =========================================================
    fp_cols = [c for c in X_struct.columns if c.startswith("FP_")]
    desc_cols = [c for c in X_struct.columns if not c.startswith("FP_")]

    fp_freq = X_struct[fp_cols].mean(axis=0)
    keep_fp_cols = fp_freq[fp_freq >= 0.03].index.tolist()

    X_struct = pd.concat(
        [X_struct[desc_cols].copy(), X_struct[keep_fp_cols].copy()],
        axis=1
    )

    print("FP before filtering:", len(fp_cols))
    print("FP after filtering :", len(keep_fp_cols))
    print("Structural feature shape after FP filtering:", X_struct.shape)


    # =========================================================
    # =========================================================
    X_env = pd.get_dummies(
        df_valid[[ENV_COL]],
        columns=[ENV_COL],
        prefix=[ENV_COL],
        drop_first=False
    )

    print("Environment feature shape:", X_env.shape)


    # =========================================================
    # =========================================================
    candidate_map = {
        "S1": ["S1"],
        "T1": ["T1"],
        "dEST": ["∆EST", "ΔEST", "dEST"],
        "taup": ["τp", "tp", "taup"],
        "f": ["f"],
        "LUMO": ["LUMO"],
        "HOMO": ["HOMO"],
        "PLQY": ["PLQY"]
    }

    def find_existing_col(possible_names, columns):
        for name in possible_names:
            if name in columns:
                return name
        return None

    resolved_cols = {}
    for key, names in candidate_map.items():
        resolved_cols[key] = find_existing_col(names, df_valid.columns)

    print("\nResolved electronic columns:")
    for k, v in resolved_cols.items():
        print(f"{k}: {v}")

    for key, col in resolved_cols.items():
        if col is not None:
            df_valid[key] = pd.to_numeric(df_valid[col], errors="coerce")
        else:
            df_valid[key] = np.nan


    # =========================================================
    # =========================================================
    df_valid["HL_gap"] = df_valid["LUMO"] - df_valid["HOMO"]
    df_valid["orbital_center"] = (df_valid["HOMO"] + df_valid["LUMO"]) / 2

    df_valid["f_div_dE"] = df_valid["f"] / (df_valid["dEST"] + 1e-6)
    df_valid["f_times_dE"] = df_valid["f"] * df_valid["dEST"]

    df_valid["log_taup"] = np.log10(df_valid["taup"] + 1e-6)

    elec_feature_cols = [
        "S1", "T1", "dEST", "f", "LUMO", "HOMO",
        "HL_gap", "orbital_center",
        "f_div_dE", "f_times_dE", "log_taup"
    ]

    if USE_PLQY:
        elec_feature_cols.append("PLQY")

    X_elec = df_valid[elec_feature_cols].copy()
    print("Electronic feature shape:", X_elec.shape)


    # =========================================================
    # =========================================================
    X_all = pd.concat(
        [
            X_struct.reset_index(drop=True),
            X_env.reset_index(drop=True),
            X_elec.reset_index(drop=True)
        ],
        axis=1
    )

    y = df_valid[TARGET_COL].values

    print("Final feature matrix shape:", X_all.shape)
    print("Using PLQY:", USE_PLQY)

    #-----------------------
    # =========================================================
    # =========================================================
    df_valid_clean = df_valid.reset_index(drop=True)
    X_all_clean = X_all.reset_index(drop=True)
    y_clean = df_valid_clean[TARGET_COL].values

    print("Using manually cleaned dataset; no automatic outlier detection is performed.")
    print("Cleaned size:", len(df_valid_clean))

    # =========================================================
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_clean, y_clean, test_size=0.2, random_state=42
    )

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))


    # =========================================================
    # =========================================================
    model_fs = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        reg_lambda=1.5,
        random_state=42
    )

    model_fs.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_fs.feature_importances_
    }).sort_values("importance", ascending=False)

    TOP_N = 30
    selected_features = importance_df.head(TOP_N)["feature"].tolist()

    print("\nTop selected features:")
    print(selected_features)


    # =========================================================
    # =========================================================
    X_train_sel = X_train[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=12,
        max_depth=4,
        min_child_samples=25,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42
    )

    model.fit(X_train_sel, y_train)

    pred_train = model.predict(X_train_sel)
    pred_test = model.predict(X_test_sel)

    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    tag = "with_PLQY" if USE_PLQY else "without_PLQY"

    print(f"\n===== FINAL RESULT ({tag}) =====")
    print(f"Train MAE:  {mae_train:.2f} nm")
    print(f"Train RMSE: {rmse_train:.2f} nm")
    print(f"Train R2:   {r2_train:.3f}")
    print(f"Test  MAE:  {mae_test:.2f} nm")
    print(f"Test  RMSE: {rmse_test:.2f} nm")
    print(f"Test  R2:   {r2_test:.3f}")



    # =========================================================
    # 4. Unified publication plotting style
    # =========================================================
    DPI = 300
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
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
        import os

        fig.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)

        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

        pdf_path = str(save_path).rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

        plt.close(fig)


    def plot_prediction_diagnostics(
        y_train, pred_train, y_test, pred_test,
        mae_train, mae_test, rmse_train, rmse_test,
        r2_train, r2_test,
        save_path: str,
    ):
        """Simple train/test fitting plot: no marginal distribution or residual panels."""
        metrics_df = pd.DataFrame({"split": ["train", "test"], "MAE_nm": [mae_train, mae_test], "RMSE_nm": [rmse_train, rmse_test], "R2": [r2_train, r2_test]})
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
            if not os.path.isabs(save_path):
                save_path = os.path.join(OUTPUT_DIR, save_path)
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            fig.savefig(str(save_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"SHAP plot failed and was skipped: {exc}")


    diag_name = f"train_test_diagnostics_{tag}.png"
    plot_prediction_diagnostics(
        y_train, pred_train, y_test, pred_test,
        mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test,
        save_path=diag_name
    )


    # =========================================================
    # =========================================================
    importance_sel = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    imp_name = f"feature_importance_top20_{tag}.png"
    group_imp_name = f"feature_group_importance_{tag}.png"
    shap_name = f"shap_summary_{tag}.png"

    plot_feature_importance(importance_sel, imp_name, top_n=20)
    plot_feature_group_importance(importance_sel, group_imp_name)
    plot_shap_summary(model, X_train_sel, shap_name, max_display=20)

    # Save prediction table in the same output folder
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
        os.path.join(OUTPUT_DIR, "train_test_predictions_expensive.csv"), index=False
    )

    print("\nSaved figures:")
    print(f"1. {os.path.join(OUTPUT_DIR, diag_name)}")
    print(f"2. {os.path.join(OUTPUT_DIR, imp_name)}")
    print(f"3. {os.path.join(OUTPUT_DIR, group_imp_name)}")
    print(f"4. {os.path.join(OUTPUT_DIR, shap_name)}  (if SHAP is installed)")
    print(f"5. {os.path.join(OUTPUT_DIR, 'train_test_predictions_expensive.csv')}")
    print(f"6. {os.path.join(OUTPUT_DIR, 'final_pl_peak_expensive.pkl')}")
    print(f"7. {os.path.join(OUTPUT_DIR, 'train_features_columns.csv')}")

    # =========================================================
    # =========================================================
    import joblib

    model_bundle = {
        "model": model,
        "selected_features": [str(c) for c in selected_features],
        "all_feature_columns": [str(c) for c in X_all_clean.columns],
        "metadata": {
            "task": "PL-peak regression",
            "feature_set": "expensive",
            "target_col": TARGET_COL,
            "smiles_col": SMILES_COL,
            "environment_col": ENV_COL,
            "use_plqy": USE_PLQY,
            "pl_peak_range_nm": [400, 600],
            "data_cleaning": "manual removal before running script",
            "n_cleaned": int(len(df_valid_clean)),
            "n_selected_features": int(len(selected_features)),
            "model_type": "LGBMRegressor",
            "random_state": 42,
        },
    }

    joblib.dump(model_bundle, os.path.join(OUTPUT_DIR, "final_pl_peak_expensive.pkl"))

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(OUTPUT_DIR, "train_features_columns.csv"), index=False
    )


if __name__ == "__main__":
    main()
