# -*- coding: utf-8 -*-
import warnings
import os
from pathlib import Path


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier




def main():
    project_root = Path(__file__).resolve().parents[2]
    # =========================
    # =========================
    FILE_PATH = project_root / "data" / "tadf_dataset.xlsx"
    PLQY_THRESHOLD = 80
    RANDOM_STATE = 42
    DPI = 300
    OUTPUT_DIR = project_root / "results" / "plqy_augmented"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    results_all = []

    rf_oof_true = None
    rf_oof_pred = None
    rf_oof_prob = None
    rf_feature_importance_mean = None
    rf_feature_names = None


    # =========================
    # =========================
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


    def outpath(filename: str) -> str:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        return os.path.join(OUTPUT_DIR, filename)


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


    # =========================
    # =========================
    def calc_structure(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "MolWt": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        }


    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "unknown"
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


    def evaluate(y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_prob),
            "PR_AUC": average_precision_score(y_true, y_prob)
        }


    def build_models():
        return {
            "LogReg": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            "RF": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),


            "LGBM": LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1
            )
        }


    def build_pipeline(model):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if isinstance(model, LogisticRegression):
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)


    # =========================
    # =========================
    df = pd.read_excel(FILE_PATH)

    smiles_col = None
    for c in ["Smiles", "SMILES", "smiles"]:
        if c in df.columns:
            smiles_col = c
            break

    if smiles_col is None:
        raise ValueError("No SMILES column found; expected Smiles, SMILES, or smiles.")

    if "PLQY" not in df.columns:
        raise ValueError("No PLQY column found in the input table.")

    df = df[df["PLQY"].notna() & df[smiles_col].notna()].reset_index(drop=True)

    df["y"] = (df["PLQY"] >= PLQY_THRESHOLD).astype(int)

    feat_list = []
    valid_idx = []

    for i, smi in enumerate(df[smiles_col]):
        f = calc_structure(smi)
        if f is not None:
            feat_list.append(f)
            valid_idx.append(i)

    df = df.iloc[valid_idx].reset_index(drop=True)
    X_struct = pd.DataFrame(feat_list)

    phys_cols = [c for c in ["S1", "T1", "∆EST", "f", "τp", "HOMO", "LUMO"] if c in df.columns]
    X_phys = df[phys_cols].copy()

    X_dict = {
        "structure_only": X_struct,
        "physical_only": X_phys,
        "structure_plus_physical": pd.concat([X_struct, X_phys], axis=1)
    }

    y = df["y"].reset_index(drop=True)
    groups = df[smiles_col].apply(get_scaffold).reset_index(drop=True)

    print("Class distribution:")
    print(y.value_counts())
    print("\nAvailable physical feature columns:", phys_cols)


    # =========================
    # =========================
    for feature_set_name, X in X_dict.items():

        X = X.reset_index(drop=True)

        # ===== random split =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
        )

        for model_name, model in build_models().items():
            pipe = build_pipeline(model)
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            res = evaluate(y_test, y_pred, y_prob)

            results_all.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "split": "random",
                **res
            })

        # ===== scaffold split =====
        gkf = GroupKFold(n_splits=5)

        for model_name, model in build_models().items():
            fold_scores = []

            if feature_set_name == "structure_plus_physical" and model_name == "RF":
                oof_true = []
                oof_pred = []
                oof_prob = []
                fold_importances = []

            for tr, te in gkf.split(X, y, groups):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]

                pipe = build_pipeline(model)
                pipe.fit(X_tr, y_tr)

                y_pred = pipe.predict(X_te)
                y_prob = pipe.predict_proba(X_te)[:, 1]

                fold_scores.append(evaluate(y_te, y_pred, y_prob))

                if feature_set_name == "structure_plus_physical" and model_name == "RF":
                    oof_true.extend(y_te.tolist())
                    oof_pred.extend(y_pred.tolist())
                    oof_prob.extend(y_prob.tolist())
                    fold_importances.append(pipe.named_steps["model"].feature_importances_)

            avg = pd.DataFrame(fold_scores).mean().to_dict()

            results_all.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "split": "scaffold",
                **avg
            })

            if feature_set_name == "structure_plus_physical" and model_name == "RF":
                rf_oof_true = np.array(oof_true)
                rf_oof_pred = np.array(oof_pred)
                rf_oof_prob = np.array(oof_prob)
                rf_feature_importance_mean = np.mean(np.vstack(fold_importances), axis=0)
                rf_feature_names = X.columns.tolist()


    # =========================
    # =========================
    df_res = pd.DataFrame(results_all)
    df_res.to_csv(outpath("all_model_results_summary.csv"), index=False, encoding="utf-8-sig")
    print("\nSaved results summary: all_model_results_summary.csv")


    # =========================
    # =========================
    def extract(metric, split):
        d = df_res[df_res["split"] == split]

        def get_val(feature_set, model):
            val = d[
                (d["feature_set"] == feature_set) &
                (d["model"] == model)
            ][metric].values
            if len(val) == 0:
                raise ValueError(f"No result found for {metric} / {split} / {feature_set} / {model}.")
            return val[0]

        return {
            "LogReg": [
                get_val("structure_only", "LogReg"),
                get_val("physical_only", "LogReg"),
                get_val("structure_plus_physical", "LogReg"),
            ],
            "RF": [
                get_val("structure_only", "RF"),
                get_val("physical_only", "RF"),
                get_val("structure_plus_physical", "RF"),
            ],
            "LGBM": [
                get_val("structure_only", "LGBM"),
                get_val("physical_only", "LGBM"),
                get_val("structure_plus_physical", "LGBM"),
            ]
        }


    roc_random = extract("ROC_AUC", "random")
    f1_random = extract("F1", "random")
    roc_scaffold = extract("ROC_AUC", "scaffold")
    f1_scaffold = extract("F1", "scaffold")


    # =========================
    # =========================
    labels = ["Structure", "Physical", "Combined"]
    x = np.arange(len(labels))
    width = 0.22

    def plot_grouped_bar(ax, data_dict, ylabel, subtitle, ylim):
        model_names = list(data_dict.keys())
        colors = [COLOR_LOGREG, COLOR_RF, COLOR_LGBM]

        for i, (model, color) in enumerate(zip(model_names, colors)):
            values = data_dict[model]
            bars = ax.bar(
                x + (i - 1) * width,
                values,
                width=width,
                label=model,
                color=color,
                edgecolor="black",
                linewidth=0.8
            )

            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.008,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(subtitle, fontsize=14, pad=10)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        prettify_axis(ax, grid_axis="y")


    # =========================
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_grouped_bar(
        axes[0, 0], roc_random,
        ylabel="ROC-AUC",
        subtitle="(a) ROC-AUC under random split",
        ylim=(0.60, 1.00)
    )

    plot_grouped_bar(
        axes[0, 1], f1_random,
        ylabel="F1 score",
        subtitle="(b) F1 score under random split",
        ylim=(0.60, 0.92)
    )

    plot_grouped_bar(
        axes[1, 0], roc_scaffold,
        ylabel="ROC-AUC",
        subtitle="(c) ROC-AUC under scaffold split",
        ylim=(0.60, 0.90)
    )

    plot_grouped_bar(
        axes[1, 1], f1_scaffold,
        ylabel="F1 score",
        subtitle="(d) F1 score under scaffold split",
        ylim=(0.60, 0.87)
    )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath("Figure_model_comparison_random_scaffold.png"), dpi=DPI, bbox_inches="tight")
    plt.show()


    # =========================
    # =========================
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC", "PR-AUC"],
        "Value": [
            accuracy_score(rf_oof_true, rf_oof_pred),
            precision_score(rf_oof_true, rf_oof_pred),
            recall_score(rf_oof_true, rf_oof_pred),
            f1_score(rf_oof_true, rf_oof_pred),
            roc_auc_score(rf_oof_true, rf_oof_prob),
            average_precision_score(rf_oof_true, rf_oof_prob),
        ]
    })

    print("\n" + "=" * 65)
    print("Final model: RF + scaffold split + structure_plus_physical")
    print("=" * 65)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nClassification report:")
    print(classification_report(rf_oof_true, rf_oof_pred, digits=4))

    metrics_df.to_csv(outpath("Table_RF_scaffold_metrics.csv"), index=False, encoding="utf-8-sig")

    from sklearn.metrics import roc_curve, auc

    # ===== ROC curve for final RF + scaffold (OOF) =====
    fpr, tpr, _ = roc_curve(rf_oof_true, rf_oof_prob)
    roc_value = auc(fpr, tpr)

    plt.figure(figsize=(5.8, 5.2))
    plt.plot(fpr, tpr, lw=2.2, label=f'RF (AUC = {roc_value:.3f})', color='#2C7FB8')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False positive rate', fontsize=13)
    plt.ylabel('True positive rate', fontsize=13)
    plt.title('ROC curve of RF under scaffold split', fontsize=14, pad=10)
    plt.legend(frameon=True, fontsize=11, loc='lower right')
    plt.grid(ls='--', alpha=0.4)

    ax = plt.gca()
    prettify_axis(ax)

    plt.tight_layout()
    plt.savefig(outpath('Figure_ROC_curve_RF_scaffold.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ===== Confusion matrix (OOF) =====
    cm = confusion_matrix(rf_oof_true, rf_oof_pred)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Low PLQY', 'High PLQY']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')

    ax.set_title('Confusion matrix of RF (scaffold OOF)', fontsize=14, pad=10)
    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_ylabel('True label', fontsize=13)
    prettify_axis(ax)

    plt.tight_layout()
    plt.savefig(outpath('Figure_confusion_matrix_RF_scaffold.png'), dpi=300, bbox_inches='tight')
    plt.show()


    # =========================
    # Feature Importance
    # ===== feature name mapping for publication-style labels =====
    feature_label_map = {
        'S1': r'$S_1$',
        'T1': r'$T_1$',
        '∆EST': r'$\Delta E_{\mathrm{ST}}$',
        'ΔEST': r'$\Delta E_{\mathrm{ST}}$',
        'dEST': r'$\Delta E_{\mathrm{ST}}$',
        'f': r'$f$',
        'τp': r'$\tau_{\mathrm{p}}$',
        'tp': r'$\tau_{\mathrm{p}}$',
        'HOMO': r'$E_{\mathrm{HOMO}}$',
        'LUMO': r'$E_{\mathrm{LUMO}}$',
        'MolWt': r'$M_w$',
        'TPSA': 'TPSA',
        'MolLogP': r'$\log P$',
        'NumHDonors': r'$N_{\mathrm{HD}}$',
        'NumHAcceptors': r'$N_{\mathrm{HA}}$',
        'NumRotatableBonds': r'$N_{\mathrm{rot}}$',
        'RingCount': r'$N_{\mathrm{ring}}$',
        'HeavyAtomCount': r'$N_{\mathrm{heavy}}$',
        'NumAromaticRings': r'$N_{\mathrm{ArRing}}$',
        'FractionCSP3': r'$f_{\mathrm{CSP3}}$'
    }

    fi_df = pd.DataFrame({
        'Feature': rf_feature_names,
        'Importance': rf_feature_importance_mean
    }).sort_values('Importance', ascending=False)
    fi_df.to_csv(outpath('Table_RF_scaffold_feature_importance.csv'), index=False, encoding='utf-8-sig')

    top_n = 15
    fi_top = fi_df.head(top_n).copy()
    fi_top['Feature_label'] = fi_top['Feature'].map(lambda x: feature_label_map.get(x, x))
    fi_top = fi_top.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))

    ax.barh(
        fi_top['Feature_label'],
        fi_top['Importance'],
        color=MAIN_BLUE,
        edgecolor=MAIN_BLUE_DARK,
        linewidth=0.4
    )

    ax.set_xlabel('Feature importance', fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel('Feature', fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title('Top 15 feature importances', fontsize=INTERPRET_TITLE_SIZE, pad=10)

    ax.grid(axis='x', linestyle='--', alpha=0.35)
    prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)

    plt.tight_layout()
    plt.savefig(outpath('Figure_feature_importance_RF_scaffold_blue.png'), dpi=300, bbox_inches='tight')
    plt.show()


    print("\nTop 15 important features:")
    print(fi_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved files:")
    print(" - all_model_results_summary.csv")
    print(" - Figure_model_comparison_random_scaffold.png")
    print(" - Table_RF_scaffold_metrics.csv")
    print(" - Figure_confusion_matrix_RF_scaffold.png")
    print(" - Table_RF_scaffold_feature_importance.csv")
    print(" - Figure_feature_importance_RF_scaffold.png")

    # =========================
    # =========================
    # 8. SHAP analysis for final RF model
    # =========================
    import shap

    X_shap = X_dict["structure_plus_physical"].copy()
    y_shap = y.copy()

    rf_final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    rf_final_pipe = build_pipeline(rf_final_model)
    rf_final_pipe.fit(X_shap, y_shap)

    imputer = rf_final_pipe.named_steps["imputer"]
    rf_model_for_shap = rf_final_pipe.named_steps["model"]

    X_shap_imp = pd.DataFrame(
        imputer.transform(X_shap),
        columns=X_shap.columns,
        index=X_shap.index
    )

    feature_label_map = {
        "S1": r"$S_1$",
        "T1": r"$T_1$",
        "∆EST": r"$\Delta E_{\mathrm{ST}}$",
        "ΔEST": r"$\Delta E_{\mathrm{ST}}$",
        "f": r"$f$",
        "τp": r"$\tau_{\mathrm{p}}$",
        "tp": r"$\tau_{\mathrm{p}}$",
        "log_tau_p": r"log$_{10}(\tau_{\mathrm{p}})$",
        "HOMO": r"$E_{\mathrm{HOMO}}$",
        "LUMO": r"$E_{\mathrm{LUMO}}$",
        "MolWt": r"$M_w$",
        "TPSA": "TPSA",
        "MolLogP": r"$\log P$",
        "NumHDonors": r"$N_{\mathrm{HD}}$",
        "NumHAcceptors": r"$N_{\mathrm{HA}}$",
        "NumRotatableBonds": r"$N_{\mathrm{rot}}$",
        "RingCount": r"$N_{\mathrm{ring}}$",
        "HeavyAtomCount": r"$N_{\mathrm{heavy}}$",
        "NumAromaticRings": r"$N_{\mathrm{ArRing}}$",
        "FractionCSP3": r"$f_{\mathrm{CSP3}}$"
    }
    shap_feature_names = [feature_label_map.get(c, c) for c in X_shap_imp.columns]

    explainer = shap.TreeExplainer(rf_model_for_shap)
    shap_raw = explainer.shap_values(X_shap_imp)

    def get_positive_class_shap(shap_obj):
        """Return a 2D SHAP array for the positive (high-PLQY) class."""
        if isinstance(shap_obj, list):
            return np.array(shap_obj[1])

        if hasattr(shap_obj, "values"):
            vals = shap_obj.values
        else:
            vals = np.array(shap_obj)

        vals = np.array(vals)

        if vals.ndim == 2:
            return vals

        if vals.ndim == 3:
            if vals.shape[2] == 2:
                return vals[:, :, 1]
            elif vals.shape[0] == 2:
                return vals[1]
            else:
                raise ValueError(f"Unexpected 3D SHAP shape: {vals.shape}")

        raise ValueError(f"Unsupported SHAP shape: {vals.shape}")

    shap_values_pos = get_positive_class_shap(shap_raw)

    print("SHAP shape:", shap_values_pos.shape)

    plt.figure(figsize=(8.6, 6.8))
    shap.summary_plot(
        shap_values_pos,
        X_shap_imp,
        feature_names=shap_feature_names,
        show=False,
        max_display=15
    )

    fig = plt.gcf()
    ax = plt.gca()
    plt.title("SHAP summary plot of RF", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    enlarge_shap_fonts(fig, ax, "SHAP value")
    prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)
    plt.tight_layout()
    plt.savefig(outpath("Figure_SHAP_summary_RF.png"), dpi=300, bbox_inches="tight")
    plt.show()

    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)

    shap_bar_df = pd.DataFrame({
        "Feature": shap_feature_names,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)

    top_n = 15
    shap_bar_top = shap_bar_df.head(top_n).sort_values("MeanAbsSHAP", ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    ax.barh(
        shap_bar_top["Feature"],
        shap_bar_top["MeanAbsSHAP"],
        color=MAIN_BLUE,
        edgecolor=MAIN_BLUE_DARK,
        linewidth=0.4
    )

    ax.set_xlabel("Mean(|SHAP value|)", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Feature", fontsize=INTERPRET_LABEL_SIZE, labelpad=8)
    ax.set_title("Top 15 SHAP feature importances", fontsize=INTERPRET_TITLE_SIZE, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    prettify_axis(ax, labelsize=INTERPRET_TICK_SIZE)

    plt.tight_layout()
    plt.savefig(outpath("Figure_SHAP_bar_RF.png"), dpi=300, bbox_inches="tight")
    plt.show()

    shap_bar_df.to_csv(outpath("Table_SHAP_importance_RF.csv"), index=False, encoding="utf-8-sig")

    print("\nSHAP results saved:")
    print(" - Figure_SHAP_summary_RF.png")
    print(" - Figure_SHAP_bar_RF.png")
    print(" - Table_SHAP_importance_RF.csv")
    print("\nTop 15 SHAP features:")
    print(shap_bar_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # =========================
    # =========================
    import joblib
    import os

    SAVE_DIR = OUTPUT_DIR

    X_deploy = X_dict["structure_plus_physical"].copy()
    y_deploy = y.copy()

    rf_deploy_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    rf_deploy_pipe = build_pipeline(rf_deploy_model)
    rf_deploy_pipe.fit(X_deploy, y_deploy)

    joblib.dump(
        rf_deploy_pipe,
        os.path.join(SAVE_DIR, "plqy_rf_final_model.pkl")
    )
    pd.DataFrame({
        "feature": X_deploy.columns.tolist()
    }).to_csv(
        os.path.join(SAVE_DIR, "plqy_train_features_columns.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved deployment files:")
    print(" - plqy_rf_final_model.pkl")
    print(" - plqy_train_features_columns.csv")


if __name__ == "__main__":
    main()
