# MR-TADF machine-learning workflows

This repository contains the training, validation, molecule-generation, and
candidate-filtering workflows used for MR-TADF property prediction. The code
preserves the supplied scientific settings, including data order, splits,
random seeds, feature construction, preprocessing, model hyperparameters,
classification boundaries, generation parameters, and filtering order.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/                 # Training, core-definition, and test inputs
├── models/               # Six fitted model bundles
├── src/
│   ├── models/             # Six final model-training scripts
│   ├── validation/         # Cross-validation and external validation
│   └── generation/         # Molecule generation and filtering
├── reference_results/     # Supplied prediction and screening outputs
├── results/               # Locally generated outputs; ignored by Git
└── docs/                  # Repository notes and upload checklist
```

## Final models

| Property | Feature set | Training script | Fitted model |
| --- | --- | --- | --- |
| PL peak | low-cost | `src/models/train_pl_peak_lowcost.py` | `models/final_pl_peak_lowcost.pkl` |
| PL peak | augmented | `src/models/train_pl_peak_augmented.py` | `models/final_pl_peak_expensive.pkl` |
| PLQY | low-cost | `src/models/train_plqy_lowcost.py` | `models/final_plqy_rf_lowcost.pkl` |
| PLQY | augmented | `src/models/train_plqy_augmented.py` | `models/final_plqy_rf_expensive.pkl` |
| FWHM | low-cost | `src/models/train_fwhm_lowcost.py` | `models/final_fwhm_rf_lowcost.pkl` |
| FWHM | augmented | `src/models/train_fwhm_augmented.py` | `models/final_fwhm_rf_expensive.pkl` |

The PL-peak models are LightGBM regressors. The final PLQY and FWHM models are
random-forest classifiers, stored either directly or inside a scikit-learn
pipeline/model bundle.

Low-cost feature sets use quantities calculated from SMILES, including RDKit
2D descriptors, fingerprints, environment fields where present, and supplied
core/fragment definitions. Augmented feature sets additionally use available
electronic-structure or excited-state columns such as S1, T1, ΔEST,
oscillator strength, HOMO, LUMO, and lifetime-related values. The exact feature
definitions and feature order remain in each training script.

The positive classification rules are:

- high PLQY: `PLQY >= 80%`;
- narrow FWHM: `FWHM <= 30 nm`.

The training and generation workflows use their existing fixed random seeds,
including seed 42 where specified in the scripts.

## Installation

The verified environment uses Python 3.10.18. Create an isolated Python 3.10
environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded RDKit version is `2022.09.5`. LightGBM also requires a compatible
OpenMP runtime on platforms where it is not bundled.

## Input files

The workflows use:

- `data/tadf_dataset.xlsx` for model training and fragment learning;
- `data/core-structure info.xlsx` for core labels and core SMILES;
- `data/test molecules.xlsx` for external validation.

Original spreadsheet column names are preserved. Do not reorder rows when
reproducing the supplied data splits or results.

## Run the workflows

Run all commands from the repository root.

### Train the six final models

```bash
python src/models/train_pl_peak_lowcost.py
python src/models/train_pl_peak_augmented.py
python src/models/train_plqy_lowcost.py
python src/models/train_plqy_augmented.py
python src/models/train_fwhm_lowcost.py
python src/models/train_fwhm_augmented.py
```

Training outputs are written below `results/`. The fitted bundles already
provided in `models/` are used by the validation and filtering workflows.

### Run PL-peak five-fold cross-validation

```bash
python src/validation/pl_peak_lowcost_cv.py
python src/validation/pl_peak_augmented_cv.py
```

Both scripts retain the supplied 80:20 holdout and shuffled five-fold training
set validation procedure.

### Run external validation

```bash
python src/validation/external_validation.py
```

The script loads all six bundles from `models/`, reads
`data/test molecules.xlsx`, and writes tables and figures to
`results/external_validation/`.

### Generate molecules

Run the supplied launcher to preserve the default generation configuration:

```bash
python src/generation/run_generation.py
```

The defaults remain seed 42, target size 10,000, maximum size 50,000, and
500,000 generation attempts. Outputs are written to
`results/generated_scaffold_database/`.

### Filter generated molecules

```bash
python src/generation/filter_molecules.py \
  --input results/generated_scaffold_database/scaffold_based_generated_molecules.csv \
  --plpeak_model models/final_pl_peak_lowcost.pkl \
  --plqy_model models/final_plqy_rf_lowcost.pkl \
  --fwhm_model models/final_fwhm_rf_lowcost.pkl \
  --output_dir results/screened_lowcost_candidates
```

The filter retains the supplied PL-peak window, positive PLQY and FWHM class
requirements, probability cutoffs, strict `SA score < 6` criterion, ranking
formula, and tie-breaking order.

## Model files and reproducibility

The six fitted model files total approximately 4.4 MB and are stored directly
in this repository; Git LFS is not required. Only load pickle files obtained
from a trusted source. Model checksums and bundle-format notes are provided in
`models/README.md`.

See the [validation record](docs/VALIDATION.md), then review the
[GitHub upload checklist](docs/GITHUB_CHECKLIST.md) before publishing.
