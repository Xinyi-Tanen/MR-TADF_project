#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run file for scaffold-constrained fragment recombination generation.

Usage:
    python run_scaffold_generation.py

Input data are read from the project's ``data`` directory. Generated files are
written below ``results``.
"""

from pathlib import Path
import subprocess
import sys


# =========================
# User settings
# =========================
DATASET_FILE = "tadf_dataset.xlsx"
CORE_FILE = "core-structure info.xlsx"
GENERATOR_SCRIPT = "generate_molecules.py"

TARGET_SIZE = 10000
MAX_SIZE = 50000
MAX_ATTEMPTS = 500000
OUTPUT_DIR = "generated_scaffold_database"

# If you want to expand generation space by allowing fragments learned from
# other A/B scaffolds to be remapped, change this to True.
ALLOW_GLOBAL_REMAP = False


# =========================
# Do not usually edit below
# =========================
def check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {description}: {path}\n"
            f"Please put this run file in the same folder as the required files, "
            f"or edit the path settings at the top of run_scaffold_generation.py."
        )


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parents[1]

    dataset_path = project_root / "data" / DATASET_FILE
    core_path = project_root / "data" / CORE_FILE
    generator_path = code_dir / GENERATOR_SCRIPT
    output_path = project_root / "results" / OUTPUT_DIR

    check_file(dataset_path, "dataset file")
    check_file(core_path, "core-structure file")
    check_file(generator_path, "generator script")

    cmd = [
        sys.executable,
        str(generator_path),
        "--dataset", str(dataset_path),
        "--cores", str(core_path),
        "--target_size", str(TARGET_SIZE),
        "--max_size", str(MAX_SIZE),
        "--max_attempts", str(MAX_ATTEMPTS),
        "--output_dir", str(output_path),
    ]

    if ALLOW_GLOBAL_REMAP:
        cmd.append("--allow_global_remap")

    print("Running scaffold-constrained database generation...")
    print("Command:")
    print(" ".join([f'\"{x}\"' if " " in x else x for x in cmd]))
    print("\nOutput directory:", output_path)
    print("-" * 80)

    result = subprocess.run(cmd, cwd=str(project_root), check=False)

    print("-" * 80)
    if result.returncode == 0:
        print("Generation finished successfully.")
        print(f"Please check output files in: {output_path}")
    else:
        print(f"Generation failed with return code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
