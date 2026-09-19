#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaffold-constrained fragment recombination for MR-TADF molecules.

The workflow identifies A/B core scaffolds, learns pendant and fused/bridged
modifications from the source dataset, builds site-aware fragment libraries,
and randomly recombines learned patterns. Candidates are sanitized with RDKit,
canonicalized, deduplicated, and checked for novelty against the training set.

This script generates structures only. Property-model screening is performed by
``filter_molecules.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs, AllChem

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


# -----------------------------
# Utilities
# -----------------------------


def json_safe(obj):
    """Convert numpy/pandas scalar containers into JSON-serializable Python objects."""
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj

def safe_mol_from_smiles(smi: Any, sanitize: bool = True) -> Optional[Chem.Mol]:
    if smi is None:
        return None
    if not isinstance(smi, str):
        smi = str(smi)
    smi = smi.strip()
    if not smi or smi.lower() in {"nan", "none", "null"}:
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
        return mol
    except Exception:
        return None


def canonical_smiles(smi_or_mol: Any) -> Optional[str]:
    try:
        mol = smi_or_mol if isinstance(smi_or_mol, Chem.Mol) else safe_mol_from_smiles(smi_or_mol)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def bond_type_to_name(bt: Chem.BondType) -> str:
    if bt == Chem.BondType.SINGLE:
        return "SINGLE"
    if bt == Chem.BondType.DOUBLE:
        return "DOUBLE"
    if bt == Chem.BondType.TRIPLE:
        return "TRIPLE"
    if bt == Chem.BondType.AROMATIC:
        return "AROMATIC"
    return str(bt)


def name_to_bond_type(name: str) -> Chem.BondType:
    name = str(name).upper()
    if "AROMATIC" in name:
        return Chem.BondType.AROMATIC
    if "DOUBLE" in name:
        return Chem.BondType.DOUBLE
    if "TRIPLE" in name:
        return Chem.BondType.TRIPLE
    return Chem.BondType.SINGLE


def atom_to_spec(atom: Chem.Atom) -> Dict[str, Any]:
    return {
        "atomic_num": int(atom.GetAtomicNum()),
        "formal_charge": int(atom.GetFormalCharge()),
        "is_aromatic": bool(atom.GetIsAromatic()),
        "chiral_tag": int(atom.GetChiralTag()),
        "num_explicit_hs": int(atom.GetNumExplicitHs()),
        "no_implicit": bool(atom.GetNoImplicit()),
    }


def spec_to_atom(spec: Dict[str, Any]) -> Chem.Atom:
    atom = Chem.Atom(int(spec["atomic_num"]))
    atom.SetFormalCharge(int(spec.get("formal_charge", 0)))
    atom.SetIsAromatic(bool(spec.get("is_aromatic", False)))
    try:
        atom.SetChiralTag(Chem.ChiralType(int(spec.get("chiral_tag", 0))))
    except Exception:
        pass
    atom.SetNumExplicitHs(int(spec.get("num_explicit_hs", 0)))
    atom.SetNoImplicit(bool(spec.get("no_implicit", False)))
    return atom


def normalise_colname(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).lower()


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {normalise_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalise_colname(cand)
        if key in lookup:
            return lookup[key]
    return None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CoreInfo:
    label: str
    smiles: str
    canonical_smiles: str
    mol: Chem.Mol
    n_atoms: int
    n_hetero: int


@dataclass
class ModificationSpec:
    mod_id: str
    source_index: int
    source_molecule: str
    source_smiles: str
    source_canonical_smiles: str
    scaffold_label: str
    modification_type: str
    attachment_sites: Tuple[int, ...]
    attachment_site_string: str
    num_attachment_points: int
    fragment_smiles_with_dummy: str
    fragment_canonical_key: str
    external_atom_count: int
    external_heavy_atom_count: int
    external_ring_count: int
    atoms: List[Dict[str, Any]]
    bonds: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    source_PL_peak: Any = None
    source_PLQY: Any = None
    source_FWHM: Any = None


@dataclass
class PatternSpec:
    pattern_id: str
    source_index: int
    source_molecule: str
    source_smiles: str
    source_canonical_smiles: str
    scaffold_label: str
    site_groups: List[Tuple[int, ...]]
    site_group_string: str
    n_modifications: int
    n_pendant: int
    n_fused: int
    source_PL_peak: Any = None
    source_PLQY: Any = None
    source_FWHM: Any = None


# -----------------------------
# Core matching
# -----------------------------

def load_cores(path: str, core_label_col: str, core_smiles_col: str) -> List[CoreInfo]:
    df = pd.read_excel(path)
    if core_label_col not in df.columns or core_smiles_col not in df.columns:
        raise ValueError(f"Core file columns not found. Available columns: {list(df.columns)}")

    cores: List[CoreInfo] = []
    for _, row in df.iterrows():
        label = str(row[core_label_col]).strip()
        smi = str(row[core_smiles_col]).strip()
        mol = safe_mol_from_smiles(smi)
        if mol is None:
            print(f"[WARN] Core {label} cannot be parsed: {smi}")
            continue
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        cores.append(CoreInfo(label, smi, can, mol, mol.GetNumAtoms(), n_hetero))

    # larger/more specific cores first
    cores.sort(key=lambda x: (x.n_atoms, x.n_hetero), reverse=True)
    return cores


def choose_best_core_match(mol: Chem.Mol, cores: List[CoreInfo]) -> Tuple[Optional[CoreInfo], Optional[Tuple[int, ...]], Dict[str, Any]]:
    candidates = []
    for core in cores:
        try:
            matches = mol.GetSubstructMatches(core.mol, uniquify=True, maxMatches=100)
        except Exception:
            matches = []
        if matches:
            # score: larger scaffold and fewer external atoms generally better
            for match in matches:
                candidates.append((core.n_atoms, core.n_hetero, core.label, core, match, len(matches)))
    if not candidates:
        return None, None, {"num_core_candidates": 0, "num_matches_for_selected_core": 0}

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, _, _, core, match, nmatch = candidates[0]
    return core, tuple(match), {
        "num_core_candidates": len(candidates),
        "num_matches_for_selected_core": nmatch,
    }


def connected_components_external(mol: Chem.Mol, core_atom_set: Set[int]) -> List[List[int]]:
    external = {a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in core_atom_set}
    seen = set()
    comps = []
    for start in sorted(external):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            atom = mol.GetAtomWithIdx(x)
            for nb in atom.GetNeighbors():
                j = nb.GetIdx()
                if j in external and j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def make_fragment_dummy_smiles(mol: Chem.Mol, comp_atoms: List[int], attachments: List[Dict[str, Any]]) -> str:
    """Create a readable fragment SMILES with dummy atoms connected at attachment positions."""
    em = Chem.RWMol()
    old_to_new = {}
    comp_set = set(comp_atoms)
    for old in comp_atoms:
        a = mol.GetAtomWithIdx(old)
        na = spec_to_atom(atom_to_spec(a))
        old_to_new[old] = em.AddAtom(na)

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in comp_set and j in comp_set:
            em.AddBond(old_to_new[i], old_to_new[j], b.GetBondType())

    for att in attachments:
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(int(att["core_site"]) + 1)  # site shown as [*:site+1]
        d_idx = em.AddAtom(dummy)
        em.AddBond(d_idx, old_to_new[int(att["external_atom_original_idx"])], name_to_bond_type(att["bond_type"]))
    frag = em.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def extract_modifications_for_molecule(
    mol: Chem.Mol,
    row: pd.Series,
    row_index: int,
    core: CoreInfo,
    match: Tuple[int, ...],
    smiles_col: str,
    molname_col: Optional[str],
    prop_cols: Dict[str, Optional[str]],
) -> Tuple[List[ModificationSpec], Optional[PatternSpec], Dict[str, Any]]:
    source_smi = str(row[smiles_col])
    source_can = canonical_smiles(mol) or ""
    source_name = str(row[molname_col]) if molname_col and molname_col in row.index else str(row_index)

    core_atom_set = set(match)
    mol_atom_to_core_site = {mol_idx: core_site for core_site, mol_idx in enumerate(match)}
    comps = connected_components_external(mol, core_atom_set)

    mods: List[ModificationSpec] = []
    warnings = []

    for comp_i, comp_atoms in enumerate(comps):
        comp_set = set(comp_atoms)
        attachment_records = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (i in comp_set and j in core_atom_set) or (j in comp_set and i in core_atom_set):
                ext_idx = i if i in comp_set else j
                core_idx = j if i in comp_set else i
                attachment_records.append({
                    "external_atom_original_idx": int(ext_idx),
                    "core_atom_original_idx": int(core_idx),
                    "core_site": int(mol_atom_to_core_site[core_idx]),
                    "bond_type": bond_type_to_name(b.GetBondType()),
                })

        if not attachment_records:
            warnings.append("external_component_without_attachment")
            continue

        # Sort attachments by core site for stable identity.
        attachment_records.sort(key=lambda x: (x["core_site"], x["external_atom_original_idx"]))
        sites = tuple(sorted(int(x["core_site"]) for x in attachment_records))
        n_attach = len(attachment_records)
        mod_type = "pendant_substituent" if n_attach == 1 else "fused_or_bridged_extension"

        old_to_local = {old: k for k, old in enumerate(comp_atoms)}
        atoms = []
        for old in comp_atoms:
            spec = atom_to_spec(mol.GetAtomWithIdx(old))
            spec["local_idx"] = int(old_to_local[old])
            atoms.append(spec)

        bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in comp_set and j in comp_set:
                bonds.append({
                    "begin_local_idx": int(old_to_local[i]),
                    "end_local_idx": int(old_to_local[j]),
                    "bond_type": bond_type_to_name(b.GetBondType()),
                    "is_aromatic": bool(b.GetIsAromatic()),
                })

        attachments = []
        for att in attachment_records:
            att2 = dict(att)
            att2["external_atom_local_idx"] = int(old_to_local[int(att["external_atom_original_idx"])])
            attachments.append(att2)

        frag_dummy = make_fragment_dummy_smiles(mol, comp_atoms, attachment_records)
        # key ignores source scaffold but keeps attachment count and fragment topology
        frag_key = f"{n_attach}|{frag_dummy}"
        ext_mol = Chem.PathToSubmol(mol, []) if False else None
        ring_count = 0
        try:
            submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=comp_atoms, canonical=True))
            ring_count = rdMolDescriptors.CalcNumRings(submol) if submol is not None else 0
        except Exception:
            pass

        mod = ModificationSpec(
            mod_id=f"M{row_index}_{comp_i}",
            source_index=int(row_index),
            source_molecule=source_name,
            source_smiles=source_smi,
            source_canonical_smiles=source_can,
            scaffold_label=core.label,
            modification_type=mod_type,
            attachment_sites=sites,
            attachment_site_string=";".join(map(str, sites)),
            num_attachment_points=n_attach,
            fragment_smiles_with_dummy=frag_dummy,
            fragment_canonical_key=frag_key,
            external_atom_count=len(comp_atoms),
            external_heavy_atom_count=sum(1 for idx in comp_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1),
            external_ring_count=int(ring_count),
            atoms=atoms,
            bonds=bonds,
            attachments=attachments,
            source_PL_peak=row[prop_cols["PL_peak"]] if prop_cols.get("PL_peak") else None,
            source_PLQY=row[prop_cols["PLQY"]] if prop_cols.get("PLQY") else None,
            source_FWHM=row[prop_cols["FWHM"]] if prop_cols.get("FWHM") else None,
        )
        mods.append(mod)

    if mods:
        site_groups = [m.attachment_sites for m in mods]
        site_groups_sorted = sorted(site_groups, key=lambda x: (len(x), x))
        pat = PatternSpec(
            pattern_id=f"P{row_index}",
            source_index=int(row_index),
            source_molecule=source_name,
            source_smiles=source_smi,
            source_canonical_smiles=source_can,
            scaffold_label=core.label,
            site_groups=site_groups_sorted,
            site_group_string="|".join([";".join(map(str, g)) for g in site_groups_sorted]),
            n_modifications=len(mods),
            n_pendant=sum(1 for m in mods if m.modification_type == "pendant_substituent"),
            n_fused=sum(1 for m in mods if m.modification_type != "pendant_substituent"),
            source_PL_peak=row[prop_cols["PL_peak"]] if prop_cols.get("PL_peak") else None,
            source_PLQY=row[prop_cols["PLQY"]] if prop_cols.get("PLQY") else None,
            source_FWHM=row[prop_cols["FWHM"]] if prop_cols.get("FWHM") else None,
        )
    else:
        pat = None

    diag = {
        "source_index": row_index,
        "source_molecule": source_name,
        "source_smiles": source_smi,
        "source_canonical_smiles": source_can,
        "scaffold_label": core.label,
        "matched_core_atoms": len(match),
        "external_component_count": len(comps),
        "modification_count": len(mods),
        "pendant_count": sum(1 for m in mods if m.modification_type == "pendant_substituent"),
        "fused_or_bridged_count": sum(1 for m in mods if m.modification_type != "pendant_substituent"),
        "largest_external_component_atoms": max([len(c) for c in comps], default=0),
        "warnings": ";".join(sorted(set(warnings))),
    }
    return mods, pat, diag


# -----------------------------
# Recombination
# -----------------------------

def remap_modification_to_sites(mod: ModificationSpec, target_sites: Tuple[int, ...]) -> Dict[str, Any]:
    """Return copy-like dict with core_site remapped to target_sites by sorted attachment order."""
    if len(target_sites) != mod.num_attachment_points:
        raise ValueError("Cannot remap modification to different number of sites")
    source_sites_sorted = sorted(list(mod.attachment_sites))
    target_sites_sorted = sorted(list(target_sites))
    mapping = {src: dst for src, dst in zip(source_sites_sorted, target_sites_sorted)}
    attachments = []
    for att in mod.attachments:
        new_att = dict(att)
        new_att["core_site"] = int(mapping[int(att["core_site"])])
        attachments.append(new_att)
    return {
        "mod_id": mod.mod_id,
        "source_scaffold_label": mod.scaffold_label,
        "modification_type": mod.modification_type,
        "source_sites": list(mod.attachment_sites),
        "target_sites": list(target_sites),
        "fragment_smiles_with_dummy": mod.fragment_smiles_with_dummy,
        "atoms": mod.atoms,
        "bonds": mod.bonds,
        "attachments": attachments,
        "num_attachment_points": mod.num_attachment_points,
        "external_heavy_atom_count": mod.external_heavy_atom_count,
    }


def attach_modifications(core: CoreInfo, mods_to_apply: List[Dict[str, Any]]) -> Tuple[Optional[Chem.Mol], str]:
    """Build molecule by adding external fragments to a core mol."""
    try:
        base = Chem.Mol(core.mol)
        rw = Chem.RWMol(base)
        used_attachment_signatures = set()
        for m in mods_to_apply:
            # Add atoms
            local_to_new = {}
            for spec in m["atoms"]:
                local_idx = int(spec["local_idx"])
                local_to_new[local_idx] = rw.AddAtom(spec_to_atom(spec))

            # Add internal bonds
            existing_bonds = set()
            for b in m["bonds"]:
                i = local_to_new[int(b["begin_local_idx"])]
                j = local_to_new[int(b["end_local_idx"])]
                key = tuple(sorted((i, j)))
                if key in existing_bonds:
                    continue
                existing_bonds.add(key)
                rw.AddBond(i, j, name_to_bond_type(b["bond_type"]))

            # Add attachment bonds. Core site index is atom index in the core molecule.
            for att in m["attachments"]:
                core_site = int(att["core_site"])
                ext_new = local_to_new[int(att["external_atom_local_idx"])]
                sig = (core_site, ext_new)
                if sig in used_attachment_signatures:
                    continue
                used_attachment_signatures.add(sig)
                if rw.GetBondBetweenAtoms(core_site, ext_new) is None:
                    rw.AddBond(core_site, ext_new, name_to_bond_type(att["bond_type"]))

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mol, smi
    except Exception as e:
        return None, f"BUILD_FAILED: {type(e).__name__}: {str(e)[:160]}"


def calc_basic_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    try:
        fp = None
        return {
            "MW": round(float(Descriptors.MolWt(mol)), 4),
            "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
            "NumAtoms": int(mol.GetNumAtoms()),
            "NumRings": int(rdMolDescriptors.CalcNumRings(mol)),
            "NumAromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "NumHeteroAtoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
            "LogP": round(float(Descriptors.MolLogP(mol)), 4),
            "RotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        }
    except Exception:
        return {}


def tanimoto_to_training(mol: Chem.Mol, train_fps: List[Any], max_compare: int = 20000) -> Tuple[Optional[float], Optional[int]]:
    if not train_fps:
        return None, None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps = train_fps if len(train_fps) <= max_compare else random.sample(train_fps, max_compare)
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if not sims:
            return None, None
        mx = max(sims)
        return round(float(mx), 4), int(sims.index(mx))
    except Exception:
        return None, None


def build_indices(mods: List[ModificationSpec]) -> Dict[str, Any]:
    by_scaffold_exact = defaultdict(list)
    by_scaffold_arity = defaultdict(list)
    by_arity = defaultdict(list)

    # deduplicate identical fragment per exact site per scaffold
    seen_exact = set()
    for m in mods:
        exact_key = (m.scaffold_label, m.attachment_sites, m.fragment_canonical_key)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        by_scaffold_exact[(m.scaffold_label, m.attachment_sites)].append(m)
        by_scaffold_arity[(m.scaffold_label, m.num_attachment_points)].append(m)
        by_arity[m.num_attachment_points].append(m)

    return {
        "by_scaffold_exact": by_scaffold_exact,
        "by_scaffold_arity": by_scaffold_arity,
        "by_arity": by_arity,
    }


def choose_mod_for_site_group(
    scaffold_label: str,
    site_group: Tuple[int, ...],
    indices: Dict[str, Any],
    allow_same_scaffold_remap: bool = True,
    allow_global_remap: bool = True,
) -> Tuple[Optional[ModificationSpec], str]:
    exact_pool = indices["by_scaffold_exact"].get((scaffold_label, site_group), [])
    if exact_pool:
        return random.choice(exact_pool), "exact_site"

    if allow_same_scaffold_remap:
        pool = indices["by_scaffold_arity"].get((scaffold_label, len(site_group)), [])
        if pool:
            return random.choice(pool), "same_scaffold_same_arity_remap"

    if allow_global_remap:
        pool = indices["by_arity"].get(len(site_group), [])
        if pool:
            return random.choice(pool), "global_same_arity_remap"

    return None, "no_pool"


def generate_candidates(
    cores_by_label: Dict[str, CoreInfo],
    patterns: List[PatternSpec],
    mods: List[ModificationSpec],
    train_cans: Set[str],
    train_fps: List[Any],
    target_size: int,
    max_size: int,
    max_attempts: int,
    seed: int,
    allow_same_scaffold_remap: bool,
    allow_global_remap: bool,
    include_unmodified_core: bool = False,
    compute_similarity: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(seed)
    indices = build_indices(mods)

    # Keep only patterns whose scaffold core is available.
    patterns2 = [p for p in patterns if p.scaffold_label in cores_by_label]
    if not patterns2:
        raise ValueError("No usable patterns for generation.")

    results = []
    seen = set()
    fail_counter = Counter()
    route_counter = Counter()
    scaffold_counter = Counter()

    attempts = 0
    start = time.time()
    while attempts < max_attempts and len(results) < max_size:
        attempts += 1
        pat = random.choice(patterns2)
        core = cores_by_label[pat.scaffold_label]
        site_groups = list(pat.site_groups)

        # optionally sample subset of modifications to increase diversity and avoid overcrowded cores
        if site_groups and random.random() < 0.25:
            # keep at least one modification, but sometimes fewer than source pattern
            k = random.randint(1, len(site_groups))
            site_groups = sorted(random.sample(site_groups, k), key=lambda x: (len(x), x))
        elif not site_groups and not include_unmodified_core:
            continue

        # avoid overlapping core sites in one generated molecule
        used_sites = set()
        filtered_groups = []
        for sg in site_groups:
            sg_set = set(sg)
            if used_sites & sg_set:
                continue
            used_sites |= sg_set
            filtered_groups.append(sg)
        site_groups = filtered_groups

        mods_to_apply = []
        route_tags = []
        used_frag_keys = set()
        ok = True
        for sg in site_groups:
            chosen, route = choose_mod_for_site_group(
                pat.scaffold_label, tuple(sg), indices,
                allow_same_scaffold_remap=allow_same_scaffold_remap,
                allow_global_remap=allow_global_remap,
            )
            route_tags.append(route)
            if chosen is None:
                ok = False
                break
            # avoid using identical fragment twice in the same molecule when possible
            local_key = (chosen.num_attachment_points, chosen.fragment_canonical_key)
            if local_key in used_frag_keys and random.random() < 0.8:
                ok = False
                break
            used_frag_keys.add(local_key)
            mods_to_apply.append(remap_modification_to_sites(chosen, tuple(sg)))

        if not ok:
            fail_counter["no_fragment_pool_or_duplicate"] += 1
            continue

        mol, smi_or_err = attach_modifications(core, mods_to_apply)
        if mol is None:
            fail_counter[smi_or_err.split(':')[0]] += 1
            continue
        can = canonical_smiles(mol)
        if not can:
            fail_counter["canonical_failed"] += 1
            continue
        if can in seen:
            fail_counter["duplicate_generated"] += 1
            continue
        seen.add(can)

        is_novel = can not in train_cans
        desc = calc_basic_descriptors(mol)
        sim = None
        if compute_similarity:
            sim, _ = tanimoto_to_training(mol, train_fps, max_compare=20000)
        record = {
            "candidate_id": f"GEN_{len(results)+1:06d}",
            "canonical_smiles": can,
            "scaffold_label": pat.scaffold_label,
            "source_pattern_id": pat.pattern_id,
            "source_pattern_sites": pat.site_group_string,
            "applied_site_groups": "|".join([";".join(map(str, sg)) for sg in site_groups]),
            "num_modifications": len(mods_to_apply),
            "num_pendant": sum(1 for m in mods_to_apply if m["modification_type"] == "pendant_substituent"),
            "num_fused_or_bridged": sum(1 for m in mods_to_apply if m["modification_type"] != "pendant_substituent"),
            "fragment_route": "+".join(route_tags),
            "fragment_mod_ids": ";".join([m["mod_id"] for m in mods_to_apply]),
            "fragment_source_scaffolds": ";".join([m["source_scaffold_label"] for m in mods_to_apply]),
            "fragment_smiles_with_dummy": ";".join([m["fragment_smiles_with_dummy"] for m in mods_to_apply]),
            "is_novel_vs_training": bool(is_novel),
            "max_tanimoto_to_training_sample": sim,
        }
        record.update(desc)
        results.append(record)
        route_counter.update(route_tags)
        scaffold_counter[pat.scaffold_label] += 1

        # Stop early once target reached, unless max_size requested and generation is still productive.
        if len(results) >= target_size:
            # keep extending only if user asked max_size > target_size and attempts are still efficient
            if max_size <= target_size:
                break
            recent_efficiency = len(results) / max(1, attempts)
            if recent_efficiency < 0.02:
                break

    summary = {
        "attempts": attempts,
        "generated_unique_valid": len(results),
        "target_size": target_size,
        "max_size": max_size,
        "max_attempts": max_attempts,
        "elapsed_sec": round(time.time() - start, 2),
        "route_counter": dict(route_counter),
        "scaffold_counter": dict(scaffold_counter),
        "fail_counter_top20": dict(fail_counter.most_common(20)),
        "allow_same_scaffold_remap": allow_same_scaffold_remap,
        "allow_global_remap": allow_global_remap,
        "compute_similarity": compute_similarity,
    }
    return results, summary


# -----------------------------
# Main
# -----------------------------

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="MR-TADF scaffold-constrained fragment recombination generator")
    parser.add_argument("--dataset", default=str(project_root / "data" / "tadf_dataset.xlsx"), help="Source dataset xlsx/csv")
    parser.add_argument("--cores", default=str(project_root / "data" / "core-structure info.xlsx"), help="Core structure xlsx/csv")
    parser.add_argument("--core_regex", default="^(A|B)", help="Only use core labels matching this regex. Default keeps A/B cores and excludes C/Other fragments. Use .* to include all.")
    parser.add_argument("--output_dir", default=str(project_root / "results" / "generated_scaffold_database"), help="Output directory")
    parser.add_argument("--smiles_col", default="Smiles", help="SMILES column in dataset")
    parser.add_argument("--molname_col", default="Molecules", help="Molecule name/id column in dataset")
    parser.add_argument("--core_label_col", default="core-type", help="Core label column in core file")
    parser.add_argument("--core_smiles_col", default="smiles", help="Core SMILES column in core file")
    parser.add_argument("--target_size", type=int, default=10000, help="Desired minimum generated size")
    parser.add_argument("--max_size", type=int, default=50000, help="Maximum generated size")
    parser.add_argument("--max_attempts", type=int, default=500000, help="Maximum random generation attempts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_similarity", action="store_true", default=False, help="Compute max Morgan Tanimoto similarity to training set. Slower for 10k-50k generation.")
    parser.add_argument("--allow_same_scaffold_remap", action="store_true", default=True, help="Allow same-scaffold same-arity fragment remapping")
    parser.add_argument("--disallow_same_scaffold_remap", dest="allow_same_scaffold_remap", action="store_false")
    parser.add_argument("--allow_global_remap", action="store_true", default=False, help="Allow global same-arity fragment remapping across scaffolds if exact pool insufficient")
    parser.add_argument("--max_rows", type=int, default=None, help="Debug: only process first N rows")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read dataset
    if args.dataset.lower().endswith(".csv"):
        df = pd.read_csv(args.dataset)
    else:
        df = pd.read_excel(args.dataset)

    if args.smiles_col not in df.columns:
        guessed = find_column(df, ["Smiles", "SMILES", "smile", "canonical_smiles"])
        if guessed:
            args.smiles_col = guessed
        else:
            raise ValueError(f"SMILES column not found. Available columns: {list(df.columns)}")
    if args.molname_col not in df.columns:
        args.molname_col = find_column(df, ["Molecules", "Molecule", "Name", "ID", "id"])

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    prop_cols = {
        "PL_peak": find_column(df, ["PL-peak", "PL peak", "PL_peak", "Emission", "lambda_em"]),
        "PLQY": find_column(df, ["PLQY", "QY", "Quantum yield"]),
        "FWHM": find_column(df, ["FWHM", "fwhm"]),
    }

    cores = load_cores(args.cores, args.core_label_col, args.core_smiles_col)
    if args.core_regex:
        rgx = re.compile(args.core_regex)
        cores = [c for c in cores if rgx.search(str(c.label))]
    if not cores:
        raise ValueError("No valid cores loaded.")
    cores_by_label = {c.label: c for c in cores}

    print(f"[INFO] Loaded dataset rows: {len(df)}")
    print(f"[INFO] Loaded cores: {[c.label for c in cores]}")
    print(f"[INFO] Property columns detected: {prop_cols}")

    all_mods: List[ModificationSpec] = []
    all_patterns: List[PatternSpec] = []
    diagnostics = []
    train_cans: Set[str] = set()
    train_fps = []

    for idx, row in df.iterrows():
        smi = row[args.smiles_col]
        mol = safe_mol_from_smiles(smi)
        if mol is None:
            diagnostics.append({
                "source_index": int(idx),
                "source_molecule": str(row[args.molname_col]) if args.molname_col else str(idx),
                "source_smiles": str(smi),
                "source_canonical_smiles": "",
                "scaffold_label": "",
                "status": "invalid_source_smiles",
                "warnings": "source_smiles_parse_failed",
            })
            continue
        can = canonical_smiles(mol)
        if can:
            train_cans.add(can)
            try:
                train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            except Exception:
                pass

        core, match, match_info = choose_best_core_match(mol, cores)
        if core is None or match is None:
            diagnostics.append({
                "source_index": int(idx),
                "source_molecule": str(row[args.molname_col]) if args.molname_col else str(idx),
                "source_smiles": str(smi),
                "source_canonical_smiles": can or "",
                "scaffold_label": "",
                "status": "no_core_match",
                "num_core_candidates": match_info.get("num_core_candidates", 0),
                "warnings": "no_A_or_B_core_match",
            })
            continue

        mods, pattern, diag = extract_modifications_for_molecule(
            mol, row, int(idx), core, match, args.smiles_col, args.molname_col, prop_cols
        )
        diag.update(match_info)
        diag["status"] = "matched"
        if match_info.get("num_matches_for_selected_core", 0) > 1:
            diag["warnings"] = (diag.get("warnings", "") + ";multiple_scaffold_matches").strip(";")
        diagnostics.append(diag)
        all_mods.extend(mods)
        if pattern is not None:
            all_patterns.append(pattern)

    # Deduplicate modification library rows for output readability but keep full list for frequency.
    mod_rows = []
    for m in all_mods:
        d = asdict(m)
        d["attachment_sites"] = ";".join(map(str, m.attachment_sites))
        d["atoms_json"] = json.dumps(m.atoms, ensure_ascii=False)
        d["bonds_json"] = json.dumps(m.bonds, ensure_ascii=False)
        d["attachments_json"] = json.dumps(m.attachments, ensure_ascii=False)
        d.pop("atoms", None); d.pop("bonds", None); d.pop("attachments", None)
        mod_rows.append(d)

    pat_rows = []
    for p in all_patterns:
        d = asdict(p)
        d["site_groups"] = p.site_group_string
        pat_rows.append(d)

    diag_df = pd.DataFrame(diagnostics)
    mod_df = pd.DataFrame(mod_rows)
    pat_df = pd.DataFrame(pat_rows)

    # Add frequency columns to modification library.
    if not mod_df.empty:
        freq_key = mod_df["scaffold_label"].astype(str) + "|" + mod_df["attachment_site_string"].astype(str) + "|" + mod_df["fragment_canonical_key"].astype(str)
        mod_df["frequency_same_scaffold_site_fragment"] = freq_key.map(freq_key.value_counts())
        frag_freq = mod_df["fragment_canonical_key"].astype(str).value_counts()
        mod_df["frequency_global_fragment"] = mod_df["fragment_canonical_key"].astype(str).map(frag_freq)

    diag_path = os.path.join(args.output_dir, "scaffold_match_diagnostics.csv")
    mod_path = os.path.join(args.output_dir, "modification_library.csv")
    pat_path = os.path.join(args.output_dir, "learned_substitution_patterns.csv")
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
    mod_df.to_csv(mod_path, index=False, encoding="utf-8-sig")
    pat_df.to_csv(pat_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Matched molecules: {(diag_df.get('status') == 'matched').sum() if 'status' in diag_df else 0}")
    print(f"[INFO] Extracted modifications: {len(all_mods)}")
    print(f"[INFO] Learned patterns: {len(all_patterns)}")
    if not mod_df.empty:
        print("[INFO] Modification types:", {str(k): int(v) for k, v in mod_df["modification_type"].value_counts().items()})
        print("[INFO] Top scaffold labels:", dict(mod_df["scaffold_label"].value_counts().head(20)))

    if not all_mods or not all_patterns:
        print("[ERROR] No modifications/patterns extracted. Check core SMILES and source SMILES.")
        sys.exit(2)

    results, gen_summary = generate_candidates(
        cores_by_label=cores_by_label,
        patterns=all_patterns,
        mods=all_mods,
        train_cans=train_cans,
        train_fps=train_fps,
        target_size=args.target_size,
        max_size=args.max_size,
        max_attempts=args.max_attempts,
        seed=args.seed,
        allow_same_scaffold_remap=args.allow_same_scaffold_remap,
        allow_global_remap=args.allow_global_remap,
        compute_similarity=args.compute_similarity,
    )

    gen_df = pd.DataFrame(results)
    gen_path = os.path.join(args.output_dir, "generated_candidates_valid_unique.csv")
    gen_df.to_csv(gen_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_dataset": args.dataset,
        "input_cores": args.cores,
        "n_dataset_rows": len(df),
        "n_training_canonical_smiles": len(train_cans),
        "n_valid_cores": len(cores),
        "core_labels": [c.label for c in cores],
        "n_diagnostics_rows": len(diag_df),
        "n_matched_molecules": int((diag_df.get("status") == "matched").sum()) if "status" in diag_df else 0,
        "n_extracted_modifications": len(all_mods),
        "n_learned_patterns": len(all_patterns),
        "modification_type_counts": {str(k): int(v) for k, v in mod_df["modification_type"].value_counts().items()} if not mod_df.empty else {},
        "pattern_scaffold_counts": {str(k): int(v) for k, v in pat_df["scaffold_label"].value_counts().items()} if not pat_df.empty else {},
        "generation": gen_summary,
        "output_files": {
            "diagnostics": diag_path,
            "modification_library": mod_path,
            "learned_patterns": pat_path,
            "generated_candidates": gen_path,
        },
    }
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print("\n[DONE] Output files:")
    print(" -", diag_path)
    print(" -", mod_path)
    print(" -", pat_path)
    print(" -", gen_path)
    print(" -", summary_path)
    print("\n[SUMMARY]")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:4000])

    if len(results) < args.target_size:
        print("\n[WARN] Generated candidates are fewer than target_size.")
        print("Suggestions:")
        print("1) increase --max_attempts, e.g. --max_attempts 2000000")
        print("2) enable --allow_global_remap")
        print("3) inspect scaffold_match_diagnostics.csv for no_core_match or multiple_scaffold_matches")
        print("4) check whether core SMILES are too large/specific and miss many source molecules")


if __name__ == "__main__":
    main()
