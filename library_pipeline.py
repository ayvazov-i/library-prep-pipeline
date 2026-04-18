import argparse
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover, FilterCatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

try:
    from dimorphite_dl.protonate import protonate_smiles as _dimorphite_protonate
    def dimorphite_protonate(smi, min_ph=6.4, max_ph=8.4):
        return _dimorphite_protonate(smi, min_ph=min_ph, max_ph=max_ph)
except ImportError:
    try:
        import dimorphite_dl
        def dimorphite_protonate(smi, min_ph=6.4, max_ph=8.4):
            return dimorphite_dl.run_with_mol_list(
                [Chem.MolFromSmiles(smi)], min_ph=min_ph, max_ph=max_ph
            )
    except ImportError:
        dimorphite_protonate = None


def load_supplier_file(filepath, supplier_name=None):
    """Load a supplier SMILES file into a DataFrame with standardised columns."""
    if supplier_name is None:
        supplier_name = Path(filepath).stem

    records = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            smiles = parts[0]
            if smiles.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue
            mol_id = parts[1] if len(parts) > 1 else f"{supplier_name}_{line_num}"
            records.append({
                "ID": mol_id,
                "SMILES": smiles,
                "original_supplier_smiles": smiles,
                "supplier": supplier_name,
            })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} molecules from {supplier_name}")
    return df


def merge_suppliers(supplier_files):
    """Load and merge multiple supplier files into one DataFrame."""
    print("\n" + "=" * 60)
    print("STEP 1: LOAD AND MERGE SUPPLIER CATALOGUES")
    print("=" * 60)

    frames = [load_supplier_file(f, Path(f).stem) for f in supplier_files]
    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total molecules after merge: {merged.shape[0]:,}")
    return merged


def strip_salts(df):
    """Strip salts from SMILES, keeping the largest fragment."""
    print("\n" + "=" * 60)
    print("STEP 2: STRIP SALTS")
    print("=" * 60)

    remover = SaltRemover.SaltRemover()
    stripped = []
    failed = 0

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            failed += 1
            continue

        clean = remover.StripMol(mol)
        clean_smi = Chem.MolToSmiles(clean)

        if "." in clean_smi:
            frags = clean_smi.split(".")
            largest = max(
                frags,
                key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                if Chem.MolFromSmiles(s) else 0,
            )
            clean_smi = largest

        new_row = row.copy()
        new_row["SMILES"] = clean_smi
        stripped.append(new_row)

    result = pd.DataFrame(stripped)
    print(f"  Parse failures removed: {failed:,}")
    print(f"  Molecules after salt stripping: {len(result):,}")
    return result


def build_filters():
    """Build PAINS and BRENK filter catalogues."""
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat = FilterCatalog.FilterCatalog(pains_params)

    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_cat = FilterCatalog.FilterCatalog(brenk_params)

    return pains_cat, brenk_cat


def apply_filters(df):
    """
    Apply compound filters: complexity, BRENK, Lipinski, rings, aggregator, PAINS.
    Returns (passed_df, failed_df).
    """
    print("\n" + "=" * 60)
    print("STEP 3: COMPOUND FILTERING")
    print("=" * 60)

    pains_cat, brenk_cat = build_filters()
    passed = []
    failed = []

    for _, row in df.iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append({**row, "fail_reason": "parse_failed"})
            continue

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 15:
            failed.append({**row, "fail_reason": f"too_small:heavy_atoms={n_heavy}"})
            continue
        if n_heavy > 70:
            failed.append({**row, "fail_reason": f"too_large:heavy_atoms={n_heavy}"})
            continue

        if brenk_cat.HasMatch(mol):
            match = brenk_cat.GetFirstMatch(mol)
            failed.append({**row, "fail_reason": f"brenk:{match.GetDescription()}"})
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        lip_violations = []
        if mw > 500: lip_violations.append(f"MW={mw:.0f}")
        if logp > 5: lip_violations.append(f"logP={logp:.1f}")
        if hbd > 5: lip_violations.append(f"HBD={hbd}")
        if hba > 10: lip_violations.append(f"HBA={hba}")
        if lip_violations:
            failed.append({**row, "fail_reason": f"lipinski:{';'.join(lip_violations)}"})
            continue

        ring_info = mol.GetRingInfo()
        n_rings = ring_info.NumRings()
        if n_rings > 6:
            failed.append({**row, "fail_reason": f"too_many_rings:{n_rings}"})
            continue
        if n_rings > 0:
            largest_ring = max(len(r) for r in ring_info.AtomRings())
            if largest_ring > 8:
                failed.append({**row, "fail_reason": f"large_ring:size={largest_ring}"})
                continue

        if logp > 4.0 and mw > 400:
            fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            if fsp3 < 0.1 and logp > 4.5:
                failed.append({**row, "fail_reason": f"aggregator:logP={logp:.1f};MW={mw:.0f};Fsp3={fsp3:.2f}"})
                continue

        if pains_cat.HasMatch(mol):
            match = pains_cat.GetFirstMatch(mol)
            failed.append({**row, "fail_reason": f"pains:{match.GetDescription()}"})
            continue

        passed.append(row)

    pass_df = pd.DataFrame(passed)
    fail_df = pd.DataFrame(failed)

    print(f"  Passed: {len(pass_df):,}")
    print(f"  Failed: {len(fail_df):,}")
    if len(fail_df) > 0 and "fail_reason" in fail_df.columns:
        reasons = fail_df["fail_reason"].apply(lambda x: x.split(":")[0])
        for reason, count in reasons.value_counts().items():
            print(f"    {reason:20s} {count:>8,}")

    return pass_df, fail_df


def count_unspecified_stereocentres(mol):
    """Count unspecified tetrahedral stereocentres."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo_info = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return sum(1 for _, label in stereo_info if label == "?")


def filter_and_enumerate_stereo(df, max_unspecified=2):
    """Remove molecules with >max_unspecified stereocentres, enumerate the rest."""
    print("\n" + "=" * 60)
    print("STEP 4: STEREO FILTERING + ENUMERATION")
    print("=" * 60)

    filtered_out = 0
    enumerated_records = []
    opts = StereoEnumerationOptions(
        tryEmbedding=False, onlyUnassigned=True, maxIsomers=16
    )

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue

        n_unspec = count_unspecified_stereocentres(mol)

        if n_unspec > max_unspecified:
            filtered_out += 1
            continue

        if n_unspec == 0:
            enumerated_records.append(row.to_dict())
        else:
            isomers = list(EnumerateStereoisomers(mol, options=opts))
            for iso_idx, iso_mol in enumerate(isomers):
                iso_smi = Chem.MolToSmiles(iso_mol, isomericSmiles=True)
                new_record = row.to_dict()
                new_record["SMILES"] = iso_smi
                new_record["ID"] = f"{row['ID']}_iso{iso_idx + 1}"
                enumerated_records.append(new_record)

    result = pd.DataFrame(enumerated_records)
    print(f"  Removed (>{max_unspecified} unspecified centres): {filtered_out:,}")
    print(f"  Molecules after enumeration: {len(result):,}")
    return result


def enumerate_tautomers(df, max_tautomers=5):
    """Enumerate tautomers for each molecule."""
    print("\n" + "=" * 60)
    print("STEP 4b: TAUTOMER ENUMERATION")
    print("=" * 60)

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers * 5)
    enumerator.SetMaxTransforms(1000)

    tautomer_records = []
    expanded = 0
    failed = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            tautomer_records.append(row.to_dict())
            continue

        try:
            tauts = list(enumerator.Enumerate(mol))
            if len(tauts) <= 1:
                tautomer_records.append(row.to_dict())
            else:
                expanded += 1
                for t_idx, t_mol in enumerate(tauts[:max_tautomers]):
                    t_smi = Chem.MolToSmiles(t_mol, isomericSmiles=True)
                    if Chem.MolFromSmiles(t_smi) is None:
                        continue
                    new_record = row.to_dict()
                    new_record["SMILES"] = t_smi
                    if t_idx > 0:
                        new_record["ID"] = f"{row['ID']}_tau{t_idx + 1}"
                    tautomer_records.append(new_record)
        except Exception:
            tautomer_records.append(row.to_dict())
            failed += 1

        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,}...")

    result = pd.DataFrame(tautomer_records)
    print(f"  Molecules with tautomers: {expanded:,}")
    print(f"  Enumeration failures (kept original): {failed:,}")
    print(f"  Molecules after tautomer enumeration: {len(result):,}")
    return result


def deduplicate(df):
    """Deduplicate by canonical SMILES, merging IDs and original supplier SMILES."""
    print("\n" + "=" * 60)
    print("STEP 5: DEDUPLICATE")
    print("=" * 60)

    before = len(df)
    df["canonical"] = df["SMILES"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        if Chem.MolFromSmiles(s) is not None else None
    )
    df = df.dropna(subset=["canonical"])

    grouped = df.groupby("canonical", as_index=False).agg({
        "ID": lambda x: ";".join(sorted(set(x))),
        "original_supplier_smiles": lambda x: ";".join(sorted(set(x))),
        "supplier": lambda x: ";".join(sorted(set(x))),
    })
    grouped = grouped.rename(columns={"canonical": "SMILES"})

    print(f"  Before: {before:,}")
    print(f"  After:  {len(grouped):,}")
    print(f"  Duplicates removed: {before - len(grouped):,}")
    return grouped


def ionise_molecules(df, min_ph=6.4, max_ph=8.4):
    """Enumerate protonation states using Dimorphite-DL."""
    print("\n" + "=" * 60)
    print(f"STEP 6: IONISE (Dimorphite-DL, pH {min_ph:.1f}\u2013{max_ph:.1f})")
    print("=" * 60)

    if dimorphite_protonate is None:
        print("  WARNING: dimorphite_dl not installed. Skipping.")
        return df

    ionised_records = []
    failed = 0

    for idx, row in df.iterrows():
        try:
            variants = dimorphite_protonate(row["SMILES"], min_ph=min_ph, max_ph=max_ph)
            if not variants:
                ionised_records.append(row.to_dict())
                continue

            for v_idx, variant in enumerate(variants):
                v_smi = variant.strip() if isinstance(variant, str) else Chem.MolToSmiles(variant)
                if not v_smi or Chem.MolFromSmiles(v_smi) is None:
                    continue
                new_record = row.to_dict()
                new_record["SMILES"] = v_smi
                if len(variants) > 1:
                    new_record["ID"] = f"{row['ID']}_pH{v_idx + 1}"
                ionised_records.append(new_record)
        except Exception:
            ionised_records.append(row.to_dict())
            failed += 1

        if (idx + 1) % 10000 == 0:
            print(f"  Ionised {idx + 1:,} / {len(df):,}...")

    result = pd.DataFrame(ionised_records)
    print(f"  Dimorphite failures (kept original): {failed:,}")
    print(f"  Molecules after ionisation: {len(result):,}")
    return result


def _run_rdkonf_worker(args):
    """Worker: run rdkonf on a single SMILES file."""
    worker_id, smi_path, rdkonf_path = args
    sdf_path = smi_path + ".sdf"
    try:
        result = subprocess.run(
            [sys.executable, rdkonf_path, smi_path],
            capture_output=True, text=True, timeout=3600 * 6,
        )
        if result.returncode != 0:
            return (worker_id, smi_path, sdf_path, False, result.stderr[:300])
    except subprocess.TimeoutExpired:
        return (worker_id, smi_path, sdf_path, False, "timeout")
    except Exception as e:
        return (worker_id, smi_path, sdf_path, False, str(e))
    return (worker_id, smi_path, sdf_path, True, None)


def generate_conformers_rdkonf(df, output_sdf, rdkonf_path="rdkonf/rdkonf.py",
                                n_conformers=1, n_workers=32):
    """Generate 3D conformers using rdkonf, parallelised across n_workers."""
    print("\n" + "=" * 60)
    print("STEP 7: CONFORMER GENERATION (rdkonf)")
    print("=" * 60)

    if not os.path.exists(rdkonf_path):
        print(f"  ERROR: rdkonf not found at {rdkonf_path}")
        return None

    total = len(df)
    actual_workers = min(n_workers, total)
    rows_per_worker = (total + actual_workers - 1) // actual_workers
    print(f"  Total molecules: {total:,}")
    print(f"  Parallel workers: {actual_workers}")

    worker_smi_paths = []
    for w in range(actual_workers):
        start = w * rows_per_worker
        end = min(start + rows_per_worker, total)
        if start >= total:
            break
        chunk = df.iloc[start:end]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_w{w}.smi", delete=False, dir=tempfile.gettempdir()
        )
        for _, row in chunk.iterrows():
            tmp.write(f"{row['SMILES']} {row['ID']}\n")
        tmp.close()
        worker_smi_paths.append((w, tmp.name, rdkonf_path))

    print(f"  Split into {len(worker_smi_paths)} files (~{rows_per_worker:,} each)")
    t_start = time.time()

    with Pool(processes=len(worker_smi_paths)) as pool:
        results = pool.map(_run_rdkonf_worker, worker_smi_paths)

    print(f"\n  All workers finished in {time.time() - t_start:.0f}s")

    failed_workers = [r for r in results if not r[3]]
    if failed_workers:
        print(f"  WARNING: {len(failed_workers)} worker(s) had errors:")
        for wid, _, _, _, err in failed_workers:
            print(f"    Worker {wid}: {err[:200]}")

    mol_count = 0
    with open(output_sdf, "w") as out:
        for wid, smi_path, sdf_path, success, _ in results:
            if success and os.path.exists(sdf_path):
                with open(sdf_path, "r") as f:
                    for line in f:
                        out.write(line)
                        if line.strip() == "$$$$":
                            mol_count += 1
                os.unlink(sdf_path)
            if os.path.exists(smi_path):
                os.unlink(smi_path)

    total_time = time.time() - t_start
    rate = mol_count / total_time if total_time > 0 else 0
    print(f"\n  Conformers written: {mol_count:,}")
    print(f"  Time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Throughput: {rate:.1f} mol/s")
    return output_sdf
def generate_conformers_nvmolkit(df, output_sdf, n_conformers=1,
                                  mmff_max_iters=200, batch_size=500,
                                  batches_per_gpu=4, gpu_ids=None,
                                  preprocessing_threads=8):
    """Generate 3D conformers using nvMolKit on GPU.

    Requires nvMolKit and an NVIDIA GPU with compute capability >= 7.0.
    Falls back with a clear error if nvMolKit is not installed.
    """
    print("\n" + "=" * 60)
    print("STEP 7: CONFORMER GENERATION (nvMolKit / GPU)")
    print("=" * 60)

    try:
        from rdkit.Chem import SDWriter, AllChem
        from rdkit.Chem.rdDistGeom import ETKDGv3
        from nvmolkit.embedMolecules import EmbedMolecules
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        from nvmolkit.types import HardwareOptions
    except ImportError as e:
        print(f"  ERROR: nvMolKit not available ({e})")
        print(f"  Install with: conda install -c conda-forge nvmolkit")
        print(f"  Or use --conformer-backend rdkonf for the CPU path.")
        return None

    t0 = time.time()

    # 1. SMILES -> Mol with explicit Hs
    mols = []
    parse_fail = 0
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(row["SMILES"])
        if m is None:
            parse_fail += 1
            continue
        m = Chem.AddHs(m)
        m.SetProp("_Name", str(row["ID"]))
        mols.append(m)
    print(f"  Prepared {len(mols):,} molecules ({parse_fail} parse failures)")

    # 2. ETKDG params — useRandomCoords=True is required by nvMolKit
    params = ETKDGv3()
    params.useRandomCoords = True

    # 3. Hardware config
    hw = HardwareOptions(
        preprocessingThreads=preprocessing_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=gpu_ids if gpu_ids else [],
    )

    # 4. GPU embed (modifies mols in-place)
    t_embed = time.time()
    print(f"  Embedding {n_conformers} confs/mol on GPU...")
    EmbedMolecules(mols, params, confsPerMolecule=n_conformers,
                   hardwareOptions=hw)
    print(f"    embed wall-clock: {time.time() - t_embed:.1f}s")

    # 5. Partition by MMFF-parametrisability
    mmff_ok, mmff_bad = [], []
    for m in mols:
        if AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s") is None:
            mmff_bad.append(m)
        else:
            mmff_ok.append(m)
    if mmff_bad:
        print(f"  MMFF parametrisable: {len(mmff_ok):,} / {len(mols):,} "
              f"({len(mmff_bad)} skipped)")

    # 6. GPU MMFF minimise the parametrisable ones
    t_mmff = time.time()
    print(f"  MMFF minimising (maxIters={mmff_max_iters}) on GPU...")
    energies = MMFFOptimizeMoleculesConfs(
        mmff_ok, maxIters=mmff_max_iters, hardwareOptions=hw
    )
    print(f"    mmff wall-clock:  {time.time() - t_mmff:.1f}s")

    # 7. Write all conformers, tagging whether minimised
    writer = SDWriter(output_sdf)
    n_written = 0
    for m, mol_energies in zip(mmff_ok, energies):
        for cid in range(m.GetNumConformers()):
            if cid < len(mol_energies):
                m.SetProp("MMFF_Energy", f"{mol_energies[cid]:.3f}")
            m.SetProp("MMFF_Minimised", "True")
            writer.write(m, confId=cid)
            n_written += 1
    for m in mmff_bad:
        for cid in range(m.GetNumConformers()):
            m.SetProp("MMFF_Minimised", "False")
            writer.write(m, confId=cid)
            n_written += 1
    writer.close()

    dt = time.time() - t0
    print(f"\n  Conformers written: {n_written:,}")
    print(f"  Total time: {dt:.1f}s ({dt/3600:.2f}h)")
    print(f"  Throughput: {n_written/dt:.1f} confs/s")
    return output_sdf    


def main():
    parser = argparse.ArgumentParser(
        description="Chemical library preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input SMILES files")
    parser.add_argument("--output", default="library_3d.sdf",
                        help="Output SDF file (default: library_3d.sdf)")
    parser.add_argument("--rdkonf", default="rdkonf/rdkonf.py",
                        help="Path to rdkonf.py")
    parser.add_argument("--n-conformers", type=int, default=1,
                        help="Conformers per molecule (default: 1)")
    parser.add_argument("--max-unspecified-stereo", type=int, default=2,
                        help="Max unspecified stereocentres (default: 2)")
    parser.add_argument("--max-tautomers", type=int, default=5,
                        help="Max tautomers per molecule (default: 5)")
    parser.add_argument("--min-ph", type=float, default=6.4,
                        help="Min pH for ionisation (default: 6.4)")
    parser.add_argument("--max-ph", type=float, default=8.4,
                        help="Max pH for ionisation (default: 8.4)")
    parser.add_argument("--n-workers", type=int, default=32,
                        help="Parallel workers (default: 32)")
    parser.add_argument("--skip-tautomers", action="store_true",
                        help="Skip tautomer enumeration")
    parser.add_argument("--skip-ionise", action="store_true",
                        help="Skip ionisation")
    parser.add_argument("--skip-conformers", action="store_true",
                        help="Skip conformer generation")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save intermediate CSV files")
    parser.add_argument("--conformer-backend", choices=["rdkonf", "nvmolkit"],
                    default="rdkonf",
                    help="Conformer generation backend: rdkonf (CPU, default) "
                         "or nvmolkit (GPU, requires NVIDIA GPU)")                    
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 60)
    print("CHEMICAL LIBRARY PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Input:   {', '.join(args.input)}")
    print(f"Output:  {args.output}")
    print(f"Workers: {args.n_workers}")

    df = merge_suppliers(args.input)
    if args.save_intermediates:
        df.to_csv("intermediate_01_merged.csv", index=False)

    df = strip_salts(df)
    if args.save_intermediates:
        df.to_csv("intermediate_02_salts_stripped.csv", index=False)

    df, failed_df = apply_filters(df)
    if args.save_intermediates:
        df.to_csv("intermediate_03_filtered.csv", index=False)
        failed_df.to_csv("intermediate_03_failed.csv", index=False)

    df = filter_and_enumerate_stereo(df, max_unspecified=args.max_unspecified_stereo)
    if args.save_intermediates:
        df.to_csv("intermediate_04_stereo_enumerated.csv", index=False)

    if not args.skip_tautomers:
        df = enumerate_tautomers(df, max_tautomers=args.max_tautomers)
        if args.save_intermediates:
            df.to_csv("intermediate_04b_tautomers.csv", index=False)

    df = deduplicate(df)
    if args.save_intermediates:
        df.to_csv("intermediate_05_deduplicated.csv", index=False)

    if not args.skip_ionise:
        df = ionise_molecules(df, min_ph=args.min_ph, max_ph=args.max_ph)
        before = len(df)
        df["canonical"] = df["SMILES"].apply(
            lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
            if Chem.MolFromSmiles(s) is not None else None
        )
        df = df.dropna(subset=["canonical"])
        df = df.drop_duplicates(subset=["canonical"])
        df["SMILES"] = df["canonical"]
        df = df.drop(columns=["canonical"])
        print(f"  Re-dedup: {before:,} -> {len(df):,}")
        if args.save_intermediates:
            df.to_csv("intermediate_06_ionised.csv", index=False)

    final_smi = args.output.replace(".sdf", "_final.smi")
    df.to_csv(final_smi, sep="\t", index=False)
    print(f"\n  Final SMILES: {final_smi}")
    print(f"  Final count:  {len(df):,}")

    if not args.skip_conformers:
        if args.conformer_backend == "nvmolkit":
            generate_conformers_nvmolkit(
                df, output_sdf=args.output,
                n_conformers=args.n_conformers,
            )
        else:
            generate_conformers_rdkonf(
                df, output_sdf=args.output, rdkonf_path=args.rdkonf,
                n_conformers=args.n_conformers, n_workers=args.n_workers,
            )

    total_time = time.time() - t_total
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Runtime: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()