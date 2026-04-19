"""
Benchmark SMARTS substructure matching: RDKit CPU vs nvMolKit GPU.

Loads PAINS SMARTS directly from RDKit's wehi_pains.csv (480 patterns).
BRENK patterns can be added via --brenk-smarts pointing to a SMARTS file
(one SMARTS per line, optional tab-separated name after it).

Usage:
    python bench_smarts.py --input scale_test_100k_final.smi
    python bench_smarts.py --input molecules.smi --scales 1000 10000
    python bench_smarts.py --input molecules.smi --skip-gpu       # CPU only
    python bench_smarts.py --input molecules.smi --verify-only    # just correctness

Input SMILES file: tab/space-separated, SMILES in the first column.
Matches the _final.smi format produced by library_pipeline.py.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

RDLogger.DisableLog("rdApp.*")


# ---------- SMARTS loading ----------

def find_pains_csv():
    """Locate PAINS SMARTS CSV. Search order:
       1. PAINS_CSV env var (explicit override)
       2. pains_smarts.csv next to this script (bundled fallback)
       3. RDKit's bundled wehi_pains.csv (may be absent in some conda builds)
    """
    env = os.environ.get("PAINS_CSV")
    if env and os.path.exists(env):
        return env

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(script_dir, "pains_smarts.csv")
    if os.path.exists(bundled):
        return bundled

    rdkit_path = os.path.join(os.path.dirname(rdkit.__file__), "Data", "Pains", "wehi_pains.csv")
    if os.path.exists(rdkit_path):
        return rdkit_path

    raise FileNotFoundError(
        "PAINS SMARTS CSV not found. Tried:\n"
        f"  $PAINS_CSV env var\n"
        f"  {bundled}\n"
        f"  {rdkit_path}\n"
        "Download pains_smarts.csv alongside this script, or set PAINS_CSV."
    )


def load_pains_smarts():
    """Load all 480 PAINS SMARTS from a CSV (WEHI format: SMARTS, <regId=name>)."""
    path = find_pains_csv()
    smarts, names = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            s = row[0]
            name = row[1].strip("<>").replace("regId=", "")
            smarts.append(s)
            names.append(name)
    return smarts, names


def load_smarts_file(path):
    """Load user-supplied SMARTS (e.g. BRENK). One SMARTS per line, optional name after tab/space."""
    smarts, names = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smarts.append(parts[0])
            names.append(parts[1] if len(parts) > 1 else f"{os.path.basename(path)}_{i}")
    return smarts, names


def smarts_to_queries(smarts_list, names):
    """Convert SMARTS strings to RDKit query Mols. Drop invalid."""
    queries, kept_names = [], []
    for s, n in zip(smarts_list, names):
        m = Chem.MolFromSmarts(s)
        if m is not None:
            queries.append(m)
            kept_names.append(n)
    return queries, kept_names


# ---------- target mol loading ----------

def load_smiles(path, limit=None):
    """Load SMILES from a tab/space-separated file. First column is SMILES."""
    smiles_list = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smi = parts[0]
            if i == 0 and smi.upper() in ("SMILES", "CANONICAL_SMILES", "SMI"):
                continue
            smiles_list.append(smi)
            if limit and len(smiles_list) >= limit:
                break
    return smiles_list


def parse_targets(smiles_list):
    mols, failures = [], 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            failures += 1
            continue
        mols.append(m)
    return mols, failures


# ---------- RDKit CPU baselines ----------

def bench_rdkit_cpu(targets, queries):
    """Per-(target, query) HasSubstructMatch loop — apples-to-apples with nvMolKit."""
    n_t, n_q = len(targets), len(queries)
    result = np.zeros((n_t, n_q), dtype=np.uint8)
    t0 = time.time()
    for i, mol in enumerate(targets):
        for j, q in enumerate(queries):
            if mol.HasSubstructMatch(q):
                result[i, j] = 1
    dt = time.time() - t0
    return result, dt


def bench_rdkit_catalog(targets):
    """RDKit FilterCatalog.HasMatch on PAINS+BRENK — what library_pipeline.py uses.
    Short-circuits at first match, so not per-pattern comparable."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    cat = FilterCatalog.FilterCatalog(params)
    t0 = time.time()
    hits = sum(1 for m in targets if cat.HasMatch(m))
    dt = time.time() - t0
    return hits, dt


# ---------- nvMolKit GPU ----------

def bench_nvmolkit_gpu(targets, queries):
    try:
        from nvmolkit.substructure import hasSubstructMatch
    except ImportError as e:
        raise RuntimeError(f"nvMolKit not available: {e}")

    # Warm up: first call incurs CUDA context setup
    _ = hasSubstructMatch(targets[: min(8, len(targets))], queries[: min(4, len(queries))])

    t0 = time.time()
    result = hasSubstructMatch(targets, queries)
    dt = time.time() - t0
    return np.asarray(result), dt


# ---------- correctness ----------

def verify_correctness(targets, queries, n_sample=200):
    if len(queries) == 0:
        print("  [skip] No queries loaded.")
        return None
    sample = targets[:n_sample]
    cpu_res, _ = bench_rdkit_cpu(sample, queries)
    try:
        gpu_res, _ = bench_nvmolkit_gpu(sample, queries)
    except RuntimeError as e:
        print(f"  [skip] nvMolKit unavailable: {e}")
        return None

    if cpu_res.shape != gpu_res.shape:
        print(f"  SHAPE MISMATCH: CPU {cpu_res.shape} vs GPU {gpu_res.shape}")
        return False

    disagree = (cpu_res != gpu_res)
    n_dis = int(disagree.sum())
    total = cpu_res.size
    print(f"  Sample:        {len(sample)} mols x {len(queries)} patterns = {total:,} checks")
    print(f"  Disagreements: {n_dis} ({100 * n_dis / total:.4f}%)")
    if n_dis > 0:
        tgt_idx, q_idx = np.where(disagree)
        print("  First disagreements (target_idx, query_idx, cpu, gpu):")
        for k in range(min(5, len(tgt_idx))):
            ti, qi = int(tgt_idx[k]), int(q_idx[k])
            print(f"    ({ti}, {qi}): CPU={cpu_res[ti, qi]}, GPU={gpu_res[ti, qi]}")
    return n_dis == 0


# ---------- scale sweep ----------

def run_scale(targets_all, queries, scale, skip_gpu):
    targets = targets_all[:scale]
    n_q = len(queries)
    total_checks = len(targets) * n_q
    print(f"\n--- SCALE: {scale:,} mols x {n_q} patterns = {total_checks:,} checks ---")

    print("  [CPU] RDKit per-pattern HasSubstructMatch...")
    cpu_res, cpu_dt = bench_rdkit_cpu(targets, queries)
    cpu_rate = total_checks / cpu_dt
    cpu_flagged = int(cpu_res.any(axis=1).sum())
    print(f"    time:         {cpu_dt:.2f}s")
    print(f"    throughput:   {cpu_rate:,.0f} checks/s")
    print(f"    mols flagged: {cpu_flagged:,} / {len(targets):,} ({100*cpu_flagged/len(targets):.1f}%)")

    print("  [CPU] RDKit FilterCatalog.HasMatch (PAINS+BRENK, short-circuit - current pipeline)...")
    cat_hits, cat_dt = bench_rdkit_catalog(targets)
    print(f"    time:         {cat_dt:.2f}s   throughput: {len(targets)/cat_dt:,.0f} mols/s   flagged: {cat_hits:,}")

    if skip_gpu:
        return {"scale": scale, "cpu_dt": cpu_dt, "cpu_rate": cpu_rate, "cat_dt": cat_dt}

    print("  [GPU] nvMolKit hasSubstructMatch...")
    try:
        gpu_res, gpu_dt = bench_nvmolkit_gpu(targets, queries)
        gpu_rate = total_checks / gpu_dt
        gpu_flagged = int(gpu_res.any(axis=1).sum())
        speedup = cpu_dt / gpu_dt
        agree = np.array_equal(cpu_res, gpu_res)
        print(f"    time:         {gpu_dt:.2f}s")
        print(f"    throughput:   {gpu_rate:,.0f} checks/s")
        print(f"    mols flagged: {gpu_flagged:,} / {len(targets):,}")
        print(f"    SPEEDUP over CPU per-pattern loop: {speedup:.1f}x")
        print(f"    full-matrix agreement with CPU:    {'YES' if agree else 'NO'}")
        return {
            "scale": scale, "cpu_dt": cpu_dt, "cpu_rate": cpu_rate, "cat_dt": cat_dt,
            "gpu_dt": gpu_dt, "gpu_rate": gpu_rate, "speedup": speedup, "agree": agree,
        }
    except RuntimeError as e:
        print(f"    [skip] {e}")
        return {"scale": scale, "cpu_dt": cpu_dt, "cpu_rate": cpu_rate, "cat_dt": cat_dt}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="SMILES file (first column SMILES)")
    ap.add_argument("--brenk-smarts", help="Optional: file with one BRENK SMARTS per line")
    ap.add_argument("--scales", type=int, nargs="+", default=[1000, 10000, 100000],
                    help="Molecule counts to test (default: 1000 10000 100000)")
    ap.add_argument("--skip-gpu", action="store_true", help="CPU only, no nvMolKit")
    ap.add_argument("--verify-only", action="store_true",
                    help="Just run correctness check, no benchmark")
    ap.add_argument("--verify-n", type=int, default=200, help="Correctness sample size (default 200)")
    args = ap.parse_args()

    print("=" * 64)
    print("SMARTS MATCHING BENCHMARK - RDKit CPU vs nvMolKit GPU")
    print("=" * 64)
    print(f"Input: {args.input}")

    # Patterns
    print("\nLoading SMARTS patterns...")
    smarts, names = load_pains_smarts()
    print(f"  PAINS (wehi_pains.csv): {len(smarts)} SMARTS")
    if args.brenk_smarts:
        b_smarts, b_names = load_smarts_file(args.brenk_smarts)
        print(f"  BRENK ({args.brenk_smarts}): {len(b_smarts)} SMARTS")
        smarts.extend(b_smarts)
        names.extend(b_names)

    queries, kept_names = smarts_to_queries(smarts, names)
    print(f"  Total valid query Mols: {len(queries)} / {len(smarts)}")
    if not queries:
        print("No valid queries. Abort.")
        sys.exit(1)

    # Targets
    max_scale = max(args.scales) if args.scales else args.verify_n
    print(f"\nLoading up to {max_scale:,} molecules from {args.input}...")
    smiles = load_smiles(args.input, limit=max_scale)
    print(f"  Read {len(smiles):,} SMILES lines")

    print(f"\nParsing {len(smiles):,} SMILES...")
    t0 = time.time()
    targets_all, parse_fail = parse_targets(smiles)
    print(f"  Parsed: {len(targets_all):,}  ({parse_fail} failures, {time.time()-t0:.1f}s)")
    if not targets_all:
        print("No molecules parsed. Abort.")
        sys.exit(1)

    # Correctness
    if not args.skip_gpu:
        print(f"\nCorrectness check ({args.verify_n} mols x {len(queries)} patterns)...")
        ok = verify_correctness(targets_all, queries, n_sample=args.verify_n)
        if ok is False:
            print("  WARNING: CPU and GPU disagree. Investigate before trusting speedups below.")
        elif ok is True:
            print("  OK - CPU and GPU agree on sample.")

    if args.verify_only:
        return

    # Sweep
    results = []
    for scale in args.scales:
        if scale > len(targets_all):
            print(f"\nSkipping scale {scale:,} - only {len(targets_all):,} mols available")
            continue
        results.append(run_scale(targets_all, queries, scale, args.skip_gpu))

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    header = f"{'scale':>10}  {'cpu time':>10}  {'cpu rate':>16}  {'gpu time':>10}  {'gpu rate':>16}  {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        scale = r["scale"]
        cpu_str = f"{r['cpu_dt']:.2f}s"
        cpu_rate_str = f"{r['cpu_rate']:,.0f} chk/s"
        gpu_str = f"{r.get('gpu_dt'):.2f}s" if r.get("gpu_dt") is not None else "-"
        gpu_rate_str = f"{r.get('gpu_rate'):,.0f} chk/s" if r.get("gpu_rate") is not None else "-"
        speedup_str = f"{r.get('speedup'):.1f}x" if r.get("speedup") is not None else "-"
        print(f"{scale:>10,}  {cpu_str:>10}  {cpu_rate_str:>16}  {gpu_str:>10}  {gpu_rate_str:>16}  {speedup_str:>8}")
    print()


if __name__ == "__main__":
    main()