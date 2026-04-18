"""Memory-bounded GPU conformer generation for large libraries.

Reads a tab-delimited SMILES file produced by library_pipeline.py
(--skip-conformers --save-intermediates), processes molecules in fixed-size
chunks, and streams conformers to the output SDF. Each chunk's RDKit mol
objects are freed before the next chunk is loaded, keeping peak RAM at
~chunk-size molecules regardless of total input size.

Validated at 2.6M molecules (≈1M input after stereo/tautomer/ionisation
expansion) producing 24.7M conformers in 3h 39m on 1× NVIDIA RTX 5090.
"""

import argparse
import gc
import re
import sys
import time
from pathlib import Path

_CXSMILES_EXT_RE = re.compile(r'\s*\|[^|]*\|\s*$')


def _load_chunk(rows):
    """Convert a list of (smiles, mol_id) pairs into AddHs'd RDKit mols."""
    from rdkit import Chem

    mols = []
    parse_fail = 0
    for smiles, mol_id in rows:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            parse_fail += 1
            continue
        m = Chem.AddHs(m)
        m.SetProp("_Name", str(mol_id))
        mols.append(m)
    return mols, parse_fail


def _embed_and_write(mols, writer, n_conformers, mmff_max_iters,
                     batch_size, batches_per_gpu, preprocessing_threads):
    """Embed conformers for mols and write them to writer. Returns conf count."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdDistGeom import ETKDGv3
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions

    params = ETKDGv3()
    params.useRandomCoords = True

    hw = HardwareOptions(
        preprocessingThreads=preprocessing_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=[],
    )

    EmbedMolecules(mols, params, confsPerMolecule=n_conformers,
                   hardwareOptions=hw)

    mmff_ok, mmff_bad = [], []
    for m in mols:
        if AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s") is None:
            mmff_bad.append(m)
        else:
            mmff_ok.append(m)

    energies = MMFFOptimizeMoleculesConfs(
        mmff_ok, maxIters=mmff_max_iters, hardwareOptions=hw
    )

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

    return n_written


def _iter_smiles_file(path):
    """Yield (smiles, mol_id) from a tab- or space-delimited SMILES file."""
    with open(path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                fields = line.split("\t")
                smiles_raw = fields[0].strip()
                mol_id = fields[1].strip() if len(fields) > 1 else f"mol_{line_num}"
            else:
                fields = line.split()
                smiles_raw = fields[0]
                non_ext = [f for f in fields[1:] if not f.startswith("|")]
                mol_id = non_ext[0] if non_ext else f"mol_{line_num}"

            bare = _CXSMILES_EXT_RE.sub("", smiles_raw).strip()
            if bare.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue

            smiles = _CXSMILES_EXT_RE.sub("", smiles_raw).strip()
            yield smiles, mol_id


def run_chunked(input_path, output_path, chunk_size, n_conformers,
                mmff_max_iters, batch_size, batches_per_gpu,
                preprocessing_threads):
    try:
        from rdkit.Chem import SDWriter
        import nvmolkit  # noqa: F401 — fail early with a clear message
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install nvMolKit: conda install -c conda-forge nvmolkit")
        sys.exit(1)

    t_total = time.time()
    total_mols = 0
    total_confs = 0
    total_parse_fail = 0
    chunk_idx = 0

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Conformers: {n_conformers}/mol")
    print()

    writer = SDWriter(str(output_path))

    chunk: list = []
    for smiles, mol_id in _iter_smiles_file(input_path):
        chunk.append((smiles, mol_id))
        if len(chunk) < chunk_size:
            continue

        chunk_idx += 1
        t_chunk = time.time()
        mols, parse_fail = _load_chunk(chunk)
        chunk = []

        print(f"Chunk {chunk_idx}: {len(mols):,} mols "
              f"({parse_fail} parse failures)")
        n_written = _embed_and_write(
            mols, writer, n_conformers, mmff_max_iters,
            batch_size, batches_per_gpu, preprocessing_threads,
        )

        total_mols += len(mols)
        total_confs += n_written
        total_parse_fail += parse_fail
        dt = time.time() - t_chunk
        print(f"  {n_written:,} confs in {dt:.0f}s "
              f"({n_written/dt:.0f} confs/s)\n")

        del mols
        gc.collect()

    # final partial chunk
    if chunk:
        chunk_idx += 1
        t_chunk = time.time()
        mols, parse_fail = _load_chunk(chunk)
        print(f"Chunk {chunk_idx} (final): {len(mols):,} mols "
              f"({parse_fail} parse failures)")
        n_written = _embed_and_write(
            mols, writer, n_conformers, mmff_max_iters,
            batch_size, batches_per_gpu, preprocessing_threads,
        )
        total_mols += len(mols)
        total_confs += n_written
        total_parse_fail += parse_fail
        dt = time.time() - t_chunk
        print(f"  {n_written:,} confs in {dt:.0f}s "
              f"({n_written/dt:.0f} confs/s)\n")
        del mols
        gc.collect()

    writer.close()

    dt_total = time.time() - t_total
    print("=" * 60)
    print(f"Done.  {total_mols:,} mols -> {total_confs:,} conformers")
    print(f"Parse failures: {total_parse_fail:,}")
    print(f"Total time: {dt_total:.0f}s ({dt_total/3600:.2f}h)")
    print(f"Throughput: {total_confs/dt_total:.0f} confs/s")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-bounded chunked GPU conformer generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Tab-delimited SMILES file (from library_pipeline.py)")
    parser.add_argument("--output", default="library.sdf",
                        help="Output SDF file (default: library.sdf)")
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Molecules per chunk (default: 200000)")
    parser.add_argument("--n-conformers", type=int, default=10,
                        help="Conformers per molecule (default: 10)")
    parser.add_argument("--mmff-max-iters", type=int, default=200,
                        help="MMFF max iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="nvMolKit GPU batch size (default: 500)")
    parser.add_argument("--batches-per-gpu", type=int, default=4,
                        help="nvMolKit batches per GPU (default: 4)")
    parser.add_argument("--preprocessing-threads", type=int, default=8,
                        help="CPU preprocessing threads (default: 8)")
    args = parser.parse_args()

    run_chunked(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chunk_size=args.chunk_size,
        n_conformers=args.n_conformers,
        mmff_max_iters=args.mmff_max_iters,
        batch_size=args.batch_size,
        batches_per_gpu=args.batches_per_gpu,
        preprocessing_threads=args.preprocessing_threads,
    )


if __name__ == "__main__":
    main()
