# Chemical Library Preparation Pipeline

Prepares raw supplier catalogues for virtual screening

## Steps

1. **Load and merge** supplier catalogues — reads SMILES files from one or more suppliers into a single table, keeping track of original supplier SMILES and IDs throughout.
2. **Strip salts** — removes counter-ions (Na+, Cl−, etc.) and keeps the largest fragment.
3. **Filter** — applies BRENK (reactive/unstable groups), Lipinski Rule of Five, ring count/size limits, an aggregator heuristic, and PAINS. Each rejection is logged with the reason.
4. **Stereoisomer handling** — removes molecules with more than 2 unspecified stereocentres, then enumerates the remaining unspecified ones (up to 4 isomers per molecule). Specified centres are preserved.
5. **Tautomer enumeration** — generates up to 5 tautomeric forms per molecule using RDKit.
6. **Deduplication** — canonicalises SMILES and merges duplicates (from different salt forms, enumerated isomers, or overlapping suppliers), keeping all original IDs.
7. **Ionisation** — enumerates protonation states at pH 6.4–8.4 using Dimorphite-DL, then deduplicates again.
8. **3D conformer generation** — generates low-energy conformers using rdkonf (Ebejer et al. method), parallelised across 32 cores.

## Requirements

```bash
conda create -n chem python=3.11
conda activate chem
conda install -c conda-forge rdkit
pip install pandas dimorphite-dl
git clone https://github.com/stevenshave/rdkonf.git
```

## Benchmarks

Conformer generation throughput, 10 conformers per molecule, Enamine REAL compounds:

| Scale     | Backend   | Hardware    | Output confs | Time     | Throughput         |
|-----------|-----------|-------------|--------------|----------|--------------------|
| 10k       | RDKit     | 8 CPU cores | 100k         | 17m      | 95 confs/s         |
| 10k       | nvMolKit  | 1 GPU       | 100k         | 32s      | 3,100 confs/s      |
| 100k      | nvMolKit  | 1 GPU       | ~2.0M        | 15m      | 2,737 confs/s      |
| **1M**    | nvMolKit  | 1 GPU       | **24.7M**    | **3h 39m** | **~1,990 confs/s** (steady-state) |

The 1M run used the chunked runner (200k mols/chunk); throughput is reported
as steady-state across chunks 6–13 to exclude shared-server contention
observed during chunks 1–5. End-to-end average including contention: 1,884 confs/s.

Speedup vs. RDKit CPU baseline: **~29×** at production scale.

Hardware: NVIDIA RTX 5090 (32 GB), Intel Xeon server with 8 CPU workers for
preprocessing.

## Usage

```bash
python library_pipeline.py \
    --input supplier1.smi supplier2.smi \
    --output library_3d.sdf \
    --rdkonf rdkonf/rdkonf.py \
    --n-workers 32
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | One or more supplier SMILES files |
| `--output` | library_3d.sdf | Output SDF with 3D coordinates |
| `--rdkonf` | rdkonf/rdkonf.py | Path to rdkonf script |
| `--n-workers` | 32 | Parallel workers for conformer generation |
| `--max-unspecified-stereo` | 2 | Max unspecified stereocentres before rejection |
| `--max-tautomers` | 5 | Max tautomers per molecule |
| `--min-ph` / `--max-ph` | 6.4 / 8.4 | pH range for ionisation |
| `--skip-tautomers` | off | Skip tautomer enumeration |
| `--skip-ionise` | off | Skip Dimorphite-DL ionisation |
| `--skip-conformers` | off | Skip 3D generation (output SMILES only) |
| `--save-intermediates` | off | Save CSV at each step for debugging |

## Large-scale runs (chunked GPU)

The built-in `generate_conformers_nvmolkit` loads all molecules into RAM
before calling `EmbedMolecules`. On servers with <128 GB RAM this OOMs
around ~1M input molecules (post stereo/tautomer expansion: ~2M+ mols).

For large runs, use the standalone chunked runner:

```bash
# First, run the CPU stages of the main pipeline up to ionisation,
# writing the final SMILES to disk:
python library_pipeline.py \
    --input enamine_real_chunk.smi \
    --output library.sdf \
    --skip-conformers \
    --save-intermediates

# Then run conformer generation in bounded-memory chunks:
python run_conformers_chunked.py \
    --input library_final.smi \
    --output library.sdf \
    --chunk-size 200000 \
    --n-conformers 10
```

The runner streams conformers directly to the output SDF, freeing each
chunk's molecules before loading the next. Memory stays bounded at
`chunk-size` mols regardless of total input size. Validated at 2.6M mols
(≈1M input after expansion) producing 24.7M conformers.

Tune `--chunk-size` down to 100k if system RAM is constrained; throughput
is essentially unchanged.

## Input formats

Supplier files can be either plain SMILES or cxsmiles:

- **Tab-delimited cxsmiles** (Enamine REAL format):
  `SMILES [|ext|]\tID\t...` — the CXSmiles extension `|...|` lives inside
  field 0 and is stripped before parsing.

- **Space-delimited SMILES**: `SMILES ID` — extensions, if present as
  free-standing tokens, are ignored when identifying the compound ID.

Header rows (`SMILES`, `CANONICAL_SMILES`, `SMI`, `SMILE`) are skipped
automatically.

## Acknowledgements

- [RDKit](https://www.rdkit.org/) — cheminformatics toolkit
- [Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl) — protonation state enumeration
- [rdkonf](https://github.com/stevenshave/rdkonf) — conformer generation (Ebejer et al. 2012)
## GPU backend (nvMolKit)

The pipeline supports GPU-accelerated conformer generation via
[nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit), which
implements batch ETKDG + MMFF on CUDA. On one NVIDIA L40S we measured
~2,800 conformers/second sustained on 50k Enamine REAL molecules,
compared to ~33 confs/s on 8 CPU cores with the default rdkonf backend.

### Requirements
- NVIDIA GPU with compute capability >= 7.0 (V100 or newer)
- CUDA driver >= 560.28
- RDKit 2025.03.1 or 2024.09.6

### Install
```bash
conda install -c conda-forge nvmolkit
```

### Usage
```bash
# CPU backend (default)
python library_pipeline.py --input mols.smi --output lib.sdf

# GPU backend
python library_pipeline.py --input mols.smi --output lib.sdf \
    --conformer-backend nvmolkit --n-conformers 10
```

### Notes
- `useRandomCoords=True` is enforced by nvMolKit; this matches RDKit's
  recommended setting for flexible molecules.
- A small fraction of Enamine REAL molecules (~0.04%) contain hypervalent
  atoms MMFF94s cannot parametrise. These are written to the output SDF
  without MMFF minimisation, tagged `MMFF_Minimised: False`.
