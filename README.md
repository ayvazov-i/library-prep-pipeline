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
