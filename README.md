# Chemical Library Preparation Pipeline

Prepares raw supplier catalogues for virtual screening.

## Steps
1. Load and merge supplier catalogues
2. Strip salts
3. Filter (BRENK, Lipinski, PAINS, complexity, rings, aggregator)
4. Enumerate stereoisomers and tautomers
5. Deduplicate
6. Ionise at biological pH (Dimorphite-DL)
7. Generate 3D conformers (rdkonf)

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
    --n-workers 32 \
    --save-intermediates
```# library-prep-pipeline
Chemical library preparation pipeline for virtual screening

