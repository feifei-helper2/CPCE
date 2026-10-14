# CPCE

CPCE is a capacity-aware clustering ensemble for rare cell discovery in scRNA-seq data.

## Install

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Run

For raw counts in a `.h5ad` file:

```bash
python -m cpce data/input.h5ad -k 18 -o results/run --input-state counts
```

Use `--input-state log1p` for expression already normalized to 10,000 counts per cell and log1p transformed. For an existing PCA representation:

```bash
python -m cpce data/input.h5ad -k 18 -o results/run --representation X_pca
```

Set `-k` to the target cluster count and `-o` to a new output directory.
