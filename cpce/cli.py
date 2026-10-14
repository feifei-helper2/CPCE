"""Command-line entry point for one CPCE run."""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np

from .model import CPCE


def _parser():
    parser = argparse.ArgumentParser(description="CPCE clustering of h5ad expression or a NumPy embedding.")
    parser.add_argument("input", type=Path, help=".h5ad or a dense cell-by-feature .npy array")
    parser.add_argument("-k", "--n-clusters", type=int, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--representation", help="existing h5ad obsm key, e.g. X_pca; bypass preprocessing")
    parser.add_argument("--input-state", choices=("counts", "log1p"), help="required when preprocessing h5ad")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--counts-layer")
    source.add_argument("--use-raw", action="store_true")
    parser.add_argument("--n-top-genes", type=int)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--capacity-buffer", type=float, default=0.15)
    parser.add_argument("--capacity-floor", type=float, default=1.0)
    parser.add_argument("--density-noise", type=float, default=0.01)
    parser.add_argument("--consensus-resolution", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--label-key", help="optional annotation column for evaluation only")
    parser.add_argument("--save-h5ad", action="store_true", help="also save expression, PCA and labels")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    adata = None
    suffix = args.input.suffix.lower()
    preprocessing = None
    if suffix == ".h5ad":
        import anndata as ad
        adata = ad.read_h5ad(args.input)
        if args.label_key and (args.label_key not in adata.obs or adata.obs[args.label_key].isna().any()):
            parser.error("label-key must name a complete annotation column.")
        if args.representation:
            if args.input_state or args.counts_layer or args.use_raw or args.n_top_genes is not None:
                parser.error("An existing representation cannot be combined with preprocessing options.")
            if args.representation not in adata.obsm:
                parser.error(f"Representation not found: {args.representation}")
            X = adata.obsm[args.representation]
        else:
            if args.input_state is None:
                parser.error("Choose --input-state counts or log1p, or select --representation X_pca.")
            from .preprocessing import prepare_adata
            adata = prepare_adata(adata, input_state=args.input_state, counts_layer=args.counts_layer,
                                  use_raw=args.use_raw, n_top_genes=args.n_top_genes,
                                  n_pcs=args.n_pcs, random_state=args.seed)
            X = adata.obsm["X_pca"]
            preprocessing = adata.uns["cpce_preprocessing"]
        cell_ids = adata.obs_names
    elif suffix == ".npy":
        if args.representation or args.input_state or args.counts_layer or args.use_raw or args.label_key or args.save_h5ad or args.n_top_genes is not None:
            parser.error("h5ad options cannot be used with .npy input.")
        X = np.load(args.input, allow_pickle=False)
        cell_ids = np.arange(len(X)) if X.ndim else []
    else:
        parser.error("Input must be .h5ad or .npy.")

    parameters = dict(n_clusters=args.n_clusters, n_rounds=args.n_rounds, alpha_max=args.alpha_max,
                      n_neighbors=args.n_neighbors, capacity_buffer=args.capacity_buffer,
                      capacity_floor=args.capacity_floor, density_noise=args.density_noise,
                      consensus_resolution=args.consensus_resolution, random_state=args.seed,
                      block_size=args.block_size)
    model = CPCE(**parameters)
    start = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - start
    summary = {"parameters": parameters, "n_cells": len(labels), "n_features": X.shape[1],
               "n_clusters": len(np.unique(labels)), "elapsed_seconds": elapsed,
               "consensus_resolution_used": model.resolution_, "preprocessing": preprocessing}
    if args.label_key:
        from .metrics import evaluate
        summary["metrics"] = evaluate(adata.obs[args.label_key].to_numpy(), labels)

    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "labels.npy", labels)
    import csv
    with (args.output / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell_id", "cpce_cluster"])
        writer.writerows(zip(cell_ids, labels))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.save_h5ad:
        import pandas as pd
        adata.obs["cpce_cluster"] = pd.Categorical(labels.astype(str))
        adata.uns["cpce"] = parameters
        adata.write_h5ad(args.output / "clustered.h5ad", compression="gzip")
    logging.info("Saved %d clusters to %s (%.2f s)", summary["n_clusters"], args.output, elapsed)


if __name__ == "__main__":
    main()
