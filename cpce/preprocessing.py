"""Expression normalization, HVG selection, and PCA for h5ad inputs."""

import numpy as np
from scipy import sparse


def prepare_adata(adata, *, input_state, counts_layer=None, use_raw=False,
                  n_top_genes=None, n_pcs=50, random_state=42):
    """Prepare a copy without using cell-type annotations.

    ``input_state`` must be ``counts`` or ``log1p``. A counts layer and
    ``adata.raw`` are used only when explicitly selected. Integer counts
    use Seurat v3 HVGs (default 4,000); log-normalized data use Seurat
    HVGs (default 4,500). PCA is centered, without gene-wise scaling.
    """
    import scanpy as sc

    if input_state not in {"counts", "log1p"}:
        raise ValueError("input_state must be 'counts' or 'log1p'.")
    if counts_layer is not None and (use_raw or input_state != "counts"):
        raise ValueError("counts_layer requires counts input and cannot be combined with use_raw.")
    if n_pcs < 1 or (n_top_genes is not None and n_top_genes < 2):
        raise ValueError("n_pcs must be positive and n_top_genes must be at least 2.")
    if use_raw:
        if adata.raw is None:
            raise ValueError("Input has no adata.raw matrix.")
        result = adata.raw.to_adata()
        result.obs = adata.obs.copy()
    else:
        result = adata.copy()
    if counts_layer is not None:
        if counts_layer not in result.layers:
            raise ValueError(f"Counts layer not found: {counts_layer}")
        result.X = result.layers[counts_layer].copy()

    values = result.X.data if sparse.issparse(result.X) else np.asarray(result.X)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expression must be finite, nonnegative, and nonempty.")
    if input_state == "counts" and not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError("Counts input contains non-integer values; choose log1p for log-normalized expression.")
    if result.n_obs < 2:
        raise ValueError("PCA preprocessing requires at least two cells.")
    if np.any(np.asarray(result.X.sum(axis=1)).ravel() == 0):
        raise ValueError("Remove zero-expression cells before preprocessing.")

    sc.pp.filter_genes(result, min_cells=1)
    result.var_names_make_unique()
    result.X = result.X.astype(np.float64)
    n_hvg = min(n_top_genes or (4000 if input_state == "counts" else 4500), result.n_vars)
    if result.n_vars < 2:
        raise ValueError("At least two expressed genes are needed for PCA.")
    if input_state == "counts":
        if n_hvg < result.n_vars:
            sc.pp.highly_variable_genes(result, n_top_genes=n_hvg, flavor="seurat_v3")
        else:
            result.var["highly_variable"] = True
        sc.pp.normalize_total(result, target_sum=1e4)
        sc.pp.log1p(result)
    else:
        # Seurat's dispersion calculation needs the logarithm base metadata.
        result.uns["log1p"] = {"base": None}
        if n_hvg < result.n_vars:
            sc.pp.highly_variable_genes(result, n_top_genes=n_hvg, flavor="seurat")
        else:
            result.var["highly_variable"] = True
    selected = int(result.var["highly_variable"].sum())
    components = min(n_pcs, result.n_obs - 1, selected - 1)
    if components < 1:
        raise ValueError("HVG selection retained too few genes for PCA.")
    sc.pp.pca(result, n_comps=components, mask_var="highly_variable",
              zero_center=True, svd_solver="arpack", random_state=random_state)
    result.uns["cpce_preprocessing"] = {
        "input_state": input_state, "n_hvg": selected, "n_pcs": components,
        "counts_layer": counts_layer or "", "use_raw": use_raw,
        "target_sum": 1e4 if input_state == "counts" else "already_normalized",
    }
    return result
