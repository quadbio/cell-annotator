import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numpy.random import binomial, negative_binomial

# Declare the dictionary of expected marker genes
expected_marker_genes = {
    "Neuron": [
        "MAP2",
        "NEFL",
        "RBFOX3",
        "SYN1",
        "GAP43",
        "DCX",
        "TUBB3",
        "NEUROD1",
        "STMN2",
        "ENO2",
        "SLC17A7",
        "GAD1",
        "GAD2",
    ],
    "Fibroblast": [
        "COL1A1",
        "COL3A1",
        "VIM",
        "ACTA2",
        "FAP",
        "PDGFRA",
        "THY1",
        "FN1",
        "SPARC",
        "S100A4",
        "MMP2",
        "MMP9",
        "CDH11",
    ],
}

# Declare the neuronal and fibroblast cell types
neuronal_cell_types = ["Neuron", "Neurons", "Neuronal cells", "neurons"]
fibroblast_cell_types = ["Fibroblast", "Fibroblasts", "fibroblast cells"]


def get_example_data(n_cells: int = 100, n_samples: int = 1) -> AnnData:
    """Create example data for testing. Adapted from scanpy.

    The data consists of two clusters with different marker genes. The first cluster is enriched for neuronal markers and the second cluster is enriched for fibroblast markers."""
    gene_names = expected_marker_genes["Neuron"] + expected_marker_genes["Fibroblast"]
    n_genes = len(gene_names)
    adata = AnnData(np.multiply(binomial(1, 0.15, (n_cells, n_genes)), negative_binomial(2, 0.25, (n_cells, n_genes))))
    adata.var_names = gene_names

    # Create marker genes for the two clusters
    n_group_1 = np.floor(0.3 * n_cells).astype(int)
    n_group_2 = n_cells - n_group_1
    n_marker_genes = int(n_genes / 2)

    adata.X[:n_group_1, :n_marker_genes] = np.multiply(
        binomial(1, 0.9, (n_group_1, n_marker_genes)), negative_binomial(1, 0.5, (n_group_1, n_marker_genes))
    )
    adata.X[n_group_1:, n_marker_genes:] = np.multiply(
        binomial(1, 0.9, (n_group_2, n_marker_genes)), negative_binomial(1, 0.5, (n_group_2, n_marker_genes))
    )

    # Create cluster according to groups
    adata.obs["leiden"] = pd.Categorical(np.concatenate((n_group_1 * ["0"], n_group_2 * ["1"])))

    # Add sample information if there are multiple samples
    if n_samples > 1:
        samples = np.random.choice([f"sample_{i}" for i in range(n_samples)], size=n_cells)
        adata.obs["sample"] = samples

    # filter, normalize and log transform the data
    sc.pp.filter_cells(adata, min_counts=2)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata
