import numpy as np
import scanpy as sc
from pandas import DataFrame, Series
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


def _get_specificity(genes: str, clust_mask: np.array, ad: sc.AnnData):
    expr_mask = (ad.raw[:, genes].X > 0).A
    fpr = np.sum(expr_mask & np.atleast_2d(~clust_mask).T, axis=0) / np.sum(~clust_mask)
    return 1 - fpr


def _get_auc(genes: str, clust_mask: np.array, ad: sc.AnnData):
    return np.array([roc_auc_score(clust_mask, ad.raw[:, g].X.A[:, 0]) for g in genes])


def get_markers_per_cluster(
    adata: sc.AnnData,
    cluster_key: str = "leiden",
    min_specificity: float = 0.75,
    min_markers: int = 15,
    max_markers: int = 200,
):
    """Get marker genes per cluster

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object
    cluster_key : str
        Key of the cluster column in adata.obs
    min_specificity : float
        Minimum specificity
    min_markers : int
        Minimum number of markers
    max_markers : int
        Maximum number of markers

    Returns
    -------
    Series
        Series of DataFrames containing marker genes per cluster

    """
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")

    marker_dfs = {}
    for cli in tqdm(adata.obs[cluster_key].unique()):
        genes = adata.uns["rank_genes_groups"]["names"][cli][:max_markers]
        clust_mask = adata.obs[cluster_key] == cli
        specificity = _get_specificity(genes, clust_mask, adata)
        mask = specificity >= min(min_specificity, sorted(specificity)[-min_markers])
        genes = genes[mask]
        specificity = specificity[mask]
        auc = _get_auc(genes, clust_mask, adata)
        marker_dfs[cli] = DataFrame({"gene": genes, "specificity": specificity, "auc": auc})

    return Series(marker_dfs)


def get_top_markers(marker_dfs: Series, min_auc: float = 0.7, max_markers: int = 7):
    """Get top markers

    Parameters
    ----------
    marker_dfs : Series
        Series of DataFrames containing marker genes per cluster
    min_auc : float
        Minimum AUC
    max_markers : int
        Maximum number of markers

    Returns
    -------
    Series
        Series of top marker genes per cluster

    """
    marker_genes = marker_dfs.map(
        lambda x: x[x.auc > min_auc].sort_values("auc", ascending=False).head(max_markers).gene.values
    )
    return marker_genes[np.argsort(marker_genes.index.astype(int))]
