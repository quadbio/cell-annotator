import numpy as np
import scanpy as sc
from pandas import DataFrame, Series
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from cell_annotator._logging import logger


def _get_specificity(genes: list[str], clust_mask: np.ndarray, adata: sc.AnnData, use_raw: bool = True):
    if use_raw:
        values = adata.raw[:, genes].X
    else:
        values = adata[:, genes].X

    if issparse(values):
        values = values.toarray()
    expr_mask = values > 0

    fpr = np.sum(expr_mask & np.atleast_2d(~clust_mask).T, axis=0) / np.sum(~clust_mask)
    return 1 - fpr


def _get_auc(genes: str, clust_mask: np.ndarray, adata: sc.AnnData):
    return np.array([roc_auc_score(clust_mask, adata.raw[:, g].X.A[:, 0]) for g in genes])


def get_markers_per_cluster(
    adata: sc.AnnData,
    cluster_key: str = "leiden",
    min_specificity: float = 0.75,
    min_markers: int = 15,
    max_markers: int = 200,
    use_raw: bool = True,
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
    use_raw : bool
        Use raw data

    Returns
    -------
    Series
        Series of DataFrames containing marker genes per cluster

    """
    logger.info("Computing marker genes per cluster")
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon", use_raw=use_raw)

    marker_dfs = {}
    logger.info("Iterating over clusters to compute specificity and AUC values.")
    for cli in tqdm(adata.obs[cluster_key].unique()):
        # get a list of differentially expressed genes
        genes = adata.uns["rank_genes_groups"]["names"][cli][:max_markers]

        # compute their specificity
        clust_mask = adata.obs[cluster_key] == cli
        logger.debug("Computing specificity for cluster %s", cli)
        specificity = _get_specificity(genes, clust_mask, adata)

        # filter genes by specificity
        mask = specificity >= min(min_specificity, sorted(specificity)[-min_markers])
        genes = genes[mask]
        specificity = specificity[mask]

        # compute AUCs
        logger.debug("Computing AUC for cluster %s", cli)
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
