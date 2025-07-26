"""Package utility functions."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score

from cell_annotator._constants import PackageConstants
from cell_annotator._logging import logger
from cell_annotator.check import check_deps


def _get_specificity(
    genes: np.ndarray | list[str], clust_mask: np.ndarray, adata: sc.AnnData, use_raw: bool = True
) -> np.ndarray:
    """
    Calculate the specificity of the given genes for the given cluster mask.

    Parameters
    ----------
    genes
        Genes to calculate the specificity for.
    clust_mask
        Boolean mask for the cluster.
    adata
        AnnData object containing the expression data.
    use_raw
        Whether to use the raw data in the AnnData object.

    Returns
    -------
    Specificity values for the given genes.
    """
    if use_raw:
        values = adata.raw[:, genes].X
    else:
        values = adata[:, genes].X

    if issparse(values):
        values = values.toarray()
    expr_mask = values > 0

    fpr = np.sum(expr_mask & np.atleast_2d(~clust_mask).T, axis=0) / np.sum(~clust_mask)
    return 1 - fpr


def _get_auc(
    genes: np.ndarray | Sequence[str],
    clust_mask: np.ndarray,
    adata: sc.AnnData,
    use_raw: bool = True,
    use_rapids: bool = False,
):
    if use_raw:
        values = adata.raw[:, genes].X
    else:
        values = adata[:, genes].X

    if issparse(values):
        values = values.toarray()

    if use_rapids:
        check_deps("cupy", "cuml")
        import cupy as cp
        from cuml.metrics import roc_auc_score as gpu_roc_auc_score

        # Transfer data to GPU
        values_gpu = cp.asarray(values)
        clust_mask_gpu = cp.asarray(clust_mask, dtype=cp.float32)

        # Compute AUC scores on GPU
        auc_scores = cp.array([gpu_roc_auc_score(clust_mask_gpu, values_gpu[:, i]) for i in range(values_gpu.shape[1])])

        # Transfer results back to CPU
        return cp.asnumpy(auc_scores)
    else:
        # Compute AUC scores on CPU
        return np.array([roc_auc_score(clust_mask, x) for x in values.T])


def _try_sorting_dict_by_keys(unsorted_dict: dict):
    def extract_number(key):
        # Convert the key to a string if it's not already
        key_str = str(key)
        # Extract the first numeric part of the string using regex
        match = re.search(r"\d+", key_str)
        return int(match.group()) if match else float("inf")  # Use a high value for keys without numbers

    try:
        sorted_dict = {k: unsorted_dict[k] for k in sorted(unsorted_dict, key=extract_number)}
    except ValueError as e:
        logger.debug("Error during sorting: %s. Keeping original order.", e)
        sorted_dict = unsorted_dict

    return sorted_dict


def _format_annotation(df: pd.DataFrame, filter_by: str, cell_type_key: str) -> str:
    """Format the annotation DataFrame by filtering and generating summary strings."""
    filtered_df = df[df[cell_type_key] != filter_by]
    return "\n".join(
        f" - Cluster {index}: {row['marker_genes']} -> {row[cell_type_key]}" for index, row in filtered_df.iterrows()
    )


def _filter_by_category_size(adata: sc.AnnData, column: str, min_size: int) -> dict:
    """
    Remove small categories in a categorical column and set their entries to `None`.

    Prints and returns information about the removed clusters.

    Parameters
    ----------
    adata
        AnnData object with the column to modify.
    column
        Name of the categorical column in `adata.obs`.
    min_size
        Minimum number of elements a category must have to remain unchanged.

    Returns
    -------
    removed_info
        Information about the removed categories, including category names and their sizes.
    """
    # Count the size of each category
    category_counts = adata.obs[column].value_counts()

    # Identify small categories
    small_categories = category_counts[category_counts < min_size]

    # If no categories to remove, return early
    if small_categories.empty:
        return {}

    # Collect information to return
    removed_info = small_categories.to_dict()

    # Filter out cells in small categories
    adata._inplace_subset_obs(~adata.obs[column].isin(small_categories.index))

    # Remove unused categories
    adata.obs[column] = adata.obs[column].cat.remove_unused_categories()

    return removed_info


def _shuffle_cluster_key_categories_within_sample(
    adata: sc.AnnData,
    sample_key: str,
    cluster_key: str = "leiden",
    key_added: str | None = None,
    base_seed: int = 42,  # The base seed from which individual seeds are derived
):
    # Ensure the cluster_key is a categorical column
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    original_categories = adata.obs[cluster_key].cat.categories

    if key_added is None:
        key_added = f"{cluster_key}_shuffled"

    # Shuffle the categories within each sample
    for i, sample in enumerate(adata.obs[sample_key].unique()):
        # Get the cells for the current sample
        sample_cells = adata.obs[adata.obs[sample_key] == sample]

        # Get the unique cluster_key categories for the current sample
        cluster_categories = sample_cells[cluster_key].cat.categories

        # Derive a new seed for the current sample using the base_seed and sample index
        sample_rng = np.random.default_rng(base_seed + i)  # Use a new RNG for each sample

        # Shuffle the cluster categories but maintain the original order in .cat.categories
        shuffled_categories = sample_rng.permutation(cluster_categories)

        # Map the old cluster categories to the shuffled ones
        category_map = dict(zip(cluster_categories, shuffled_categories, strict=False))

        # Update the cluster_key column with the shuffled categories
        adata.obs.loc[adata.obs[sample_key] == sample, key_added] = adata.obs.loc[
            adata.obs[sample_key] == sample, cluster_key
        ].map(category_map)

    adata.obs[key_added] = adata.obs[key_added].cat.set_categories(original_categories)

    return adata


def _get_unique_cell_types(
    adata: sc.AnnData, keys: list[str], unknown_key: str = PackageConstants.unknown_name
) -> list[str]:
    """
    Given a set of .obs keys, return a list of all unique cell type names across these keys, excluding any "unknown" labels.

    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        List of .obs keys pointing to categorical adata.obs annotations.
    unknown_key
        The label to exclude from the list, by default "unknown".

    Returns
    -------
    List of unique cell type names.
    """
    unique_cell_types = set()
    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        categories = adata.obs[key].unique()
        unique_cell_types.update(cat for cat in categories if cat != unknown_key)

    return list(unique_cell_types)


def _get_consistent_ordering(
    adata: sc.AnnData, global_names_and_colors: dict[str, str], keys: list[str] | str
) -> dict[str, dict[str, str]]:
    consistent_label_sets = {}

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        labels = adata.obs[key].unique()
        consistent_label_sets[key] = {name: color for name, color in global_names_and_colors.items() if name in labels}
    return consistent_label_sets


def _validate_list_mapping(list_a: list, list_b: list, context: str | None = None) -> None:
    """
    Validate that the elements in list_b match the elements in list_a, ignoring NaN values.

    Parameters
    ----------
    list_a
        The original list of elements.
    list_b
        The list of elements after mapping.
    context
        Optional context for the error message.
    """
    # Convert to strings and filter out NaN values
    set_a = {str(item) for item in list_a if pd.notna(item)}
    set_b = {str(item) for item in list_b if pd.notna(item)}

    if set_a != set_b:
        added_elements = set_b - set_a
        removed_elements = set_a - set_b
        if added_elements or removed_elements:
            if context:
                error_message = f"New elements for context `{context}` differ from original elements."
            else:
                error_message = "New elements differ from original elements."
            if added_elements:
                error_message += f" Added elements: {', '.join(sorted(added_elements))}."
            if removed_elements:
                error_message += f" Removed elements: {', '.join(sorted(removed_elements))}."
            raise ValueError(error_message)


def _filter_marker_genes_to_adata(marker_genes: dict[str, list[str]], adata: sc.AnnData) -> dict[str, list[str]]:
    """Filter marker genes to only include those present in adata.var_names.

    Parameters
    ----------
    marker_genes
        Dictionary mapping cell types to lists of marker genes
    adata
        AnnData object containing gene expression data

    Returns
    -------
    Dictionary mapping cell types to filtered lists of marker genes
    """
    available_genes = set(adata.var_names)
    available_genes_lower = {g.lower() for g in available_genes}
    gene_map = {g.lower(): g for g in available_genes}  # map lower-case to original
    filtered_marker_genes = {}
    total_filtered = 0
    removed_cell_types = []

    logger.info("Filtering marker genes to only include those present in adata.var_names.")
    for cell_type, markers in marker_genes.items():
        filtered_markers = []
        filtered_out = set()
        for gene in markers:
            gene_l = gene.lower()
            if gene_l in available_genes_lower:
                # Use the AnnData var_name capitalization
                filtered_markers.append(gene_map[gene_l])
            else:
                filtered_out.add(gene)

        if filtered_out:
            total_filtered += len(filtered_out)
            logger.debug(
                "Cell type '%s': filtered %d marker genes not found in adata.var_names: %s",
                cell_type,
                len(filtered_out),
                ", ".join(sorted(filtered_out)),
            )

        if filtered_markers:
            filtered_marker_genes[cell_type] = filtered_markers
        else:
            removed_cell_types.append(cell_type)

    if total_filtered > 0 or removed_cell_types:
        logger.info(
            "Filtered %d marker genes and removed %d cell types with no marker genes left after filtering.",
            total_filtered,
            len(removed_cell_types),
        )
    return filtered_marker_genes
