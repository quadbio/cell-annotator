from collections.abc import Sequence

import numpy as np
import openai
import scanpy as sc
from openai import OpenAI
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score

from cell_annotator._constants import ExpectedCellTypeOutput, ExpectedMarkerGeneOutput, PredictedCellTypeOutput
from cell_annotator._logging import logger

ResponseOutput = ExpectedCellTypeOutput | ExpectedMarkerGeneOutput | PredictedCellTypeOutput


def _query_openai(
    agent_description: str,
    instruction: str,
    model: str,
    response_format: ResponseOutput,
    other_messages: list | None = None,
    max_tokens: int | None = None,
) -> ResponseOutput:
    client = OpenAI()

    if other_messages is None:
        other_messages = []
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": agent_description}, {"role": "user", "content": instruction}]
            + other_messages,
            response_format=response_format,
            max_tokens=max_tokens,
        )

        response = completion.choices[0].message
        if response.parsed:
            return response.parsed
        elif response.refusal:
            failure_reason = f"Model refused to respond: {response.refusal}"
            logger.warning(failure_reason)
            return response_format.default_failure(failure_reason=failure_reason)
        else:
            failure_reason = "Unknown model failure."
            logger.warning(failure_reason)
            return response_format.default_failure(failure_reason=failure_reason)
    except openai.LengthFinishReasonError:
        failure_reason = "Maximum number of tokens exceeded. Try increasing `max_tokens`."
        logger.warning(failure_reason)
        return response_format.default_failure(failure_reason=failure_reason)

    except openai.OpenAIError as e:
        failure_reason = f"OpenAI API error: {str(e)}"
        logger.warning(failure_reason)
        return response_format.default_failure(failure_reason=failure_reason)


def _get_specificity(
    genes: np.ndarray | Sequence[str], clust_mask: np.ndarray, adata: sc.AnnData, use_raw: bool = True
):
    if use_raw:
        values = adata.raw[:, genes].X
    else:
        values = adata[:, genes].X

    if issparse(values):
        values = values.toarray()
    expr_mask = values > 0

    fpr = np.sum(expr_mask & np.atleast_2d(~clust_mask).T, axis=0) / np.sum(~clust_mask)
    return 1 - fpr


def _get_auc(genes: np.ndarray | Sequence[str], clust_mask: np.ndarray, adata: sc.AnnData, use_raw: bool = True):
    if use_raw:
        values = adata.raw[:, genes].X
    else:
        values = adata[:, genes].X

    if issparse(values):
        values = values.toarray()

    return np.array([roc_auc_score(clust_mask, x) for x in values.T])


def _try_sorting_dict_by_keys(unsorted_dict):
    try:
        sorted_dict = {k: unsorted_dict[k] for k in sorted(unsorted_dict, key=lambda x: int(x))}
    except ValueError:
        logger.debug("Cluster keys cannot be converted to integers. Keeping original order.")
        sorted_dict = unsorted_dict

    return sorted_dict


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
