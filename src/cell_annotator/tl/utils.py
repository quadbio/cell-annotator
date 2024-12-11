from collections.abc import Sequence

import numpy as np
import scanpy as sc
from openai import OpenAI
from pydantic import BaseModel
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score

ResponseOutput = type[BaseModel]


def _query_openai(
    agent_description: str,
    instruction: str,
    model: str,
    response_format: ResponseOutput,
    other_messages: list | None = None,
    max_tokens: int | None = None,
) -> BaseModel:
    client = OpenAI()

    if other_messages is None:
        other_messages = []

    res = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": agent_description}, {"role": "user", "content": instruction}]
        + other_messages,
        response_format=response_format,
        max_tokens=max_tokens,
    )

    return res


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
