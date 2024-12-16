from collections.abc import Sequence

import numpy as np
import openai
import scanpy as sc
from openai import OpenAI
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score

from cell_annotator._constants import ExpectedCellTypeOutput, ExpectedMarkerGeneOutput, PredictedCellTypeOutput

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
            raise ValueError(f"The model refused to respond: {response.refusal}")
    except openai.LengthFinishReasonError as e:
        raise ValueError("Maximum number of tokens exceeded. Try increasing `max_tokens`.") from e
    except Exception as e:
        raise e

    raise ValueError("Failed to get a valid response from the model.")


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
