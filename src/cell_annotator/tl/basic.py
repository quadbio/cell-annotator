from pandas import Series
from tqdm.auto import tqdm

from cell_annotator._constants import ExpectedCellTypeOutput, ExpectedMarkerGeneOutput, PredictedCellTypeOutput, Prompts
from cell_annotator._logging import logger
from cell_annotator.tl.utils import _query_openai


def get_expected_cell_types(species: str, tissue: str, n_markers: int = 5, model: str = "gpt-4o-mini", **kwargs):
    """Query expected cell types per species and tissue.

    Parameters
    ----------
    species : str
        Species name.
    tissue : str
        Tissue name.
    n_markers : int, optional
        Number of markers to query per cell type, by default 5.
    model : str, optional
        OpenAI model name, by default "gpt-4o-mini".

    Returns
    -------
    str
        Expected cell types.
    str
        Expected cell type markers.

    """
    cell_type_prompt = Prompts.CELL_TYPE_PROMPT.format(species=species, tissue=tissue)
    agent_desc = Prompts.AGENT_DESCRIPTION.format(species=species)

    logger.info("Querying cell types.")
    res_types = _query_openai(
        agent_description=agent_desc,
        instruction=cell_type_prompt,
        response_format=ExpectedCellTypeOutput,
        model=model,
        **kwargs,
    )
    expected_cell_types = res_types.choices[0].message.parsed

    marker_gene_prompt = [
        {"role": "assistant", "content": "; ".join(expected_cell_types.expected_cell_types)},
        {"role": "user", "content": Prompts.CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)},
    ]

    logger.info("Querying cell type markers.")
    res_markers = _query_openai(
        agent_description=agent_desc,
        instruction=cell_type_prompt,
        other_messages=marker_gene_prompt,
        response_format=ExpectedMarkerGeneOutput,
        model=model,
        **kwargs,
    )
    expected_marker_genes = res_markers.choices[0].message.parsed

    return expected_cell_types, expected_marker_genes


def annotate_clusters(
    marker_genes: Series,
    species: str,
    tissue: str,
    expected_markers: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs,
):
    """Annotate clusters based on marker genes.

    Parameters
    ----------
    marker_genes : Series
        Marker genes per cluster.
    species : str
        Species name.
    tissue : str
        Tissue name.
    expected_markers : str, optional
        Expected markers, by default None.
    model : str, optional
        OpenAI model name, by default "gpt-4o-mini".

    Returns
    -------
    dict
        Annotation results.

    """
    answers = {}
    if expected_markers is None:
        logger.info("Querying expected markers.")
        expected_markers = get_expected_cell_types(species=species, tissue=tissue, max_tokens=800)[1]

    # parse expected markers into a string
    expected_markers_string = "\n".join(
        [
            f"{marker.cell_type_name}: {', '.join(marker.expected_marker_genes)}"
            for marker in expected_markers.expected_markers_per_cell_type
        ]
    )

    actual_markers_all = "\n".join([f'- Cluster {i}: {", ".join(gs)}' for i, gs in marker_genes.items()])
    agent_desc = Prompts.AGENT_DESCRIPTION.format(species=species)

    # loop over clusters to annotate
    logger.info("Looping over clusters to annotate.")
    for cluster in tqdm(marker_genes.index):
        actual_markers_cluster = ", ".join(marker_genes[cluster])

        # fill in the annotation prompt
        annotation_prompt = Prompts.ANNOTATION_PROMPT.format(
            species=species,
            tissue=tissue,
            actual_markers_all=actual_markers_all,
            cluster_id=cluster,
            actual_markers_cluster=actual_markers_cluster,
            expected_markers=expected_markers_string,
        )

        res = _query_openai(
            agent_description=agent_desc,
            instruction=annotation_prompt,
            response_format=PredictedCellTypeOutput,
            model=model,
            **kwargs,
        )
        answers[cluster] = res.choices[0].message.parsed

    return answers
