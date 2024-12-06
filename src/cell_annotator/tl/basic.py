import time

import openai
from openai import OpenAI
from pandas import DataFrame, Series
from tqdm.auto import tqdm

from cell_annotator._constants import Prompts
from cell_annotator._logging import logger


def _query_openai(
    agent_description: str,
    instruction: str,
    other_messages: list | None = None,
    n_repeats: int = 4,
    model: str = "gpt-3.5-turbo",
    **kwargs,
):
    client = OpenAI()

    if other_messages is None:
        other_messages = []

    for _ in range(n_repeats):
        res = None
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": agent_description}, {"role": "user", "content": instruction}]
                + other_messages,
                **kwargs,
            )
            break
        except openai.OpenAIError as e:
            print(e)
            time.sleep(1)
            continue

    if res is None:
        raise ValueError("Failed to get response from OpenAI")

    return res


def get_expected_cell_types(species: str, tissue: str, n_markers: int = 5, **kwargs):
    """Query expected cell types per species and tissue.

    Parameters
    ----------
    species : str
        Species name.
    tissue : str
        Tissue name.
    n_markers : int, optional
        Number of markers to query per cell type, by default 5.

    Returns
    -------
    str
        Expected cell types.
    str
        Expected cell type markers.

    """
    prompt = Prompts.CELL_TYPE_PROMPT.format(species=species, tissue=tissue)
    agent_desc = f"You're an expert in {species} cell biology."

    logger.info("Querying cell types...")
    res_types = _query_openai(agent_desc, agent_desc + " " + prompt, **kwargs)
    expected_types = res_types.choices[0].message.content

    prompts2 = [
        {"role": "assistant", "content": expected_types},
        {"role": "user", "content": Prompts.CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)},
    ]

    logger.info("Querying cell type markers...")
    res_markers = _query_openai(agent_desc, agent_desc + " " + prompt, prompts2, **kwargs)
    expected_markers = res_markers.choices[0].message.content

    return expected_types, expected_markers


def annotate_clusters(
    marker_genes: Series,
    species: str,
    tissue: str,
    expected_markers: str | None = None,
    annotation_prompt: str = Prompts.ANNOTATION_PROMPT,
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
    annotation_prompt : str, optional
        Annotation prompt, by default Prompts.ANNOTATION_PROMPT.

    Returns
    -------
    dict
        Annotation results.

    """
    answers = {}
    if expected_markers is None:
        expected_markers = get_expected_cell_types(species=species, tissue=tissue, model="gpt-4", max_tokens=800)[1]

    marker_txt = "\n".join([f'- Cluster {i}: {", ".join(gs)}' for i, gs in marker_genes.items()])
    agent_desc = Prompts.AGENT_DESCRIPTION.format(species=species)

    for cli in tqdm(marker_genes.index):
        cli_markers = ", ".join(marker_genes[cli])

        pf = annotation_prompt.format(
            species=species,
            tissue=tissue,
            marker_list=marker_txt,
            cluster_id=cli,
            cli_markers=cli_markers,
            expected_markers=expected_markers,
        )

        res = _query_openai(agent_desc, pf, **kwargs)
        answers[cli] = res.choices[0].message.content

    return answers


def parse_annotation(annotation_res: str):
    """Parse annotation string into a DataFrame.

    Parameters
    ----------
    annotation_res : str
        Annotation results.

    Returns
    -------
    DataFrame
        Parsed annotation results.

    """
    ann_df = DataFrame(
        Series(annotation_res)
        .map(lambda ann: dict([l.strip().lstrip("- ").split(": ") for l in ann.split("\n") if len(l.strip()) > 0]))
        .to_dict()
    ).T[["Marker description", "Cell type", "Cell state", "Confidence", "Reason"]]

    ann_df["Cell type, raw"] = ann_df["Cell type"]
    ann_df["Cell type"] = ann_df["Cell type"].map(
        lambda x: x.replace("cells", "")
        .replace("cell", "")
        .replace(".", "")
        .strip()
        .split("(")[0]
        .split(",")[0]
        .strip()
        .replace("  ", " ")
    )

    ann_df["Confidence"] = ann_df["Confidence"].str.rstrip(".").str.lower()

    return ann_df
