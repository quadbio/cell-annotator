class Prompts:
    """Prompts for scRNA-seq cell annotation tasks."""

    CELL_TYPE_PROMPT = "Provide me a comprehensive hierarchy of cell types that are expected in {species} {tissue}. Write it as an unordered list."
    CELL_TYPE_MARKER_PROMPT = "Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Provide the answer as an unordered list without any additional comments. Example: `- Type X: gene_id1, gene_id2, ...`. Make sure that you provided markers for **each** cell type you mentioned above."

    ANNOTATION_PROMPT = """
    You need to annotate a {species} {tissue} dataset. You found gene markers for each cluster and need to determine clusters' identities.
    Below is a short list of markers for each cluster:

    {marker_list}

    We expect the following cell types to be present in the dataset, however additionall types may also be present:
    {expected_markers}

    Determine cell type and state for cluster {cluster_id} (markers {cli_markers})
    Only output data in the following format:
    ```
    - Marker description: (description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y)
    - Cell type: (cell type name)
    - Cell state: (cell state if there is anything specific, 'normal' otherwise)
    - Confidence: one of 'low', 'medium', 'high'
    - Reason: (reason for the confidence estimate)
    ```
    """.strip()

    AGENT_DESCRIPTION = (
        "You're an expert bioinformatician, proficient in scRNA-seq analysis with background in {species} cell biology."
    )
