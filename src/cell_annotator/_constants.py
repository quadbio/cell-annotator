from pydantic import BaseModel


class Prompts:
    """Prompts for scRNA-seq cell annotation tasks."""

    CELL_TYPE_PROMPT = "Provide me a comprehensive hierarchy of cell types that are expected in {species} {tissue}."
    CELL_TYPE_MARKER_PROMPT = "Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Make sure that you provided markers for **each** cell type you mentioned above."

    ANNOTATION_PROMPT = """
    You need to annotate a {species} {tissue} dataset. You found gene markers for each cluster and need to determine clusters' identities.
    Below is a short list of markers for each cluster:

    {actual_markers_all}

    We expect the following cell types (with associated marker genes) to be present in the dataset, however, some might be absent, and additionall cell types may also be present:
    {expected_markers}

    Determine cell type and state for cluster {cluster_id} (markers {actual_markers_cluster})
    Only output data in the following format:
    ```
    - marker_gene_description: description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y
    - cell_type_annotation: name of this cell type. If unknown, use 'Unknown'. Capitalize the first letter.
    - cell_state_annotation: cell state if there is anything specific, 'Normal' otherwise
    - annotation_confidence: one of 'Low', 'Medium', 'High'
    - reason_for_confidence_estimate: reason for the confidence estimate
    ```
    """.strip()

    AGENT_DESCRIPTION = (
        "You're an expert bioinformatician, proficient in scRNA-seq analysis with background in {species} cell biology."
    )


class ExpectedCellTypeOutput(BaseModel):
    """Expected cell types output."""

    expected_cell_types: list[str]


class CellTypeMarkers(BaseModel):
    """Expected marker genes for a cell type."""

    cell_type_name: str
    expected_marker_genes: list[str]


class ExpectedMarkerGeneOutput(BaseModel):
    """Marker gene output."""

    expected_markers_per_cell_type: list[CellTypeMarkers]


class PredictedCellTypeOutput(BaseModel):
    """Cell type annotation results."""

    marker_gene_description: str
    cell_type_annotation: str
    cell_state_annotation: str
    annotation_confidence: str
    reason_for_confidence_estimate: str
