from pydantic import BaseModel


class Prompts:
    """Prompts for scRNA-seq cell annotation tasks.

    These were heavily inspired by https://github.com/VPetukhov/GPTCellAnnotator.
    """

    CELL_TYPE_PROMPT = (
        "Provide me a comprehensive list of cell types that are expected in {species} {tissue} at stage `{stage}`."
    )
    CELL_TYPE_MARKER_PROMPT = "Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Make sure that you provided markers for **each** cell type you mentioned above."

    ANNOTATION_PROMPT = """
    You need to annotate a {species} {tissue} dataset at stage `{stage}`. You found gene markers for each cluster and need to determine clusters' identities.
    Below is a short list of markers for each cluster:

    {actual_markers_all}

    We expect the following cell types (with associated marker genes) to be present in the dataset, however, some might be absent, and additionall cell types may also be present:
    {expected_markers}

    Determine cell type and state for cluster {cluster_id} (markers {actual_markers_cluster})
    Only output data in the following format:
    ```
    - marker_gene_description: description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y
    - cell_type_annotation: name of this cell type. If unknown, use 'Unknown'.
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
    reason_for_failure: str | None = None  # Optional field for failure reasons

    @classmethod
    def default_failure(cls, failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(expected_cell_types=[], reason_for_failure=failure_reason)


class CellTypeMarkers(BaseModel):
    """Expected marker genes for a cell type."""

    cell_type_name: str
    expected_marker_genes: list[str]


class ExpectedMarkerGeneOutput(BaseModel):
    """Marker gene output."""

    expected_markers_per_cell_type: list[CellTypeMarkers]
    reason_for_failure: str | None = None  # Optional field for failure reasons

    @classmethod
    def default_failure(cls, failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(expected_markers_per_cell_type=[], reason_for_failure=failure_reason)


class PredictedCellTypeOutput(BaseModel):
    """Cell type annotation results."""

    marker_gene_description: str
    cell_type_annotation: str
    cell_state_annotation: str
    annotation_confidence: str
    reason_for_confidence_estimate: str

    @classmethod
    def default_failure(cls, failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(
            marker_gene_description="Unknown",
            cell_type_annotation="Unknown",
            cell_state_annotation="Unknown",
            annotation_confidence="Low",
            reason_for_confidence_estimate=failure_reason,
        )
