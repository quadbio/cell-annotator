from pydantic import BaseModel


class CellTypeColor(BaseModel):
    """Color assignment for a cell type."""

    original_cell_type_label: str
    assigned_color: str


class CellTypeColorOutput(BaseModel):
    """Color assignment for cell types."""

    cell_type_to_color_mapping: list[CellTypeColor]
    reason_for_failure: str | None = None

    @classmethod
    def default_failure(cls, failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(cell_type_to_color_mapping=[], reason_for_failure=failure_reason)


class LabelOrderOutput(BaseModel):
    """Dict of cell type labels and colors."""

    ordered_cell_type_list: list[str]
    reason_for_failure: str | None = None

    @classmethod
    def default_failure(cls, failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(ordered_cell_type_list=[], reason_for_failure=failure_reason)


class ExpectedCellTypeOutput(BaseModel):
    """List of cell types"""

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
            annotation_confidence="Unknown",
            reason_for_confidence_estimate=failure_reason,
        )
