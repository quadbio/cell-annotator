"""Response formats for the cell annotator API."""

from pydantic import BaseModel, Field


class BaseOutput(BaseModel):
    """Base class for output models with a failure reason."""

    reason_for_failure: str | None = None

    @classmethod
    def default_failure(cls: type["BaseOutput"], failure_reason: str = "Manual fallback due to model failure."):
        """Return a default output in case of failure, with a custom failure reason."""
        return cls(reason_for_failure=failure_reason)


class CellTypeColor(BaseOutput):
    """Color assignment for a cell type."""

    original_cell_type_label: str = Field(default_factory=str)
    assigned_color: str = Field(default_factory=str)


class CellTypeColorOutput(BaseOutput):
    """Color assignment for cell types."""

    cell_type_to_color_mapping: list[CellTypeColor] = Field(default_factory=list)


class CellTypeListOutput(BaseOutput):
    """List of cell type names."""

    cell_type_list: list[str] = Field(default_factory=list)


class CellTypeMappingOutput(BaseOutput):
    """Dict mapping old to new cell type names."""

    mapped_global_name: str = Field(default_factory=str)


class CellTypeMarkers(BaseOutput):
    """Expected marker genes for a cell type."""

    cell_type_name: str = Field(default_factory=str)
    expected_marker_genes: list[str] = Field(default_factory=list)


class ExpectedMarkerGeneOutput(BaseOutput):
    """Marker gene output."""

    expected_markers_per_cell_type: list[CellTypeMarkers] = Field(default_factory=list)


class PredictedCellTypeOutput(BaseOutput):
    """Cell type annotation results."""

    marker_gene_description: str = Field(default_factory=lambda: "Unknown")
    cell_type: str = Field(default_factory=lambda: "Unknown")
    cell_state: str = Field(default_factory=lambda: "Unknown")
    annotation_confidence: str = Field(default_factory=lambda: "Unknown")
    reason_for_confidence_estimate: str = Field(default_factory=lambda: "Unknown")


class OutputForTesting(BaseOutput):
    """Output class for testing purposes."""

    parsed_response: str = Field(default_factory=str)
