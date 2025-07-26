import pytest

from cell_annotator.model.obs_beautifier import ObsBeautifier


class TestObsBeautifier:
    @pytest.mark.real_llm_query()
    def test_reorder_and_color_clusters(self, cell_annotator_single):
        """Test reordering and coloring clusters."""

        cell_annotator = cell_annotator_single
        # Add a second annotation key to the adata object
        cell_annotator.adata.obs["leiden_2"] = cell_annotator.adata.obs["leiden"].copy()

        # Map cluster names to meaningful names for testing
        cell_annotator.adata.obs["leiden"] = cell_annotator.adata.obs["leiden"].map(
            {"0": "Neuron Cluster", "1": "Fibroblast Cluster"}
        )
        cell_annotator.adata.obs["leiden_2"] = cell_annotator.adata.obs["leiden_2"].map(
            {"0": "Neuron Cluster", "1": "Fibroblast Cluster"}
        )

        # Call reorder_and_color_clusters and run checks
        beautifier = ObsBeautifier(adata=cell_annotator.adata)
        beautifier.reorder_and_color(keys=["leiden", "leiden_2"], assign_colors=True)

        for key in ["leiden", "leiden_2"]:
            assert f"{key}_colors" in cell_annotator.adata.uns
            assert len(cell_annotator.adata.uns[f"{key}_colors"]) == cell_annotator.adata.obs[key].nunique()
