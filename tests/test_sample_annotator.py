from unittest.mock import patch

import pandas as pd
import pytest
from tests.utils import (
    expected_marker_genes,
    fibroblast_cell_types,
    neuronal_cell_types,
)

from cell_annotator._response_formats import (
    CellTypeMappingOutput,
    PredictedCellTypeOutput,
)


class TestSampleAnnotator:
    @patch("cell_annotator.sample_annotator.SampleAnnotator.query_llm")
    def test_annotate_clusters(self, mock_query_llm, sample_annotator):
        """Test annotate_clusters with mocked response across all providers."""
        mock_response = PredictedCellTypeOutput(cell_type="Neuron")
        mock_query_llm.return_value = mock_response

        sample_annotator.marker_genes = {
            "0": expected_marker_genes["Neuron"][:2],
            "1": expected_marker_genes["Fibroblast"][:2],
        }
        sample_annotator.annotate_clusters(min_markers=1, expected_marker_genes=None)

        assert sample_annotator.annotation_dict["0"].cell_type == "Neuron"
        assert sample_annotator.annotation_dict["1"].cell_type == "Neuron"

    @patch("cell_annotator.sample_annotator.SampleAnnotator.query_llm")
    def test_harmonize_annotations(self, mock_query_llm, sample_annotator):
        """Test harmonize_annotations with mocked response across all providers."""
        mock_response = CellTypeMappingOutput(mapped_global_name="Neuron")
        mock_query_llm.return_value = mock_response

        sample_annotator.annotation_df = pd.DataFrame({"cell_type": ["type1", "type2"]})
        sample_annotator.harmonize_annotations(global_cell_type_list=["Neuron", "Astrocyte"])

        assert sample_annotator.local_cell_type_mapping["type1"] == "Neuron"
        assert sample_annotator.local_cell_type_mapping["type2"] == "Neuron"
        assert "cell_type_harmonized" in sample_annotator.annotation_df.columns
        assert all(sample_annotator.annotation_df["cell_type_harmonized"] == "Neuron")

    def test_get_cluster_markers(self, sample_annotator):
        sample_annotator.get_cluster_markers(min_auc=0.6)

        assert sample_annotator.marker_gene_dfs is not None
        assert sample_annotator.marker_genes is not None

        for _cluster, df in sample_annotator.marker_gene_dfs.items():
            assert not df.empty
            assert "gene" in df.columns
            assert "specificity" in df.columns
            assert "auc" in df.columns

        for _cluster, genes in sample_annotator.marker_genes.items():
            assert len(genes) > 0

    @pytest.mark.real_llm_query()
    def test_annotate_clusters_actual(self, sample_annotator):
        sample_annotator.get_cluster_markers(min_auc=0.6)
        sample_annotator.annotate_clusters(min_markers=1, expected_marker_genes=expected_marker_genes)

        neuron_annotation_found = any(
            neuron_synonym in sample_annotator.annotation_dict["0"].cell_type for neuron_synonym in neuronal_cell_types
        )
        fibroblast_annotation_found = any(
            fibroblast_synonym in sample_annotator.annotation_dict["1"].cell_type
            for fibroblast_synonym in fibroblast_cell_types
        )

        assert neuron_annotation_found
        assert fibroblast_annotation_found
