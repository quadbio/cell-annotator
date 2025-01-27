from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.random import binomial, negative_binomial

from cell_annotator._response_formats import CellTypeMappingOutput, PredictedCellTypeOutput
from cell_annotator.sample_annotator import SampleAnnotator


def get_example_data() -> AnnData:
    """Create example data for testing. Adapted from scanpy."""
    adata = AnnData(np.multiply(binomial(1, 0.15, (100, 20)), negative_binomial(2, 0.25, (100, 20))))
    # Adapt marker genes for cluster (so as to have some form of reasonable input)
    adata.X[0:10, 0:5] = np.multiply(binomial(1, 0.9, (10, 5)), negative_binomial(1, 0.5, (10, 5)))

    # Create cluster according to groups
    adata.obs["leiden"] = pd.Categorical(
        np.concatenate((np.zeros((10,), dtype=int), np.ones((90,), dtype=int))).astype(str)
    )

    # filter, normalize and log transform the data
    sc.pp.filter_cells(adata, min_counts=2)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


class TestSampleAnnotator:
    @pytest.fixture
    def sample_annotator(self):
        adata = get_example_data()

        return SampleAnnotator(
            adata=adata,
            sample_name="sample1",
            species="human",
            tissue="brain",
            stage="adult",
            cluster_key="leiden",
            model="gpt-4o-mini",
            max_tokens=100,
        )

    @patch("cell_annotator.sample_annotator.SampleAnnotator.query_openai")
    def test_annotate_clusters(self, mock_query_openai, sample_annotator):
        mock_response = PredictedCellTypeOutput(cell_type="Neuron")
        mock_query_openai.return_value = mock_response

        sample_annotator.marker_genes = {"0": ["gene1", "gene2"], "1": ["gene3", "gene4"]}
        sample_annotator.annotate_clusters(min_markers=1, expected_marker_genes=None)

        assert sample_annotator.annotation_dict["0"].cell_type == "Neuron"
        assert sample_annotator.annotation_dict["1"].cell_type == "Neuron"

    @patch("cell_annotator.sample_annotator.SampleAnnotator.query_openai")
    def test_harmonize_annotations(self, mock_query_openai, sample_annotator):
        mock_response = CellTypeMappingOutput(mapped_global_name="Neuron")
        mock_query_openai.return_value = mock_response

        sample_annotator.annotation_df = pd.DataFrame({"cell_type": ["type1", "type2"]})
        sample_annotator.harmonize_annotations(global_cell_type_list=["Neuron", "Astrocyte"])

        assert sample_annotator.local_cell_type_mapping["type1"] == "Neuron"
        assert sample_annotator.local_cell_type_mapping["type2"] == "Neuron"
        assert "cell_type_harmonized" in sample_annotator.annotation_df.columns
        assert all(sample_annotator.annotation_df["cell_type_harmonized"] == "Neuron")

    def test_get_cluster_markers(self, sample_annotator):
        sample_annotator.get_cluster_markers(min_auc=0.5)

        assert sample_annotator.marker_gene_dfs is not None
        assert sample_annotator.marker_genes is not None

        for _cluster, df in sample_annotator.marker_gene_dfs.items():
            assert not df.empty
            assert "gene" in df.columns
            assert "specificity" in df.columns
            assert "auc" in df.columns

        for _cluster, genes in sample_annotator.marker_genes.items():
            assert len(genes) > 0
