from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.random import binomial, negative_binomial

from cell_annotator._response_formats import CellTypeMappingOutput, PredictedCellTypeOutput
from cell_annotator.sample_annotator import SampleAnnotator

# Declare the dictionary of expected marker genes
expected_marker_genes = {
    "Neuron": ["MAP2", "NEFL", "RBFOX3", "SYN1", "GAP43", "DCX", "TUBB3", "NEUROD1", "STMN2", "ENO2"],
    "Fibroblast": ["COL1A1", "COL3A1", "VIM", "ACTA2", "FAP", "PDGFRA", "THY1", "FN1", "SPARC", "S100A4"],
}


def get_example_data(n_cells: int = 100) -> AnnData:
    """Create example data for testing. Adapted from scanpy.

    The data consists of two clusters with different marker genes. The first cluster is enriched for neuronal markers and the second cluster is enriched for fibroblast markers."""
    gene_names = expected_marker_genes["Neuron"] + expected_marker_genes["Fibroblast"]
    n_genes = len(gene_names)
    adata = AnnData(np.multiply(binomial(1, 0.15, (n_cells, n_genes)), negative_binomial(2, 0.25, (n_cells, n_genes))))
    adata.var_names = gene_names

    # Create marker genes for the two clusters
    n_group_1 = np.floor(0.3 * n_cells).astype(int)
    n_group_2 = n_cells - n_group_1
    n_marker_genes = int(n_genes / 2)

    adata.X[:n_group_1, :10] = np.multiply(
        binomial(1, 0.9, (n_group_1, n_marker_genes)), negative_binomial(1, 0.5, (n_group_1, n_marker_genes))
    )
    adata.X[n_group_1:, 10:] = np.multiply(
        binomial(1, 0.9, (n_group_2, n_marker_genes)), negative_binomial(1, 0.5, (n_group_2, n_marker_genes))
    )

    # Create cluster according to groups
    adata.obs["leiden"] = pd.Categorical(np.concatenate((n_group_1 * ["0"], n_group_2 * ["1"])))

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
        )

    @patch("cell_annotator.sample_annotator.SampleAnnotator.query_openai")
    def test_annotate_clusters(self, mock_query_openai, sample_annotator):
        mock_response = PredictedCellTypeOutput(cell_type="Neuron")
        mock_query_openai.return_value = mock_response

        sample_annotator.marker_genes = {
            "0": expected_marker_genes["Neuron"][:2],
            "1": expected_marker_genes["Fibroblast"][:2],
        }
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
        sample_annotator.get_cluster_markers(min_auc=0.6)

        assert sample_annotator.marker_gene_dfs is not None
        assert sample_annotator.marker_genes is not None

        for _cluster, df in sample_annotator.marker_gene_dfs.items():
            print(f"Cluster {_cluster} Marker Genes:")
            print(df)
            assert not df.empty
            assert "gene" in df.columns
            assert "specificity" in df.columns
            assert "auc" in df.columns

        for _cluster, genes in sample_annotator.marker_genes.items():
            assert len(genes) > 0

    @pytest.mark.openai()
    def test_annotate_clusters_actual(self, sample_annotator):
        sample_annotator.get_cluster_markers(min_auc=0.6)
        sample_annotator.annotate_clusters(min_markers=1, expected_marker_genes=expected_marker_genes)

        print("Annotations:")
        print(sample_annotator.annotation_df[["n_cells", "cell_type"]])
        assert sample_annotator.annotation_dict["0"].cell_type == "Neuron"
        assert sample_annotator.annotation_dict["1"].cell_type == "Fibroblast"
