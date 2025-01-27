import pytest

from cell_annotator.cell_annotator import CellAnnotator

from .utils import expected_marker_genes, fibroblast_cell_types, get_example_data, neuronal_cell_types


class TestCellAnnotator:
    @pytest.fixture
    def cell_annotator(self):
        adata = get_example_data(n_cells=200, n_samples=2)

        return CellAnnotator(
            adata=adata,
            species="human",
            tissue="In vitro neurons and fibroblasts",
            stage="adult",
            cluster_key="leiden",
            sample_key="sample",
            model="gpt-4o-mini",
        )

    @pytest.mark.openai()
    def test_get_expected_cell_type_markers(self, cell_annotator):
        cell_annotator.get_expected_cell_type_markers()
        expected_markers = cell_annotator.expected_marker_genes
        print("Expected Markers:", expected_markers)

        assert expected_markers is not None
        assert isinstance(expected_markers, dict)

        neuron_markers_found = any(
            any(neuron_synonym in key for neuron_synonym in neuronal_cell_types)
            and set(expected_marker_genes["Neuron"]).intersection(expected_markers[key])
            for key in expected_markers
        )
        fibroblast_markers_found = any(
            any(fibroblast_synonym in key for fibroblast_synonym in fibroblast_cell_types)
            and set(expected_marker_genes["Fibroblast"]).intersection(expected_markers[key])
            for key in expected_markers
        )

        assert neuron_markers_found
        assert fibroblast_markers_found

    @pytest.mark.openai()
    def test_annotate_clusters(self, cell_annotator):
        # Step 1: Call get_cluster_markers and run checks
        cell_annotator.get_cluster_markers(min_auc=0.6)

        for sample in cell_annotator.sample_annotators.values():
            assert sample.marker_gene_dfs is not None
            assert sample.marker_genes is not None

            for _cluster, df in sample.marker_gene_dfs.items():
                assert not df.empty
                assert "gene" in df.columns
                assert "specificity" in df.columns
                assert "auc" in df.columns

            for _cluster, genes in sample.marker_genes.items():
                assert len(genes) > 0

        # Step 2: Call annotate_clusters and run checks
        cell_annotator.expected_marker_genes = expected_marker_genes
        cell_annotator.annotate_clusters(min_markers=1)

        for sample in cell_annotator.sample_annotators.values():
            print("Sample Annotation:\n", sample.annotation_df[["n_cells", "cell_type"]])

            neuron_annotation_found = any(
                neuron_synonym in sample.annotation_dict["0"].cell_type for neuron_synonym in neuronal_cell_types
            )
            fibroblast_annotation_found = any(
                fibroblast_synonym in sample.annotation_dict["1"].cell_type
                for fibroblast_synonym in fibroblast_cell_types
            )

            assert neuron_annotation_found
            assert fibroblast_annotation_found
