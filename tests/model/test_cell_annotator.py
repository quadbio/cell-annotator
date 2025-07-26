import pytest
from tests.utils import expected_marker_genes, fibroblast_cell_types, neuronal_cell_types


class TestCellAnnotator:
    @pytest.mark.real_llm_query()
    def test_get_expected_cell_type_markers(self, cell_annotator_single):
        """Test getting expected cell type markers with single sample data."""
        cell_annotator = cell_annotator_single
        cell_annotator.get_expected_cell_type_markers()
        expected_markers = cell_annotator.expected_marker_genes
        print("Expected Markers:", expected_markers)

        assert expected_markers is not None
        assert isinstance(expected_markers, dict)

        neuron_markers_found = False
        fibroblast_markers_found = False

        for key, markers in expected_markers.items():
            print(f"Cell Type: {key}, Markers: {markers}")
            if any(neuron_synonym in key for neuron_synonym in neuronal_cell_types):
                if any(
                    any(marker in model_marker for model_marker in markers)
                    for marker in expected_marker_genes["Neuron"]
                ):
                    neuron_markers_found = True
            if any(fibroblast_synonym in key for fibroblast_synonym in fibroblast_cell_types):
                if any(
                    any(marker in model_marker for model_marker in markers)
                    for marker in expected_marker_genes["Fibroblast"]
                ):
                    fibroblast_markers_found = True

        assert neuron_markers_found
        assert fibroblast_markers_found

    @pytest.mark.real_llm_query()
    def test_annotate_clusters_single(self, cell_annotator_single):
        """Test annotating clusters with single sample data."""
        cell_annotator = cell_annotator_single
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

        # get the summary annotation string
        print(cell_annotator._get_annotation_summary_string())

    @pytest.mark.real_llm_query()
    def test_annotate_clusters_multi(self, cell_annotator_multi):
        """Test annotating clusters with multi-sample data."""
        cell_annotator = cell_annotator_multi
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

        # get the summary annotation string
        print(cell_annotator._get_annotation_summary_string())
