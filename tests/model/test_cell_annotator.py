import pytest
from flaky import flaky
from tests.utils import expected_marker_genes, fibroblast_cell_types, neuronal_cell_types


class TestCellAnnotator:
    @flaky
    @pytest.mark.real_llm_query()
    def test_get_expected_cell_type_markers(self, cell_annotator_single, provider_name):
        """Test getting expected cell type markers with single sample data."""
        cell_annotator = cell_annotator_single
        cell_annotator.get_expected_cell_type_markers()
        expected_markers = cell_annotator.expected_marker_genes

        assert expected_markers is not None, f"[{provider_name}] expected_marker_genes is None"
        assert isinstance(expected_markers, dict), (
            f"[{provider_name}] expected_marker_genes is not a dict: got {type(expected_markers).__name__}"
        )
        assert expected_markers, (
            f"[{provider_name}] LLM returned an empty marker-gene dict — likely a structured-output "
            f"parsing fallback. Full response: {expected_markers!r}"
        )

        def _find_matches(synonyms: list[str], expected: list[str]) -> tuple[list[str], list[str]]:
            matching_keys = [k for k in expected_markers if any(syn in k for syn in synonyms)]
            hits = [
                (k, marker)
                for k in matching_keys
                for marker in expected
                if any(marker in m for m in expected_markers[k])
            ]
            return matching_keys, hits

        neuron_keys, neuron_hits = _find_matches(neuronal_cell_types, expected_marker_genes["Neuron"])
        fibroblast_keys, fibroblast_hits = _find_matches(fibroblast_cell_types, expected_marker_genes["Fibroblast"])

        context = (
            f"[{provider_name}] returned {len(expected_markers)} cell types: {list(expected_markers)}. "
            f"Full response: {expected_markers!r}"
        )

        assert neuron_keys, f"No returned cell-type key matched any neuron synonym {neuronal_cell_types}. {context}"
        assert neuron_hits, (
            f"Neuron-like keys {neuron_keys} contained none of the expected markers "
            f"{expected_marker_genes['Neuron']}. {context}"
        )
        assert fibroblast_keys, (
            f"No returned cell-type key matched any fibroblast synonym {fibroblast_cell_types}. {context}"
        )
        assert fibroblast_hits, (
            f"Fibroblast-like keys {fibroblast_keys} contained none of the expected markers "
            f"{expected_marker_genes['Fibroblast']}. {context}"
        )

    @flaky
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

    @flaky
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
