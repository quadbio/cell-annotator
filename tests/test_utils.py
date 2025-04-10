from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cell_annotator._response_formats import OutputForTesting
from cell_annotator.utils import (
    _filter_by_category_size,
    _format_annotation,
    _get_auc,
    _get_consistent_ordering,
    _get_specificity,
    _get_unique_cell_types,
    _query_openai,
    _shuffle_cluster_key_categories_within_sample,
    _try_sorting_dict_by_keys,
    _validate_list_mapping,
)


@pytest.fixture
def setup_data():
    genes = np.array(["gene1", "gene2"])
    clust_mask = np.array([True, False, True, False])

    # Create raw count data with gene names
    raw_counts = np.array([[1, 0], [0, 1], [1, 1], [4, 0]])
    adata = sc.AnnData(X=raw_counts, var=pd.DataFrame(index=genes))
    adata.raw = adata.copy()  # Set raw data

    # Normalize and log transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return genes, clust_mask, adata


class TestUtils:
    def test_get_specificity(self, setup_data):
        genes, clust_mask, adata = setup_data
        specificity = _get_specificity(genes, clust_mask, adata)
        assert specificity.shape == (2,)
        assert np.all(specificity >= 0) and np.all(specificity <= 1)

    def test_get_auc(self, setup_data):
        genes, clust_mask, adata = setup_data
        auc = _get_auc(genes, clust_mask, adata)
        assert auc.shape == (2,)
        assert np.all(auc >= 0) and np.all(auc <= 1)

    def test_try_sorting_dict_by_keys(self):
        unsorted_dict = {"item10": 1, "item2": 2, "item1": 3}
        sorted_dict = _try_sorting_dict_by_keys(unsorted_dict)
        assert list(sorted_dict.keys()) == ["item1", "item2", "item10"]

    def test_format_annotation(self):
        df = pd.DataFrame(
            {
                "marker_genes": ["gene1", "gene2"],
                "cell_type_harmonized": ["type1", "type2"],
            }
        )
        formatted = _format_annotation(df, filter_by="type1", cell_type_key="cell_type_harmonized")
        assert formatted == " - Cluster 1: gene2 -> type2"

    def test_filter_by_category_size(self):
        adata = sc.AnnData(obs=pd.DataFrame({"category": ["A", "A", "B", "C", "C", "C"]}))
        adata.obs["category"] = adata.obs["category"].astype("category")  # Convert to categorical type
        removed_info = _filter_by_category_size(adata, column="category", min_size=3)
        assert removed_info == {"A": 2, "B": 1}
        assert "A" not in adata.obs["category"].cat.categories
        assert "B" not in adata.obs["category"].cat.categories

    def test_shuffle_cluster_key_categories_within_sample(self):
        adata = sc.AnnData(obs=pd.DataFrame({"sample": ["s1", "s1", "s2", "s2"], "leiden": ["A", "B", "A", "B"]}))
        shuffled_adata = _shuffle_cluster_key_categories_within_sample(adata, sample_key="sample")
        assert "leiden_shuffled" in shuffled_adata.obs.columns

    def test_get_unique_cell_types(self):
        adata = sc.AnnData(
            obs=pd.DataFrame({"key1": ["type1", "type2", "unknown"], "key2": ["type3", "type1", "type2"]})
        )
        unique_cell_types = _get_unique_cell_types(adata, keys=["key1", "key2"], unknown_key="unknown")
        assert set(unique_cell_types) == {"type1", "type2", "type3"}

    def test_get_consistent_ordering(self):
        adata = sc.AnnData(obs=pd.DataFrame({"key1": ["type1", "type2"], "key2": ["type3", "type1"]}))
        global_names_and_colors = {"type1": "red", "type2": "blue", "type3": "green"}
        consistent_ordering = _get_consistent_ordering(adata, global_names_and_colors, keys=["key1", "key2"])
        assert consistent_ordering == {
            "key1": {"type1": "red", "type2": "blue"},
            "key2": {"type1": "red", "type3": "green"},
        }

    def test_validate_list_mapping(self):
        list_a = ["a", "b", "c"]
        list_b = ["c", "b", "a"]
        _validate_list_mapping(list_a, list_b)
        with pytest.raises(ValueError):
            _validate_list_mapping(list_a, ["d", "e", "f"])

    @patch("cell_annotator.utils.OpenAI")
    def test_query_openai(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.parsed = OutputForTesting(parsed_response="parsed_response")
        mock_client.beta.chat.completions.parse.return_value = mock_response

        response = _query_openai(
            agent_description="Test agent",
            instruction="Test instruction",
            model="gpt-4o-mini",
            response_format=OutputForTesting,
        )

        assert response.parsed_response == "parsed_response"
        mock_client.beta.chat.completions.parse.assert_called_once()

    @pytest.mark.openai()
    def test_query_openai_actual(self):
        response = _query_openai(
            agent_description="Test agent",
            instruction="Test instruction",
            model="gpt-4o-mini",
            response_format=OutputForTesting,
            max_completion_tokens=100,
        )

        assert response is not None
        assert isinstance(response, OutputForTesting)
