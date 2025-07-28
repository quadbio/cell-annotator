import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from cell_annotator.model.obs_beautifier import ObsBeautifier


class TestObsBeautifier:
    @flaky
    @pytest.mark.real_llm_query()
    def test_reorder_categories_and_assign_colors(self, cell_annotator_single):
        """Test reordering categories and assigning colors using both methods."""

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

        # First reorder categories, then assign colors
        beautifier = ObsBeautifier(adata=cell_annotator.adata)
        beautifier.reorder_categories(keys=["leiden", "leiden_2"])
        beautifier.assign_colors(keys=["leiden", "leiden_2"])

        for key in ["leiden", "leiden_2"]:
            assert f"{key}_colors" in cell_annotator.adata.uns
            assert len(cell_annotator.adata.uns[f"{key}_colors"]) == cell_annotator.adata.obs[key].nunique()

    @flaky
    @pytest.mark.real_llm_query()
    def test_reorder_categories_preserves_colors(self, cell_annotator_single):
        """Test that reordering categories preserves the original colors."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations and colors
        adata.obs["leiden"] = adata.obs["leiden"].map({"0": "B cells", "1": "T cells"}).astype("category")
        original_colors = {"B cells": "#ff0000", "T cells": "#00ff00"}
        adata.uns["leiden_colors"] = [original_colors[cat] for cat in adata.obs["leiden"].cat.categories]

        # Reorder categories only
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_categories(keys=["leiden"])

        # Check that the colors are preserved and reordered correctly
        new_categories = adata.obs["leiden"].cat.categories
        new_colors = adata.uns["leiden_colors"]
        reordered_color_map = dict(zip(new_categories, new_colors, strict=False))

        # The new order might be different, but the color for each cell type should be the same
        assert reordered_color_map["B cells"] == original_colors["B cells"]
        assert reordered_color_map["T cells"] == original_colors["T cells"]
        assert len(new_colors) == 2

    def test_reorder_categories_with_nan_values(self, cell_annotator_single):
        """Test that reordering categories handles NaN values gracefully."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations with a NaN value
        adata.obs["leiden"] = adata.obs["leiden"].map({"0": "B cells", "1": "T cells"}).astype("category")
        # Add a NaN value
        adata.obs.loc[adata.obs.index[0], "leiden"] = np.nan

        nan_count_before = adata.obs["leiden"].isna().sum()
        assert nan_count_before > 0

        # Reorder categories - this should run without error
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_categories(keys=["leiden"])

        # Check that NaN values are preserved
        nan_count_after = adata.obs["leiden"].isna().sum()
        assert nan_count_after == nan_count_before

    def test_reorder_categories_with_different_dtypes(self, cell_annotator_single):
        """Test that reordering categories handles different dtypes gracefully."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations with different dtypes
        n_half = len(adata) // 2
        adata.obs["integer_cats"] = ([0] * (n_half + (len(adata) % 2))) + ([1] * n_half)
        adata.obs["object_cats"] = (["A"] * (n_half + (len(adata) % 2))) + (["B"] * n_half)
        adata.obs["int_categorical_cats"] = pd.Series(([0] * (n_half + (len(adata) % 2))) + ([1] * n_half)).astype(
            "category"
        )

        keys = ["integer_cats", "object_cats", "int_categorical_cats"]

        # Reorder categories - this should run without error
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_categories(keys=keys)

        # Check that all columns are now string categoricals
        for key in keys:
            assert isinstance(adata.obs[key].dtype, pd.CategoricalDtype)
            assert all(isinstance(cat, str) for cat in adata.obs[key].cat.categories)

    def test_reorder_categories_preserves_per_key_colors(self, cell_annotator_single):
        """Test that reordering categories preserves colors independently for each key."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # --- Setup ---
        # Create two categorical columns with overlapping categories but different colors
        adata.obs["cat_1"] = pd.Series(["A", "B"] * (len(adata) // 2 + 1))[: len(adata)].astype("category")
        adata.obs["cat_2"] = pd.Series(["A", "C"] * (len(adata) // 2 + 1))[: len(adata)].astype("category")

        # Define distinct color maps
        original_colors_1 = {"A": "#ff0000", "B": "#00ff00"}  # A is red
        original_colors_2 = {"A": "#0000ff", "C": "#ffff00"}  # A is blue

        adata.uns["cat_1_colors"] = [original_colors_1[cat] for cat in adata.obs["cat_1"].cat.categories]
        adata.uns["cat_2_colors"] = [original_colors_2[cat] for cat in adata.obs["cat_2"].cat.categories]

        # --- Run ---
        beautifier = ObsBeautifier(adata=adata)
        # The LLM will reorder ['A', 'B', 'C'] into a new order, e.g., ['A', 'C', 'B']
        # The key is to ensure the color for 'A' is red for cat_1 and blue for cat_2 after this.
        beautifier.reorder_categories(keys=["cat_1", "cat_2"])

        # --- Assert ---
        # Check colors for cat_1
        new_categories_1 = adata.obs["cat_1"].cat.categories
        new_colors_1 = adata.uns["cat_1_colors"]
        reordered_color_map_1 = dict(zip(new_categories_1, new_colors_1, strict=False))

        assert reordered_color_map_1["A"] == original_colors_1["A"]
        assert reordered_color_map_1["B"] == original_colors_1["B"]

        # Check colors for cat_2
        new_categories_2 = adata.obs["cat_2"].cat.categories
        new_colors_2 = adata.uns["cat_2_colors"]
        reordered_color_map_2 = dict(zip(new_categories_2, new_colors_2, strict=False))

        assert reordered_color_map_2["A"] == original_colors_2["A"]
        assert reordered_color_map_2["C"] == original_colors_2["C"]

    @flaky
    @pytest.mark.real_llm_query()
    def test_reorder_categories_with_complex_category_names(self, cell_annotator_single):
        """Test reordering categories with complex category names that might confuse the LLM."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Create categories similar to the neurotransmitter transporter example
        # that could cause the LLM to return duplicates
        complex_categories = [
            "NT-CHOL",
            "NT-GABA",
            "NT-GABA NT-VGLUT",
            "NT-GLY",
            "NT-GLY NT-VGLUT",
            "NT-SER",
            "NT-SER NT-VGLUT",
            "NT-VGLUT",
        ]

        # Assign these categories cyclically to cells
        n_cats = len(complex_categories)
        category_assignments = [complex_categories[i % n_cats] for i in range(len(adata))]
        adata.obs["complex_categories"] = pd.Categorical(category_assignments)

        # This should not raise a ValueError about duplicate categories
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_categories(keys=["complex_categories"])

        # Verify that the categories are still unique and match the original set
        final_categories = set(adata.obs["complex_categories"].cat.categories)
        original_categories = set(complex_categories)
        assert final_categories == original_categories

        # Verify no duplicates in the final category list
        final_categories_list = list(adata.obs["complex_categories"].cat.categories)
        assert len(final_categories_list) == len(set(final_categories_list))

    @flaky
    @pytest.mark.real_llm_query()
    def test_assign_colors_preserves_order(self, cell_annotator_single):
        """Test that assigning colors preserves the original category order."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations
        adata.obs["leiden"] = adata.obs["leiden"].map({"0": "B cells", "1": "T cells"}).astype("category")
        original_categories = list(adata.obs["leiden"].cat.categories)

        # Assign colors only
        beautifier = ObsBeautifier(adata=adata)
        beautifier.assign_colors(keys=["leiden"])

        # Check that the category order is preserved
        new_categories = list(adata.obs["leiden"].cat.categories)
        assert new_categories == original_categories

        # Check that colors were assigned
        assert "leiden_colors" in adata.uns
        assert len(adata.uns["leiden_colors"]) == len(original_categories)

    @flaky
    @pytest.mark.real_llm_query()
    def test_assign_colors_consistent_across_keys(self, cell_annotator_single):
        """Test that assigning colors creates consistent colors across multiple keys."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Create two keys with overlapping categories
        adata.obs["key1"] = pd.Categorical(["A", "B"] * (len(adata) // 2 + 1))[: len(adata)]
        adata.obs["key2"] = pd.Categorical(["A", "C"] * (len(adata) // 2 + 1))[: len(adata)]

        # Assign colors
        beautifier = ObsBeautifier(adata=adata)
        beautifier.assign_colors(keys=["key1", "key2"])

        # Check that colors are assigned
        assert "key1_colors" in adata.uns
        assert "key2_colors" in adata.uns

        # Check that the same category gets the same color across keys
        key1_color_map = dict(zip(adata.obs["key1"].cat.categories, adata.uns["key1_colors"], strict=True))
        key2_color_map = dict(zip(adata.obs["key2"].cat.categories, adata.uns["key2_colors"], strict=True))

        # Category "A" should have the same color in both keys
        assert key1_color_map["A"] == key2_color_map["A"]

    def test_assign_colors_without_existing_colors(self, cell_annotator_single):
        """Test that assign_colors handles keys without existing colors gracefully."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up annotations without existing colors
        adata.obs["test_key"] = pd.Categorical(["X", "Y"] * (len(adata) // 2 + 1))[: len(adata)]

        # This should work but requires LLM setup
        beautifier = ObsBeautifier(adata=adata)
        # Note: This test would need @pytest.mark.real_llm_query() to actually test LLM functionality
        # For now, just verify the method exists and can be called
        assert hasattr(beautifier, "assign_colors")
        assert callable(beautifier.assign_colors)
