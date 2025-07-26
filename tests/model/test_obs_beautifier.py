import numpy as np
import pandas as pd
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

    @pytest.mark.real_llm_query()
    def test_reorder_only_preserves_colors(self, cell_annotator_single):
        """Test that reordering clusters without assigning new colors preserves the original colors."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations and colors
        adata.obs["leiden"] = adata.obs["leiden"].map({"0": "B cells", "1": "T cells"}).astype("category")
        original_colors = {"B cells": "#ff0000", "T cells": "#00ff00"}
        adata.uns["leiden_colors"] = [original_colors[cat] for cat in adata.obs["leiden"].cat.categories]

        # Reorder clusters
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_and_color(keys=["leiden"], assign_colors=False)

        # Check that the colors are preserved and reordered correctly
        new_categories = adata.obs["leiden"].cat.categories
        new_colors = adata.uns["leiden_colors"]
        reordered_color_map = dict(zip(new_categories, new_colors, strict=False))

        # The new order might be different, but the color for each cell type should be the same
        assert reordered_color_map["B cells"] == original_colors["B cells"]
        assert reordered_color_map["T cells"] == original_colors["T cells"]
        assert len(new_colors) == 2

    def test_reorder_with_nan_values(self, cell_annotator_single):
        """Test that reordering handles NaN values gracefully."""

        cell_annotator = cell_annotator_single
        adata = cell_annotator.adata

        # Set up initial annotations with a NaN value
        adata.obs["leiden"] = adata.obs["leiden"].map({"0": "B cells", "1": "T cells"}).astype("category")
        # Add a NaN value
        adata.obs.loc[adata.obs.index[0], "leiden"] = np.nan

        nan_count_before = adata.obs["leiden"].isna().sum()
        assert nan_count_before > 0

        # Reorder clusters - this should run without error
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_and_color(keys=["leiden"], assign_colors=False)

        # Check that NaN values are preserved
        nan_count_after = adata.obs["leiden"].isna().sum()
        assert nan_count_after == nan_count_before

    def test_reorder_with_different_dtypes(self, cell_annotator_single):
        """Test that reordering handles different dtypes gracefully."""

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

        # Reorder clusters - this should run without error
        beautifier = ObsBeautifier(adata=adata)
        beautifier.reorder_and_color(keys=keys, assign_colors=False)

        # Check that all columns are now string categoricals
        for key in keys:
            assert isinstance(adata.obs[key].dtype, pd.CategoricalDtype)
            assert all(isinstance(cat, str) for cat in adata.obs[key].cat.categories)
