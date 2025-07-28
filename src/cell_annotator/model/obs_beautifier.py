"""Beautify categorical observations in AnnData objects."""

from typing import cast

import pandas as pd
import scanpy as sc
from pandas.api.types import CategoricalDtype

from cell_annotator._constants import PackageConstants
from cell_annotator._docs import d
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import CellTypeColorOutput, CellTypeListOutput
from cell_annotator.check import check_deps
from cell_annotator.model.llm_interface import LLMInterface
from cell_annotator.utils import _get_consistent_ordering, _get_unique_cell_types, _validate_list_mapping


@d.dedent
class ObsBeautifier(LLMInterface):
    """
    Beautifies categorical annotations in an AnnData object.

    Uses an LLM to reorder cluster labels into a biologically meaningful
    order and assign them visually distinct and consistent colors.

    Parameters
    ----------
    %(adata)s
    %(model)s
    %(max_completion_tokens)s
    %(provider)s
    %(api_key)s
    """

    def __init__(
        self,
        adata: sc.AnnData,
        model: str | None = None,
        max_completion_tokens: int | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            max_completion_tokens=max_completion_tokens,
            provider=provider,
            api_key=api_key,
        )
        self.adata = adata

    def __repr__(self) -> str:
        """Return a string representation of the ObsBeautifier."""
        lines = []
        lines.append(f"🎨 {self.__class__.__name__}")
        lines.append("=" * (len(self.__class__.__name__) + 3))

        # Model configuration
        lines.append(f"🤖 Provider: {self._provider_name}")
        lines.append(f"🧠 Model: {self.model}")
        if self.max_completion_tokens:
            lines.append(f"🎚️ Max tokens: {self.max_completion_tokens}")

        # Status
        lines.append("")
        try:
            test_result = self.test_query()
            status = "✅ Ready" if test_result else "❌ Not working"
        except Exception as e:  # noqa: BLE001
            logger.debug("Status check failed: %s", str(e))
            status = "⚠️ Unknown"
        lines.append(f"🔋 Status: {status}")

        return "\n".join(lines)

    def _preprocess_categories(self, keys: list[str] | str) -> list[str]:
        """Preprocess categorical observations for consistency.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` to preprocess.

        Returns
        -------
        List of processed keys.
        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            # Convert to categorical if not already
            if not isinstance(self.adata.obs[key].dtype, CategoricalDtype):
                self.adata.obs[key] = self.adata.obs[key].astype("category")

            # Ensure categories are strings for processing
            if not all(isinstance(c, str) for c in self.adata.obs[key].cat.categories):
                self.adata.obs[key] = self.adata.obs[key].cat.rename_categories(
                    {cat: str(cat) for cat in self.adata.obs[key].cat.categories}
                )

        # make the naming consistent: replaces underscores with spaces
        for key in keys:
            self.adata.obs[key] = self.adata.obs[key].map(lambda x: x.replace("_", " ") if isinstance(x, str) else x)

        return keys

    def reorder_categories(self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Reorder categorical annotations using biologically meaningful ordering.

        Uses an LLM to determine biologically meaningful ordering of cell type categories
        while preserving existing colors. This method replaces underscores with spaces.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` to reorder.
        unknown_key
            Name of the unknown category.

        Returns
        -------
        Updates the following attributes:
        - `self.adata.obs[keys]` category order
        """
        keys = self._preprocess_categories(keys)

        # Take the union of all cell types across keys and re-order the list
        cell_type_list = self._get_cluster_ordering(keys, unknown_key=unknown_key)

        # Preserve existing colors for each key separately
        for key in keys:
            color_key = f"{key}_colors"
            key_colors = {}
            if color_key in self.adata.uns:
                old_categories = self.adata.obs[key].cat.categories
                old_colors = self.adata.uns[color_key]
                key_colors = dict(zip(old_categories, old_colors, strict=True))

            # Use the globally ordered cell_type_list to set category order
            ordered_cats = [cat for cat in cell_type_list if cat in self.adata.obs[key].cat.categories]
            # Add any missing categories that were in the original data but not in the global list
            for cat in self.adata.obs[key].cat.categories:
                if cat not in ordered_cats:
                    ordered_cats.append(cat)

            logger.info("Reordering categories for key '%s'", key)
            self.adata.obs[key] = self.adata.obs[key].cat.set_categories(ordered_cats)

            # Preserve existing colors in the new order
            if key_colors:
                new_colors = [key_colors.get(cat, "") for cat in self.adata.obs[key].cat.categories]
                if any(new_colors):
                    self.adata.uns[f"{key}_colors"] = new_colors

    def assign_colors(
        self,
        keys: list[str] | str,
        unknown_key: str = PackageConstants.unknown_name,
        min_color_distance: float = 10.0,
    ) -> None:
        """Assign consistent colors across cell type annotations.

        Uses an LLM to assign biologically meaningful and visually distinct colors
        while preserving existing category order. This method replaces underscores with spaces.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` to assign colors to.
        unknown_key
            Name of the unknown category.
        min_color_distance
            Minimum Delta E distance for color validation. Colors with lower distance
            will be flagged as potentially too similar. Set to 0 to disable validation.

        Returns
        -------
        Updates the following attributes:
        - `self.adata.uns[f"{key}_colors"]` for each key
        """
        keys = self._preprocess_categories(keys)

        # Get unique cell types across all keys for consistent coloring
        unique_cell_types = _get_unique_cell_types(self.adata, keys, unknown_key)
        string_cell_types = [str(cell_type) for cell_type in unique_cell_types if pd.notna(cell_type)]

        # Get global color mapping from LLM
        global_names_and_colors = self._get_cluster_colors(clusters=string_cell_types, unknown_key=unknown_key)

        # Validate color distinguishability if enabled
        if min_color_distance > 0:
            colors_only = list(global_names_and_colors.values())
            is_valid, problematic_pairs = self._validate_color_distinguishability(
                colors_only, min_delta_e=min_color_distance
            )

            if not is_valid:
                logger.warning(
                    "Found %d color pairs that may be too similar (ΔE < %.1f). "
                    "Consider requesting new colors from LLM or adjusting min_color_distance.",
                    len(problematic_pairs),
                    min_color_distance,
                )
                for color1, color2, distance in problematic_pairs:
                    logger.debug("Similar colors: %s vs %s (ΔE = %.1f)", color1, color2, distance)

        label_sets = _get_consistent_ordering(self.adata, global_names_and_colors, keys)

        # Apply colors to each key while preserving category order
        for key in keys:
            if key not in label_sets:
                continue

            name_and_color = label_sets[key]

            # Add unknown category color if present
            if unknown_key in self.adata.obs[key].cat.categories:
                logger.debug("Adding unknown category color for key '%s'", unknown_key)
                name_and_color[unknown_key] = PackageConstants.unknown_color

            # Validate that we have all categories
            _validate_list_mapping(list(self.adata.obs[key].cat.categories), list(name_and_color.keys()), context=key)

            logger.info("Assigning colors for key '%s'", key)
            # Sort colors according to current category order
            new_colors = [name_and_color.get(cat, "") for cat in self.adata.obs[key].cat.categories]

            if any(new_colors):
                self.adata.uns[f"{key}_colors"] = new_colors

    def _get_cluster_ordering(self, keys: list[str], unknown_key: str = PackageConstants.unknown_name) -> list[str]:
        """Query LLM for relational cluster ordering.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` whose categories should be ordered.
        unknown_key
            Name of the unknown category.

        Returns
        -------
        List of cell types in some biologically meaningful order.
        """
        # format the current annotations sets as a string and prepare the query prompt
        unique_cell_types = _get_unique_cell_types(self.adata, keys, unknown_key)
        # Filter out non-string types (like NaN) before joining
        string_cell_types = [str(cell_type) for cell_type in unique_cell_types if pd.notna(cell_type)]
        prompts = Prompts(species="human", tissue="cell", stage="adult")
        order_prompt = prompts.get_order_prompt(unique_cell_types=", ".join(string_cell_types))

        # query llm and format the response as a dict
        logger.info("Querying label ordering.")
        response = self.query_llm(
            instruction=order_prompt,
            response_format=CellTypeListOutput,
        )

        cell_type_list = cast(CellTypeListOutput, response).cell_type_list

        # Remove duplicates while preserving order
        seen = set()
        deduplicated_list = []
        for item in cell_type_list:
            if item not in seen:
                seen.add(item)
                deduplicated_list.append(item)

        logger.debug("Removed %d duplicate(s) from LLM response", len(cell_type_list) - len(deduplicated_list))

        return deduplicated_list

    def _get_cluster_colors(
        self, clusters: str | list[str], unknown_key: str = PackageConstants.unknown_name
    ) -> dict[str, str]:
        """Query LLM for relational cluster colors.

        Parameters
        ----------
        clusters
            Either a key in `adata.obs` or a list of cell type names.
        unknown_key
            Name of the unknown category.

        Returns
        -------
        Mapping of cell types to colors.

        """
        if isinstance(clusters, str):
            if clusters not in self.adata.obs:
                raise ValueError(f"Key '{clusters}' not found in `adata.obs`.")
            cluster_list = list(self.adata.obs[clusters].unique())
        elif isinstance(clusters, list):
            cluster_list = clusters
        else:
            raise ValueError(f"Invalid type for 'clusters': {type(clusters)}")

        cluster_names = ", ".join(str(cl) for cl in cluster_list if cl != unknown_key and pd.notna(cl))

        logger.info("Querying cluster colors.")
        prompts = Prompts(species="human", tissue="cell", stage="adult")
        response = self.query_llm(
            instruction=prompts.get_color_prompt(cluster_names), response_format=CellTypeColorOutput
        )
        response = cast(CellTypeColorOutput, response)
        color_dict = {
            item.original_cell_type_label: item.assigned_color for item in response.cell_type_to_color_mapping
        }

        # Re-add the unknown category, make sure that we retained all categories, and write to adata
        if unknown_key in cluster_list:
            logger.debug(
                "Readding the unknown category with key '%s'",
                unknown_key,
            )
            color_dict[unknown_key] = PackageConstants.unknown_color

        # make sure that the new categories are the same as the original categories
        _validate_list_mapping(cluster_list, list(color_dict.keys()))

        # fix their order in case it was changed
        color_dict = {key: color_dict[key] for key in cluster_list}

        return color_dict

    def _validate_color_distinguishability(
        self, colors: list[str], min_delta_e: float = 10.0
    ) -> tuple[bool, list[tuple[str, str, float]]]:
        """Validate that colors are sufficiently distinguishable.

        Uses Delta E (CIE2000) distance in perceptually uniform Lab color space
        to measure color differences. Requires the optional colorspacious dependency.

        Parameters
        ----------
        colors
            List of hex color codes (e.g., '#FF0000')
        min_delta_e
            Minimum Delta E distance for colors to be considered distinguishable.
            Typical thresholds:
            - ΔE < 1: Not perceptible by human eye
            - ΔE 1-3: Perceptible through close observation
            - ΔE 3-5: Perceptible at a glance
            - ΔE > 5: Colors appear different
            - ΔE > 10: Very different colors (recommended for data visualization)

        Returns
        -------
        is_valid
            Whether all color pairs meet the minimum distance threshold
        problematic_pairs
            List of (color1, color2, distance) tuples that are too similar
        """
        try:
            check_deps("colorspacious")
            from colorspacious import cspace_convert, deltaE
        except (ImportError, RuntimeError):
            logger.warning("colorspacious not available, skipping color validation")
            return True, []

        problematic_pairs = []

        def hex_to_rgb01(hex_color):
            """Convert hex color to RGB values in 0-1 range."""
            hex_color = hex_color.lstrip("#")
            return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

        for i, color1 in enumerate(colors):
            for color2 in colors[i + 1 :]:
                try:
                    # Convert hex colors to RGB values (0-1 range) for colorspacious
                    rgb1 = hex_to_rgb01(color1)
                    rgb2 = hex_to_rgb01(color2)

                    # Convert RGB to Lab space and calculate Delta E
                    lab1 = cspace_convert(rgb1, "sRGB1", "CIELab")
                    lab2 = cspace_convert(rgb2, "sRGB1", "CIELab")
                    distance = deltaE(lab1, lab2, input_space="CIELab")

                    if distance < min_delta_e:
                        problematic_pairs.append((color1, color2, distance))

                except (ValueError, TypeError) as e:
                    logger.warning("Failed to calculate color distance for %s vs %s: %s", color1, color2, str(e))

        return len(problematic_pairs) == 0, problematic_pairs
