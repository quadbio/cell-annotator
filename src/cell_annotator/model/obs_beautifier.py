"""Beautify categorical observations in AnnData objects."""

import pandas as pd
import scanpy as sc
from pandas.api.types import CategoricalDtype

from cell_annotator._constants import PackageConstants
from cell_annotator._docs import d
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import CellTypeColorOutput, CellTypeListOutput
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

    @d.dedent
    def reorder_and_color(
        self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name, assign_colors: bool = False
    ) -> None:
        """Assign consistent ordering across cell type annotations.

        Note that for multiple samples with many clusters each, this typically requires a more powerful model
        to work well. This method replaces underscores with spaces.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` to reorder.
        unknown_key
            Name of the unknown category.
        assign_colors
            Assign colors to the cell types across keys. These are supposed to be consistent across keys and meaningful bioligically,
            such that similar cell types get similar colors.

        Returns
        -------
        Updates the following attributes:
        - `self.adata.obs[keys]`
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

        # Take the union of all cell types across keys and re-order the list
        cell_type_list = self._get_cluster_ordering(keys, unknown_key=unknown_key)

        # assign meaningful colors to the reordered list of cell types
        if assign_colors:
            global_names_and_colors = self._get_cluster_colors(clusters=cell_type_list, unknown_key=unknown_key)
        else:
            # If not assigning new colors, try to preserve existing ones.
            # Start with all cell types mapped to empty strings
            global_names_and_colors = dict.fromkeys(cell_type_list, "")
            # Collect existing colors from all keys and update
            for key in keys:
                color_key = f"{key}_colors"
                if color_key in self.adata.uns:
                    old_categories = self.adata.obs[key].cat.categories
                    old_colors = self.adata.uns[color_key]
                    for cat, color in zip(old_categories, old_colors, strict=True):
                        if cat in global_names_and_colors:
                            global_names_and_colors[cat] = color

        label_sets = _get_consistent_ordering(self.adata, global_names_and_colors, keys)

        # Re-add the unknown category, make sure that we retained all categories, and write to adata
        for obs_key, name_and_color in label_sets.items():
            if unknown_key in self.adata.obs[obs_key].cat.categories:
                logger.debug(
                    "Readding the unknown category with key '%s'",
                    unknown_key,
                )
                name_and_color[unknown_key] = PackageConstants.unknown_color

            _validate_list_mapping(list(self.adata.obs[obs_key].unique()), list(name_and_color.keys()), context=obs_key)

            logger.info("Writing categories for key '%s'", obs_key)
            self.adata.obs[obs_key] = self.adata.obs[obs_key].cat.set_categories(list(name_and_color.keys()))
            new_colors = list(name_and_color.values())
            # Only write colors if there are any. This handles both assign_colors=True and preserving old colors.
            if any(new_colors):
                self.adata.uns[f"{obs_key}_colors"] = new_colors

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

        return response.cell_type_list

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
