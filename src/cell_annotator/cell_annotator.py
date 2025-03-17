"""Cell annotator class for annotating cell types across multiple samples."""

import os

import scanpy as sc
from dotenv import load_dotenv
from scanpy.tools._rank_genes_groups import _Method
from tqdm.auto import tqdm

from cell_annotator._constants import PackageConstants
from cell_annotator._logging import logger
from cell_annotator._response_formats import CellTypeColorOutput, CellTypeListOutput, ExpectedMarkerGeneOutput
from cell_annotator.base_annotator import BaseAnnotator
from cell_annotator.sample_annotator import SampleAnnotator
from cell_annotator.utils import (
    _format_annotation,
    _get_consistent_ordering,
    _get_unique_cell_types,
    _try_sorting_dict_by_keys,
    _validate_list_mapping,
)


class CellAnnotator(BaseAnnotator):
    """
    Main class for annotating cell types, including handling multiple samples.

    Parameters
    ----------
    adata
        Full AnnData object with multiple samples.
    sample_key
        Key in :attr:`~anndata.AnnData.obs` indicating batch membership.
    species
        Species name.
    tissue
        Tissue name.
    stage
        Developmental stage.
    cluster_key
        Key of the cluster column in adata.obs.
    model
        OpenAI model name.
    max_completion_tokens
        Maximum number of tokens for OpenAI queries.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = PackageConstants.default_cluster_key,
        sample_key: str | None = None,
        model: str = PackageConstants.default_model,
        max_completion_tokens: int | None = None,
    ):
        super().__init__(species, tissue, stage, cluster_key, model, max_completion_tokens)
        self.adata = adata
        self.sample_key = sample_key

        self.sample_annotators: dict[str, SampleAnnotator] = {}
        self.expected_cell_types: list[str] = []
        self.expected_marker_genes: dict[str, list[str]] | None = None
        self.annotated: bool = False
        self.cell_type_key: str = PackageConstants.cell_type_key
        self.global_cell_type_list: list[str] | None = None

        # laod environmental variables
        load_dotenv()

        # Check if the environment variable OPENAI_API_KEY is set
        if os.getenv("OPENAI_API_KEY"):
            logger.info("The environment variable `OPENAI_API_KEY` is set (that's good).")
        else:
            logger.warning(
                "The environment variable `OPENAI_API_KEY` is not set. Head over to https://platform.openai.com/api-keys to get a key and store it in an .env file."
            )

        # Initialize SampleAnnotators for each batch
        self._initialize_sample_annotators()

    def __repr__(self):
        sample_summary = ", ".join(f"'{sample_id}'" for sample_id in self.sample_annotators)
        return (
            f"CellAnnotator(model={self.model!r}, species={self.species!r}, "
            f"tissue={self.tissue!r}, stage={self.stage!r}, cluster_key={self.cluster_key!r}, "
            f"sample_key={self.sample_key!r})\n"
            f"with `{len(self.sample_annotators)!r}` sample(s) in `.sample_annotators`: {sample_summary}"
        )

    def _initialize_sample_annotators(self) -> None:
        """Create a SampleAnnotator for each batch."""
        # If sample_key is None, treat the entire dataset as a single batch
        if self.sample_key is None:
            logger.info("Batch key is None. Treating the entire dataset as a single sample.")
            self.adata.obs["pseudo_batch"] = "single_sample"  # Add a pseudo-batch column
            self.sample_key = "pseudo_batch"

        self.sample_annotators = {}
        samples = self.adata.obs[self.sample_key].unique()
        logger.info("Initializing `%s` SampleAnnotator objects(s).", len(samples))
        for sample_id in samples:
            logger.debug("Initializing SampleAnnotator for sample '%s'.", sample_id)
            batch_adata = self.adata[self.adata.obs[self.sample_key] == sample_id].copy()
            self.sample_annotators[sample_id] = SampleAnnotator(
                adata=batch_adata,
                sample_name=sample_id,
                species=self.species,
                tissue=self.tissue,
                stage=self.stage,
                cluster_key=self.cluster_key,
                model=self.model,
                max_completion_tokens=self.max_completion_tokens,
            )

        # sort by keys for visual pleasure
        self.sample_annotators = _try_sorting_dict_by_keys(self.sample_annotators)

    def get_expected_cell_type_markers(self, n_markers: int = 5) -> None:
        """Get expected cell types and marker genes.

        Parameters
        ----------
        n_markers
            Number of marker genes per cell type.

        Returns
        -------
        Updates the following attributes:
        - `self.expected_cell_types`
        - `self.expected_marker_genes`
        """
        logger.info("Querying cell types.")
        res_types = self.query_openai(
            instruction=self.prompts.get_cell_type_prompt(),
            response_format=CellTypeListOutput,
        )

        logger.info("Writing expected cell types to `self.expected_cell_types`")
        self.expected_cell_types = res_types.cell_type_list

        marker_gene_prompt = [
            {"role": "assistant", "content": "; ".join(self.expected_cell_types) if self.expected_cell_types else ""},
            {"role": "user", "content": self.prompts.get_cell_type_marker_prompt(n_markers)},
        ]

        logger.info("Querying cell type markers.")
        res_markers = self.query_openai(
            instruction=self.prompts.get_cell_type_prompt(),
            other_messages=marker_gene_prompt,
            response_format=ExpectedMarkerGeneOutput,
        )

        logger.info("Writing expected marker genes to `self.expected_marker_genes`.")
        self.expected_marker_genes = {
            cell_type_markers.cell_type_name: cell_type_markers.expected_marker_genes
            for cell_type_markers in res_markers.expected_markers_per_cell_type
        }

    def get_cluster_markers(
        self,
        method: _Method | None = "wilcoxon",
        min_specificity: float = 0.75,
        min_auc: float = 0.7,
        max_markers: int = 7,
        use_raw: bool = PackageConstants.use_raw,
        use_rapids: bool = False,
    ) -> None:
        """Get marker genes per cluster

        Parameters
        ----------
        method
            Method for `sc.tl.rank_genes_groups`
        min_specificity
            Minimum specificity
        min_auc
            Minimum AUC
        max_markers
            Maximum number of markers
        use_raw
            Use raw data
        use_rapids
            Whether to use rapids for GPU acceleration

        Returns
        -------
        Updates the following attributes:
        - `self.marker_dfs`
        - `self.marker_genes`

        """
        logger.info("Iterating over samples to compute cluster marker genes. ")

        if use_rapids and method != "logreg":
            logger.warning(
                "Rapids acceleration is only available for method `logreg`. Running `rank_genes_groups` on CPU instead (AUC computation will still be GPU accelerated. )"
            )
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.get_cluster_markers(
                method=method,
                min_specificity=min_specificity,
                min_auc=min_auc,
                max_markers=max_markers,
                use_raw=use_raw,
                use_rapids=use_rapids,
            )

    def annotate_clusters(
        self, min_markers: int = 2, restrict_to_expected: bool = False, key_added: str = "cell_type_predicted"
    ):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        min_markers
            Minimal number of required marker genes per cluster.
        key_added
            Name of the key in .obs where updated annotations will be written
        restrict_to_expected
            If True, only use expected cell types for annotation.

        Returns
        -------
        Updates the following attributes:
        - `self.annotation_df`
        - `self.adata.obs[key_added]`
        - `self.annotated`

        """
        if self.expected_marker_genes is None:
            logger.debug(
                "Querying expected cell type markers with default parameters. Run `get_expected_cell_type_markers` for more control. "
            )
            self.get_expected_cell_type_markers()

        logger.info("Iterating over samples to annotate clusters. ")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.annotate_clusters(
                min_markers=min_markers,
                restrict_to_expected=restrict_to_expected,
                expected_marker_genes=self.expected_marker_genes,
            )

        # set the annotated flag to True
        self.annotated = True

        # harmonize annotations across samples if necessary
        self._harmonize_annotations()

        # write the annotation results back to self.adata
        self._update_adata_annotations(key_added=key_added)

        return self

    def _update_adata_annotations(self, key_added: str) -> None:
        """Update cluster labels in adata object."""
        if not self.annotated:
            raise ValueError("No annotations found. Run `annotate_clusters` first.")

        logger.info("Writing updated cluster labels to `adata.obs[`%s'].", key_added)
        self.adata.obs[key_added] = None

        for sample, annotator in self.sample_annotators.items():
            mask = self.adata.obs[self.sample_key] == sample
            label_mapping = annotator.annotation_df[self.cell_type_key].to_dict()
            self.adata.obs.loc[mask, key_added] = self.adata.obs.loc[mask, self.cluster_key].map(label_mapping)

        self.adata.obs[key_added] = self.adata.obs[key_added].astype("category")

    def _get_annotation_summary_string(self, filter_by: str = "") -> str:
        if not self.annotated:
            raise ValueError("No annotations found. Run `annotate_clusters` first.")

        summary_string = "\n\n".join(
            f"Sample: {key}\n{_format_annotation(annotator.annotation_df, filter_by, self.cell_type_key)}"
            for key, annotator in self.sample_annotators.items()
        )

        return summary_string

    def _harmonize_annotations(self, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Harmonize annotations across samples.

        Parameters
        ----------
        unknown_key
            Name of the unknown category.

        Returns
        -------
        Updates the following attributes:
        - `self.global_cell_type_list`
        - `self.sample_annotators[sample].annotation_df['cell_type_harmonized']`

        """
        if not self.annotated:
            raise ValueError("No annotations found. Run `annotate_clusters` first.")

        # Step 1: get a list of all unique cell types
        cell_types = set()
        for annotator in self.sample_annotators.values():
            categories = annotator.annotation_df["cell_type"].unique()
            cell_types.update(cat for cat in categories if cat != unknown_key)

        deduplication_prompt = self.prompts.get_duplicate_removal_prompt(list_with_duplicates=", ".join(cell_types))

        # query openai
        logger.info("Querying cell-type label de-duplication.")
        response = self.query_openai(
            instruction=deduplication_prompt,
            response_format=CellTypeListOutput,
        )
        self.global_cell_type_list = response.cell_type_list
        logger.info("Removed %s/%s cell types.", len(cell_types) - len(self.global_cell_type_list), len(cell_types))

        # Step 2: map cell types to harmonized names for each sample annotator
        logger.info("Iterating over samples to harmonize cell type annotations.")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.harmonize_annotations(self.global_cell_type_list, unknown_key=unknown_key)

    def reorder_and_color_clusters(
        self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name, assign_colors: bool = False
    ) -> None:
        """Assign consistent ordering across cell type annotations.

        Note that for multiple samples with many clusters each, this typically requires a more powerful model
        like `gpt-4o` to work well. This method replaces underscores with spaces.

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

        # make the naming consistent: replaces underscores with spaces
        for key in keys:
            self.adata.obs[key] = self.adata.obs[key].map(lambda x: x.replace("_", " "))

        # Take the union of all cell types across keys and re-order the list
        cell_type_list = self._get_cluster_ordering(keys, unknown_key=unknown_key)

        # assign meaningful colors to the reordered list of cell types
        if assign_colors:
            global_names_and_colors = self._get_cluster_colors(clusters=cell_type_list, unknown_key=unknown_key)
        else:
            global_names_and_colors = dict.fromkeys(cell_type_list, "")

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
            if assign_colors:
                self.adata.uns[f"{obs_key}_colors"] = name_and_color.values()

    def _get_cluster_ordering(self, keys: list[str], unknown_key: str = PackageConstants.unknown_name) -> list[str]:
        """Query OpenAI for relational cluster ordering.

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
        order_prompt = self.prompts.get_order_prompt(unique_cell_types=", ".join(unique_cell_types))

        # query openai and format the response as a dict
        logger.info("Querying label ordering.")
        response = self.query_openai(
            instruction=order_prompt,
            response_format=CellTypeListOutput,
        )

        return response.cell_type_list

    def _get_cluster_colors(
        self, clusters: str | list[str], unknown_key: str = PackageConstants.unknown_name
    ) -> dict[str, str]:
        """Query OpenAI for relational cluster colors.

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

        cluster_names = ", ".join(cl for cl in cluster_list if cl != unknown_key)

        logger.info("Querying cluster colors.")
        response = self.query_openai(
            instruction=self.prompts.get_color_prompt(cluster_names), response_format=CellTypeColorOutput
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
