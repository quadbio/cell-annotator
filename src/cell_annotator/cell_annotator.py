"""Cell annotator class for annotating cell types across multiple samples."""

import os

import scanpy as sc
from dotenv import load_dotenv
from scanpy.tools._rank_genes_groups import _Method
from tqdm.auto import tqdm

from cell_annotator._constants import PackageConstants, PromptExamples
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import (
    CellTypeColorOutput,
    CellTypeListOutput,
    CellTypeMappingOutput,
    ExpectedMarkerGeneOutput,
)
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
        Key in `adata.obs` indicating batch membership.
    species
        Species name (inherited from BaseAnnotator).
    tissue
        Tissue name (inherited from BaseAnnotator).
    stage
        Developmental stage (inherited from BaseAnnotator).
    cluster_key
        Key of the cluster column in adata.obs.
    model
        OpenAI model name (inherited from BaseAnnotator).
    """

    def __init__(
        self,
        adata: sc.AnnData,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        sample_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        super().__init__(species, tissue, stage, cluster_key, model)
        self.adata = adata
        self.sample_key = sample_key

        self.sample_annotators: dict[str, SampleAnnotator] = {}
        self.expected_cell_types: list[str] = []
        self.expected_marker_genes: dict[str, list[str]] | None = None
        self.annotated: bool = False
        self.cell_type_key: str | None = None

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
                max_tokens=self.max_tokens,
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
        cell_type_prompt = Prompts.CELL_TYPE_PROMPT.format(species=self.species, tissue=self.tissue, stage=self.stage)

        logger.info("Querying cell types.")
        res_types = self.query_openai(
            instruction=cell_type_prompt,
            response_format=CellTypeListOutput,
        )

        logger.info("Writing expected cell types to `self.expected_cell_types`")
        self.expected_cell_types = res_types.cell_type_list

        marker_gene_prompt = [
            {"role": "assistant", "content": "; ".join(self.expected_cell_types) if self.expected_cell_types else ""},
            {"role": "user", "content": Prompts.CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)},
        ]

        logger.info("Querying cell type markers.")
        res_markers = self.query_openai(
            instruction=cell_type_prompt,
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

        Returns
        -------
        Updates the following attributes:
        - `self.marker_dfs`
        - `self.marker_genes`

        """
        logger.info("Iterating over samples to compute cluster marker genes. ")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.get_cluster_markers(
                method=method,
                min_specificity=min_specificity,
                min_auc=min_auc,
                max_markers=max_markers,
                use_raw=use_raw,
            )

    def annotate_clusters(self, min_markers: int = 2, key_added: str = "cell_type_predicted"):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        min_markers
            Minimal number of required marker genes per cluster.
        key_added
            Name of the key in .obs where updated annotations will be written

        Returns
        -------
        Updates the following attributes:
        - `self.annotation_df`
        - `self.adata.obs[key_added]`
        - `self.annotated`
        - `self.cell_type_key`

        """
        if self.expected_marker_genes is None:
            logger.debug(
                "Querying expected cell type markers wiht default parameters. Run `get_expected_cell_type_markers` for more control. "
            )
            self.get_expected_cell_type_markers()

        logger.info("Iterating over samples to annotate clusters. ")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.annotate_clusters(min_markers=min_markers, expected_marker_genes=self.expected_marker_genes)

        # set the annotated flag to True
        self.annotated = True

        # harmonize annotations across samples
        try:
            self._harmonize_annotations()
            self.cell_type_key = "cell_type_harmonized"
        except ValueError as e:
            logger.warning("Error during annotation harmonization: %s. Skipping.", e)
            self.cell_type_key = "cell_type"

        # write the annotatation results back to self.adata
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
        """Harmonize annotations across samples."""
        if not self.annotated:
            raise ValueError("No annotations found. Run `annotate_clusters` first.")

        # Step 1: get a list of all unique cell types
        cell_types = set()
        for annotator in self.sample_annotators.values():
            categories = annotator.annotation_df["cell_type"].unique()
            cell_types.update(cat for cat in categories if cat != unknown_key)

        deduplication_prompt = Prompts.DUPLICATE_REMOVAL_PROMPT.format(list_with_duplicates=", ".join(cell_types))

        # query openai
        logger.info("Querying cell-type label de-duplication.")
        response = self.query_openai(
            instruction=deduplication_prompt,
            response_format=CellTypeListOutput,
        )
        global_cell_type_list = response.cell_type_list
        logger.info("Removed %s/%s cell types.", len(cell_types) - len(global_cell_type_list), len(cell_types))

        # Step 2: map cell types to harmonized names for each sample annotator
        logger.info("Iterating over samples to harmonize cell type annotations.")
        for sample, annotator in tqdm(self.sample_annotators.items()):
            local_cell_types = [cat for cat in annotator.annotation_df["cell_type"].unique() if cat != unknown_key]

            mapping_prompt = [
                {"role": "assistant", "content": "; ".join(global_cell_type_list)},
                {
                    "role": "user",
                    "content": Prompts.MAPPING_PROMPT.format(
                        cell_type_list=", ".join(local_cell_types),
                    ),
                },
            ]

            response = self.query_openai(
                instruction=deduplication_prompt,
                other_messages=mapping_prompt,
                response_format=CellTypeMappingOutput,
            )

            # Convert to dictionary
            cell_type_mapping_dict = {
                mapping.original_name: mapping.unique_name for mapping in response.cell_type_mapping
            }

            # validate keys and values in the mapping
            _validate_list_mapping(local_cell_types, cell_type_mapping_dict.keys(), context=sample)

            # Check if all values in cell_type_mapping_dict are in the global_cell_type_set
            missing_cell_types = [
                value for value in cell_type_mapping_dict.values() if value not in global_cell_type_list
            ]
            if missing_cell_types:
                raise ValueError(
                    f"For sample {sample}, some cell types were not found in the global list: {missing_cell_types}"
                )

            # Re-add the unkonwn category if it was present originally
            original_categories = annotator.annotation_df["cell_type"].unique()
            if unknown_key in original_categories:
                cell_type_mapping_dict[unknown_key] = unknown_key

            # Introduce a new column "cell_type_harmonized" in annotator.annotation_df
            annotator.annotation_df["cell_type_harmonized"] = annotator.annotation_df["cell_type"].map(
                cell_type_mapping_dict
            )

            # Check for any unmapped cell types and raise an error if found
            unmapped_cell_types = annotator.annotation_df["cell_type_harmonized"].isna()
            if unmapped_cell_types.any():
                raise ValueError(
                    f"For sample {sample}, some cell types were not mapped: {annotator.annotation_df['cell_type'][unmapped_cell_types].unique()}"
                )

    def reorder_clusters(self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Assign consistent ordering across cell type annotations.

        Note that for multiple samples with many clusters each, this typically requires a more powerful model
        like `gpt-4o` to work well. This method replaces underscores with spaces.

        Parameters
        ----------
        keys
            List of keys in `adata.obs` to reorder.
        unknown_key
            Name of the unknown category.

        Returns
        -------
        Updated the following attributes:
        - `self.adata.obs[keys]`
        """
        if isinstance(keys, str):
            keys = [keys]

        # make the naming consistent: replaces underscores with spaces
        for key in keys:
            self.adata.obs[key] = self.adata.obs[key].map(lambda x: x.replace("_", " "))

        # format the current annotations sets as a string and prepare the query prompt
        unique_cell_types = _get_unique_cell_types(self.adata, keys, unknown_key)
        order_prompt = Prompts.ORDER_PROMPT.format(
            unique_cell_types=", ".join(unique_cell_types),
            example_unordered=PromptExamples.unordered_cell_types,
            example_ordered=PromptExamples.ordered_cell_types,
        )

        # query openai and format the response as a dict
        logger.info("Querying label ordering.")
        response = self.query_openai(
            instruction=order_prompt,
            response_format=CellTypeListOutput,
        )
        label_sets = _get_consistent_ordering(self.adata, response.cell_type_list, keys)

        # Re-add the unknown category, make sure that we retained all categories, and write to adata
        for obs_key, new_cluster_names in label_sets.items():
            if unknown_key in self.adata.obs[obs_key].cat.categories:
                logger.debug(
                    "Readding the unknown category with key '%s'",
                    unknown_key,
                )
                new_cluster_names.append(unknown_key)

            original_categories = self.adata.obs[obs_key].unique()
            _validate_list_mapping(original_categories, new_cluster_names, context=obs_key)

            logger.info("Writing categories for key '%s'", obs_key)
            self.adata.obs[obs_key] = self.adata.obs[obs_key].cat.set_categories(new_cluster_names)

    def get_cluster_colors(self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Query OpenAI for relational cluster colors.

        Parameters
        ----------
        keys
            Keys in `adata.obs` to query colors for.
        unknown_key
            Name of the unknown category.

        Returns
        -------
        Updates the following attributes:
        - `self.adata.uns[f"{keys}_colors"]`

        """
        if isinstance(keys, str):
            keys = [keys]

        logger.info("Iterating over obs keys")
        for key in tqdm(keys):
            # format the cluster names as a string and prepare the query prompt
            cluster_names = ", ".join(cl for cl in self.adata.obs[key].cat.categories if cl != unknown_key)
            color_prompt = Prompts.COLOR_PROMPT.format(cluster_names=cluster_names)

            response = self.query_openai(instruction=color_prompt, response_format=CellTypeColorOutput)
            color_dict = {
                item.original_cell_type_label: item.assigned_color for item in response.cell_type_to_color_mapping
            }

            # Re-add the unknown category, make sure that we retained all categories, and write to adata
            if unknown_key in self.adata.obs[key].cat.categories:
                logger.debug(
                    "Readding the unknown category with key '%s'",
                    unknown_key,
                )
                color_dict[unknown_key] = PackageConstants.unknown_color

            if list(self.adata.obs[key].cat.categories) != list(color_dict.keys()):
                raise ValueError(f"New categories for key {key} differ from original categories.")
            self.adata.uns[f"{key}_colors"] = color_dict.values()
