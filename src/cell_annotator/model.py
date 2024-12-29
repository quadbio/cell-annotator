import os

import numpy as np
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv
from pandas import DataFrame
from scanpy.tools._rank_genes_groups import _Method
from tqdm.auto import tqdm

from cell_annotator._constants import PackageConstants, PromptExamples
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import (
    CellTypeColorOutput,
    ExpectedCellTypeOutput,
    ExpectedMarkerGeneOutput,
    LabelOrderOutput,
    PredictedCellTypeOutput,
)
from cell_annotator.utils import (
    ResponseOutput,
    _filter_by_category_size,
    _format_annotation,
    _get_auc,
    _get_consistent_ordering,
    _get_specificity,
    _get_unique_cell_types,
    _query_openai,
    _try_sorting_dict_by_keys,
)


class BaseAnnotator:
    """
    Shared base class for annotation-related functionality.

    Parameters
    ----------
    species : str
        Species name.
    tissue : str
        Tissue name.
    stage : str, optional
        Developmental stage. Default is 'adult'.
    cluster_key : str, optional
        Key of the cluster column in adata.obs. Default is 'leiden'.
    model : str, optional
        OpenAI model name. Default is 'gpt-4o-mini'.
    max_tokens : int, optional
        Maximum number of tokens the model is allowed to use.
    """

    def __init__(
        self,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        model: str = "gpt-4o-mini",
        max_tokens: int | None = None,
    ):
        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.model = model
        self.max_tokens = max_tokens

    def _query_openai(
        self,
        instruction: str,
        response_format: ResponseOutput,
        other_messages: list | None = None,
    ) -> ResponseOutput:
        agent_description = Prompts.AGENT_DESCRIPTION.format(species=self.species)

        response = _query_openai(
            agent_description=agent_description,
            instruction=instruction,
            model=self.model,
            response_format=response_format,
            other_messages=other_messages,
            max_tokens=self.max_tokens,
        )

        return response


class SampleAnnotator(BaseAnnotator):
    """
    Handles annotation for a single batch/sample.

    Parameters
    ----------
    adata : sc.AnnData
        Subset of the main AnnData object corresponding to a single batch.
    species : str
        Species name (inherited from BaseAnnotator).
    tissue : str
        Tissue name (inherited from BaseAnnotator).
    stage : str, optional
        Developmental stage (inherited from BaseAnnotator).
    expected_marker_genes : dict[str, list[str]], optional
        Precomputed dict, mapping expected cell types to marker genes.
    cluster_key : str, optional
        Key of the cluster column in adata.obs (inherited from BaseAnnotator).
    model : str, optional
        OpenAI model name (inherited from BaseAnnotator).
    max_tokens : int, optional
        Maximum number of tokens the model is allowed to use.

    """

    def __init__(
        self,
        adata: sc.AnnData,
        sample_name: str,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        model: str = "gpt-4o-mini",
        max_tokens: int | None = None,
    ):
        super().__init__(species, tissue, stage, cluster_key, model, max_tokens)
        self.adata = adata
        self.sample_name = sample_name

        self.annotation_df: pd.DataFrame | None = None
        self.marker_gene_dfs: dict[str, pd.DataFrame] | None = None
        self.marker_genes: dict[str, list[str]] = {}
        self.annotation_dict: dict[str, ResponseOutput] = {}

        # compute the number of cells per cluster
        self.n_cells_per_cluster = _try_sorting_dict_by_keys(self.adata.obs[self.cluster_key].value_counts().to_dict())

    def __repr__(self):
        return f"SampleAnnotator(sample_name={self.sample_name!r}, n_clusters={self.adata.obs[self.cluster_key].nunique()}, n_cells={self.adata.n_obs:,})"

    def get_cluster_markers(
        self,
        method: _Method | None = "wilcoxon",
        min_cells_per_cluster: int = 3,
        min_specificity: float = 0.75,
        min_auc: float = 0.7,
        max_markers: int = 7,
        use_raw: bool = PackageConstants.use_raw,
    ) -> None:
        """Get marker genes per cluster

        Parameters
        ----------
        method : str
            Method for `sc.tl.rank_genes_groups`
        min_cells_per_cluster : int
            Include only clusters with at least this many cells.
        min_specificity : float
            Minimum specificity
        min_auc : float
            Minimum AUC
        max_markers : int
            Maximum number of markers
        use_raw : bool
            Use raw data

        Returns
        -------
        Nothing, sets `self.marker_dfs` and `self.marker_genes`.

        """
        # filter out very small clusters
        self._filter_clusters_by_cell_number(min_cells_per_cluster)

        logger.debug("Computing marker genes per cluster using method `%s`.", method)
        sc.tl.rank_genes_groups(
            self.adata, groupby=self.cluster_key, method=method, use_raw=use_raw, n_genes=PackageConstants.max_markers
        )

        marker_dfs = {}
        logger.debug("Iterating over clusters to compute specificity and AUC values.")
        for cli in self.adata.obs[self.cluster_key].unique():
            logger.debug("Computing specificity for cluster %s", cli)

            # get a list of differentially expressed genes
            genes = np.array(self.adata.uns["rank_genes_groups"]["names"][cli])

            # compute their specificity
            clust_mask = self.adata.obs[self.cluster_key] == cli
            specificity = _get_specificity(genes=genes, clust_mask=clust_mask, adata=self.adata, use_raw=use_raw)

            # filter genes by specificity
            mask = specificity >= min(min_specificity, sorted(specificity)[-PackageConstants.min_markers])
            genes, specificity = genes[mask], specificity[mask]

            # compute AUCs
            logger.debug("Computing AUC for cluster %s", cli)
            auc = _get_auc(genes=genes, clust_mask=clust_mask, adata=self.adata, use_raw=use_raw)
            marker_dfs[cli] = DataFrame({"gene": genes, "specificity": specificity, "auc": auc})

        logger.debug(
            "Writing marker gene DataFrames to `self.sample_annotators['%s'].marker_gene_dfs`.", self.sample_name
        )
        self.marker_gene_dfs = marker_dfs

        # filter to the top markers
        logger.debug("Writing top marker genes to `self.sample_annotators['%s'].marker_genes`.", self.sample_name)
        self.marker_genes = self._filter_cluster_markers(min_auc=min_auc, max_markers=max_markers)

    def _filter_clusters_by_cell_number(self, min_cells_per_cluster: int) -> None:
        removed_info = _filter_by_category_size(self.adata, column=self.cluster_key, min_size=min_cells_per_cluster)
        if removed_info:
            for cat, size in removed_info.items():
                failure_reason = f"Not enough cells for cluster {cat} in sample `{self.sample_name}` ({size}<{min_cells_per_cluster})."
                logger.warning(failure_reason)
                self.annotation_dict[cat] = PredictedCellTypeOutput.default_failure(failure_reason=failure_reason)

    def _filter_cluster_markers(self, min_auc: float, max_markers: int) -> dict[str, list[str]]:
        """Get top markers

        Parameters
        ----------
        min_auc : float
            Minimum AUC
        max_markers : int
            Maximum number of markers

        Returns
        -------
        dict[str, list[str]]
            Top marker genes per cluster.

        """
        if self.marker_gene_dfs is None:
            raise ValueError("Run `get_markers` first to compute marker genes per cluster.")

        marker_genes = {}
        for cluster, df in self.marker_gene_dfs.items():
            top_genes = df[df.auc > min_auc].sort_values("auc", ascending=False).head(max_markers).gene.values
            marker_genes[cluster] = list(top_genes)

        return _try_sorting_dict_by_keys(marker_genes)

    def annotate_clusters(self, min_markers: int, expected_marker_genes: dict[str, list[str]] | None) -> None:
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        max_tokens : int
            Maximum number of tokens for OpenAI API.
        min_markers : int
            Minimum number of requires marker genes per cluster.
        expected_marker_genes :

        Returns
        -------
        Nothing, writes annotation results to `self.annotation_df`.

        """
        if not self.marker_genes:
            logger.debug(
                "Computing cluster marker genes using default parameters. Run `get_cluster_markers` for more control. "
            )
            self.get_cluster_markers()

        # parse expected markers into a string
        if expected_marker_genes:
            expected_markers_string = "\n".join(
                [f"{cell_type}: {', '.join(genes)}" for cell_type, genes in expected_marker_genes.items()]
            )
        else:
            expected_markers_string = ""

        actual_markers_all = "\n".join([f'- Cluster {i}: {", ".join(gs)}' for i, gs in self.marker_genes.items()])

        # loop over clusters to annotate
        logger.debug("Iterating over clusters to annotate.")
        for cluster in self.marker_genes:
            actual_markers_cluster = self.marker_genes[cluster]

            if len(actual_markers_cluster) < min_markers:
                failure_reason = f"Too few markers provided for cluster {cluster} in sample `{self.sample_name}` ({len(actual_markers_cluster)}<{min_markers})."
                logger.warning(failure_reason)
                self.annotation_dict[cluster] = PredictedCellTypeOutput.default_failure(failure_reason=failure_reason)
            else:
                actual_markers_cluster_string = ", ".join(actual_markers_cluster)

                # fill in the annotation prompt
                annotation_prompt = Prompts.ANNOTATION_PROMPT.format(
                    species=self.species,
                    tissue=self.tissue,
                    stage=self.stage,
                    actual_markers_all=actual_markers_all,
                    cluster_id=cluster,
                    actual_markers_cluster=actual_markers_cluster_string,
                    expected_markers=expected_markers_string,
                )

                self.annotation_dict[cluster] = self._query_openai(
                    instruction=annotation_prompt,
                    response_format=PredictedCellTypeOutput,
                )

        logger.debug("Writing annotation results to `self.sample_annotators['%s'].annotation_df`.", self.sample_name)
        self.annotation_df = DataFrame.from_dict(
            {k: v.model_dump() for k, v in _try_sorting_dict_by_keys(self.annotation_dict).items()}, orient="index"
        )

        # add the marker genes we used
        self.annotation_df.insert(
            0, "marker_genes", {key: ", ".join(value) for key, value in self.marker_genes.items()}
        )

        # add the number of cells per cluster
        self.annotation_df.insert(0, "n_cells", self.n_cells_per_cluster)


class CellAnnotator(BaseAnnotator):
    """
    Main class for annotating cell types, including handling multiple samples.

    Parameters
    ----------
    adata : sc.AnnData
        Full AnnData object with multiple samples.
    sample_key : str
        Key in `adata.obs` indicating batch membership.
    species : str
        Species name (inherited from BaseAnnotator).
    tissue : str
        Tissue name (inherited from BaseAnnotator).
    stage : str, optional
        Developmental stage (inherited from BaseAnnotator).
    cluster_key : str, optional
        Key of the cluster column in adata.obs. Default is 'leiden'.
    model : str, optional
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
        n_markers : int
            Number of marker genes per cell type.

        Returns
        -------
        Nothing, sets `self.expected_cell_types` and `self.expected_marker_genes`.
        """
        cell_type_prompt = Prompts.CELL_TYPE_PROMPT.format(species=self.species, tissue=self.tissue, stage=self.stage)

        logger.info("Querying cell types.")
        res_types = self._query_openai(
            instruction=cell_type_prompt,
            response_format=ExpectedCellTypeOutput,
        )

        logger.info("Writing expected cell types to `self.expected_cell_types`")
        self.expected_cell_types = res_types.expected_cell_types

        marker_gene_prompt = [
            {"role": "assistant", "content": "; ".join(self.expected_cell_types) if self.expected_cell_types else ""},
            {"role": "user", "content": Prompts.CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)},
        ]

        logger.info("Querying cell type markers.")
        res_markers = self._query_openai(
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
        method : str
            Method for `sc.tl.rank_genes_groups`
        min_specificity : float
            Minimum specificity
        min_auc : float
            Minimum AUC
        max_markers : int
            Maximum number of markers
        use_raw : bool
            Use raw data

        Returns
        -------
        Nothing, sets `self.marker_dfs`.

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
        min_markers: int, optional
            Minimal number of required marker genes per cluster.
        key_added: str, optional
            Name of the key in .obs where updated annotations will be written

        Returns
        -------
        Nothing, writes annotation results to `self.annotation_df` and annotations to `self.adata.obs[key_added]`

        """
        if self.expected_marker_genes is None:
            logger.debug(
                "Querying expected cell type markers wiht default parameters. Run `get_expected_cell_type_markers` for more control. "
            )
            self.get_expected_cell_type_markers()

        logger.info("Iterating over samples to annotate clusters. ")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.annotate_clusters(min_markers=min_markers, expected_marker_genes=self.expected_marker_genes)

        # write the annotatation results back to self.adata
        self._update_adata_annotations(key_added=key_added)

        return self

    def _update_adata_annotations(self, key_added: str) -> None:
        """Update cluster labels in adata object."""
        logger.info("Writing updated cluster labels to `adata.obs[`%s'].", key_added)
        self.adata.obs[key_added] = None

        for sample, annotator in self.sample_annotators.items():
            mask = self.adata.obs[self.sample_key] == sample
            label_mapping = annotator.annotation_df["cell_type_annotation"].to_dict()
            self.adata.obs.loc[mask, key_added] = self.adata.obs.loc[mask, self.cluster_key].map(label_mapping)

        self.adata.obs[key_added] = self.adata.obs[key_added].astype("category")

    def _get_annotation_summary_string(self, filter_by: str = "") -> str:
        if not self.sample_annotators:
            raise ValueError("No SampleAnnotators found. Run `annotate_clusters` first.")

        summary_string = "\n\n".join(
            f"Sample: {key}\n{_format_annotation(annotator.annotation_df, filter_by)}"
            for key, annotator in self.sample_annotators.items()
        )

        return summary_string

    def harmonize_annotations(self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Assign consistent ordering and naming across cell type annotations.

        Note that for multiple samples with many clusters each, this typically requires a more powerful model
        like `gpt-4o` to work well.

        Parameters
        ----------
        keys : list[str] | str
            List of keys in `adata.obs` to reorder.
        unknown_key : str, optional
            Name of the unknown category. Default is 'Unknown'.

        Returns
        -------
        Nothing, updates `adata.obs` with reordered categories. Underscores will be replaced with spaces.
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
        response = self._query_openai(
            instruction=order_prompt,
            response_format=LabelOrderOutput,
        )
        label_sets = _get_consistent_ordering(self.adata, response.ordered_cell_type_list, keys)

        # Re-add the unknown category, make sure that we retained all categories, and write to adata
        for obs_key, new_cluster_names in label_sets.items():
            if unknown_key in self.adata.obs[obs_key].cat.categories:
                logger.debug(
                    "Readding the unknown category with key '%s'",
                    unknown_key,
                )
                new_cluster_names.append(unknown_key)

            original_categories = self.adata.obs[obs_key].unique()
            if not set(original_categories) == set(new_cluster_names):
                added_categories = set(new_cluster_names) - set(original_categories)
                removed_categories = set(original_categories) - set(new_cluster_names)
                if added_categories or removed_categories:
                    error_message = f"New categories for key `{obs_key}` differ from original categories."
                    if added_categories:
                        error_message += f" Added categories: `{', '.join(added_categories)}`."
                    if removed_categories:
                        error_message += f" Removed categories: `{', '.join(removed_categories)}`."
                    raise ValueError(error_message)

            logger.info("Writing categories for key '%s'", obs_key)
            self.adata.obs[obs_key] = self.adata.obs[obs_key].cat.set_categories(new_cluster_names)

    def get_cluster_colors(self, keys: list[str] | str, unknown_key: str = PackageConstants.unknown_name) -> None:
        """Query OpenAI for relational cluster colors.

        Parameters
        ----------
        keys : list[str] | str
            List of keys in `adata.obs` to query colors for.
        unknown_key : str
            Name of the unknown category.

        Returns
        -------
        Nothing, updates `adata.uns` with cluster colors.

        """
        if isinstance(keys, str):
            keys = [keys]

        logger.info("Iterating over obs keys")
        for key in tqdm(keys):
            # format the cluster names as a string and prepare the query prompt
            cluster_names = ", ".join(cl for cl in self.adata.obs[key].cat.categories if cl != unknown_key)
            color_prompt = Prompts.COLOR_PROMPT.format(cluster_names=cluster_names)

            response = self._query_openai(instruction=color_prompt, response_format=CellTypeColorOutput)
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
