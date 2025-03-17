"""Sample annotator class to annotate an individual sample."""

import numpy as np
import pandas as pd
import scanpy as sc
from pandas import DataFrame
from scanpy.tools._rank_genes_groups import _Method

from cell_annotator._constants import PackageConstants
from cell_annotator._logging import logger
from cell_annotator._response_formats import BaseOutput, CellTypeMappingOutput, PredictedCellTypeOutput
from cell_annotator.base_annotator import BaseAnnotator
from cell_annotator.utils import _filter_by_category_size, _get_auc, _get_specificity, _try_sorting_dict_by_keys

try:
    import rapids_singlecell as rsc

    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False


class SampleAnnotator(BaseAnnotator):
    """
    Handles annotation for a single batch/sample.

    Parameters
    ----------
    adata
        Subset of the main AnnData object corresponding to a single batch.
    species
        Species name (inherited from BaseAnnotator).
    tissue
        Tissue name (inherited from BaseAnnotator).
    stage
        Developmental stage (inherited from BaseAnnotator).
    expected_marker_genes
        Precomputed dict, mapping expected cell types to marker genes.
    cluster_key
        Key of the cluster column in adata.obs (inherited from BaseAnnotator).
    model
        OpenAI model name (inherited from BaseAnnotator).
    max_completion_tokens
        Maximum number of tokens the model is allowed to use.

    """

    def __init__(
        self,
        adata: sc.AnnData,
        sample_name: str,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = PackageConstants.default_cluster_key,
        model: str = PackageConstants.default_model,
        max_completion_tokens: int | None = None,
    ):
        super().__init__(species, tissue, stage, cluster_key, model, max_completion_tokens)
        self.adata = adata
        self.sample_name = sample_name

        self.annotation_df: pd.DataFrame | None = None
        self.marker_gene_dfs: dict[str, pd.DataFrame] | None = None
        self.marker_genes: dict[str, list[str]] = {}
        self.annotation_dict: dict[str, BaseOutput] = {}
        self.local_cell_type_mapping: dict[str, str] = {}

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
        use_rapids: bool = False,
    ) -> None:
        """Get marker genes per cluster

        Parameters
        ----------
        method
            Method for `sc.tl.rank_genes_groups`
        min_cells_per_cluster
            Include only clusters with at least this many cells.
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
        # filter out very small clusters
        self._filter_clusters_by_cell_number(min_cells_per_cluster)

        if use_rapids and method == "logreg":
            if not RAPIDS_AVAILABLE:
                raise ImportError(
                    "`rapids_singelcell` is not installed. Please install it following the instructions from https://rapids-singlecell.readthedocs.io/en/latest/Installation.html"
                )
            logger.debug("Computing marker genes per cluster on GPU using method `%s`.", method)
            rsc.tl.rank_genes_groups_logreg(
                self.adata, groupby=self.cluster_key, use_raw=use_raw, n_genes=PackageConstants.max_markers
            )
        else:
            # Compute AUC scores on CPU
            logger.debug("Computing marker genes per cluster on CPU using method `%s`.", method)
            sc.tl.rank_genes_groups(
                self.adata,
                groupby=self.cluster_key,
                method=method,
                use_raw=use_raw,
                n_genes=PackageConstants.max_markers,
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
            auc = _get_auc(genes=genes, clust_mask=clust_mask, adata=self.adata, use_raw=use_raw, use_rapids=use_rapids)
            marker_dfs[cli] = DataFrame({"gene": genes, "specificity": specificity, "auc": auc})

        logger.debug(
            "Writing marker gene DataFrames to `self.sample_annotators['%s'].marker_gene_dfs`.", self.sample_name
        )
        self.marker_gene_dfs = marker_dfs

        # filter to the top markers
        logger.debug("Writing top marker genes to `self.sample_annotators['%s'].marker_genes`.", self.sample_name)
        self._filter_cluster_markers(min_auc=min_auc, max_markers=max_markers)

    def _filter_clusters_by_cell_number(self, min_cells_per_cluster: int) -> None:
        removed_info = _filter_by_category_size(self.adata, column=self.cluster_key, min_size=min_cells_per_cluster)
        if removed_info:
            for cat, size in removed_info.items():
                failure_reason = f"Not enough cells for cluster {cat} in sample `{self.sample_name}` ({size}<{min_cells_per_cluster})."
                logger.warning(failure_reason)
                self.annotation_dict[cat] = PredictedCellTypeOutput.default_failure(failure_reason=failure_reason)

    def _filter_cluster_markers(self, min_auc: float, max_markers: int) -> None:
        """Get top markers

        Parameters
        ----------
        min_auc
            Minimum AUC
        max_markers
            Maximum number of markers

        Returns
        -------
        Updates the following attributes:
        - `self.marker_genes`

        """
        if self.marker_gene_dfs is None:
            raise ValueError("Run `get_markers` first to compute marker genes per cluster.")

        marker_genes = {}
        for cluster, df in self.marker_gene_dfs.items():
            top_genes = df[df.auc > min_auc].sort_values("auc", ascending=False).head(max_markers).gene.values
            marker_genes[cluster] = list(top_genes)

        self.marker_genes = _try_sorting_dict_by_keys(marker_genes)

    def annotate_clusters(
        self, min_markers: int, expected_marker_genes: dict[str, list[str]] | None, restrict_to_expected: bool = False
    ) -> None:
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        max_completion_tokens
            Maximum number of tokens for OpenAI API.
        min_markers
            Minimum number of requires marker genes per cluster.
        expected_marker_genes
            Expected marker genes per cell type.
        restrict_to_expected
            If True, only use expected cell types for annotation.

        Returns
        -------
        Updates the following attributes:
        - `self.annotation_dict`
        - `self.annotation_df`

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

        actual_markers_all = "\n".join([f"- Cluster {i}: {', '.join(gs)}" for i, gs in self.marker_genes.items()])

        # loop over clusters to annotate
        logger.debug("Iterating over clusters to annotate.")
        for cluster in self.marker_genes:
            actual_markers_cluster = self.marker_genes[cluster]

            if len(actual_markers_cluster) < min_markers:
                failure_reason = f"Not enough markers provided for cluster {cluster} in sample `{self.sample_name}` ({len(actual_markers_cluster)}<{min_markers})."
                logger.warning(failure_reason)
                self.annotation_dict[cluster] = PredictedCellTypeOutput.default_failure(failure_reason=failure_reason)
            else:
                actual_markers_cluster_string = ", ".join(actual_markers_cluster)

                # fill in the annotation prompt
                annotation_prompt = self.prompts.get_annotation_prompt(
                    actual_markers_all=actual_markers_all,
                    cluster_id=cluster,
                    actual_markers_cluster=actual_markers_cluster_string,
                    expected_markers=expected_markers_string,
                    restrict_to_expected=restrict_to_expected,
                )

                self.annotation_dict[cluster] = self.query_openai(
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

    def harmonize_annotations(
        self, global_cell_type_list: list[str], unknown_key: str = PackageConstants.unknown_name
    ) -> None:
        """Map local cell type names to global cell type names.

        Parameters
        ----------
        global_cell_type_list
            List of global cell types.
        unknown_key
            Key for the unknown category.

        Returns
        -------
        Updates the following fields:
        - `self.local_cell_type_mapping`
        - `self.annotation_df["cell_type_harmonized"]`

        """
        if self.annotation_df is None:
            raise ValueError("The annotation DataFrame is not initialized. Run `annotate_clusters` first.")
        original_categories = self.annotation_df["cell_type"].unique()
        local_cell_type_mapping = {cat: "" for cat in original_categories if cat != unknown_key}

        logger.debug("Iterating over clusters to map local annotations to global naming scheme.")
        for cat in local_cell_type_mapping.keys():
            mapping_prompt = self.prompts.get_mapping_prompt(
                global_cell_type_list=", ".join(global_cell_type_list),
                local_cell_type_list=", ".join(local_cell_type_mapping.keys()),
                current_cell_type=cat,
            )

            response = self.query_openai(
                instruction=mapping_prompt,
                response_format=CellTypeMappingOutput,
            )

            # Verfiy the mapping
            global_name = response.mapped_global_name
            if global_name not in global_cell_type_list:
                logger.warning(
                    "Invalid global cell type name: %s in sample `%s`. Re-using the original name: %s.",
                    global_name,
                    self.sample_name,
                    cat,
                )
                local_cell_type_mapping[cat] = cat
            else:
                local_cell_type_mapping[cat] = global_name

        # Re-add the unkonwn category if it was present originally
        if unknown_key in original_categories:
            local_cell_type_mapping[unknown_key] = unknown_key

        # Write to self
        self.local_cell_type_mapping = local_cell_type_mapping

        # Introduce a new column "cell_type_harmonized" in annotator.annotation_df
        self.annotation_df["cell_type_harmonized"] = self.annotation_df["cell_type"].map(local_cell_type_mapping)
