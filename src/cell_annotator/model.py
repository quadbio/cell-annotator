import os

import numpy as np
import scanpy as sc
from dotenv import load_dotenv
from pandas import DataFrame
from scanpy.tools._rank_genes_groups import _Method
from tqdm.auto import tqdm

from cell_annotator._constants import ExpectedCellTypeOutput, ExpectedMarkerGeneOutput, PredictedCellTypeOutput, Prompts
from cell_annotator._logging import logger
from cell_annotator.utils import (
    _filter_by_category_size,
    _get_auc,
    _get_specificity,
    _query_openai,
    _try_sorting_dict_by_keys,
)

ResponseOutput = ExpectedCellTypeOutput | ExpectedMarkerGeneOutput | PredictedCellTypeOutput


MAX_MARKERS_RAW = 200
MIN_MARKERS_RAW = 15


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
    """

    def __init__(
        self,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        model: str = "gpt-4o-mini",
    ):
        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.model = model
        self.expected_marker_genes = {}  # this gets set in the SampleAnnotator only when the user calls `.annotate_clusters`

    def _query_openai(
        self,
        instruction: str,
        response_format: ResponseOutput,
        other_messages: list | None = None,
        max_tokens: int | None = None,
    ) -> ResponseOutput:
        agent_description = Prompts.AGENT_DESCRIPTION.format(species=self.species)

        response = _query_openai(
            agent_description=agent_description,
            instruction=instruction,
            model=self.model,
            response_format=response_format,
            other_messages=other_messages,
            max_tokens=max_tokens,
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
    ):
        super().__init__(species, tissue, stage, cluster_key, model)
        self.adata = adata
        self.sample_name = sample_name
        self.n_cells_per_cluster = _try_sorting_dict_by_keys(self.adata.obs[self.cluster_key].value_counts().to_dict())
        self.annotation_dict = {}
        self.annotation_df = None
        self.marker_gene_dfs = None
        self.marker_genes = None

    def __repr__(self):
        return f"SampleAnnotator(sample_name={self.sample_name!r}, n_clusters={self.adata.obs[self.cluster_key].nunique()}, n_cells={self.adata.n_obs:,})"

    def get_cluster_markers(
        self,
        method: _Method | None = "wilcoxon",
        min_cells_per_cluster: int = 3,
        min_specificity: float = 0.75,
        min_auc: float = 0.7,
        max_markers: int = 7,
        use_raw: bool = True,
    ):
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
        Nothing, sets `self.marker_dfs`.

        """
        # filter out very small clusters
        self._filter_clusters_by_cell_number(min_cells_per_cluster)

        logger.debug("Computing marker genes per cluster using method `%s`.", method)
        sc.tl.rank_genes_groups(
            self.adata, groupby=self.cluster_key, method=method, use_raw=use_raw, n_genes=MAX_MARKERS_RAW
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
            mask = specificity >= min(min_specificity, sorted(specificity)[-MIN_MARKERS_RAW])
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
        self._filter_cluster_markers(min_auc=min_auc, max_markers=max_markers)

    def _filter_clusters_by_cell_number(self, min_cells_per_cluster: int):
        removed_info = _filter_by_category_size(self.adata, column=self.cluster_key, min_size=min_cells_per_cluster)
        if removed_info:
            for cat, size in removed_info.items():
                failure_reason = f"Not enough cells for cluster {cat} in sample `{self.sample_name}` ({size}<{min_cells_per_cluster})."
                logger.warning(failure_reason)
                self.annotation_dict[cat] = PredictedCellTypeOutput.default_failure(failure_reason=failure_reason)

    def _filter_cluster_markers(self, min_auc: float, max_markers: int):
        """Get top markers

        Parameters
        ----------
        min_auc : float
            Minimum AUC
        max_markers : int
            Maximum number of markers

        Returns
        -------
        Nothing, sets `self.marker_genes`.

        """
        if self.marker_gene_dfs is None:
            raise ValueError("Run `get_markers` first to compute marker genes per cluster.")

        marker_genes = {}
        for cluster, df in self.marker_gene_dfs.items():
            top_genes = df[df.auc > min_auc].sort_values("auc", ascending=False).head(max_markers).gene.values
            marker_genes[cluster] = list(top_genes)

        logger.debug("Writing top marker genes to `self.sample_annotators['%s'].marker_genes`.", self.sample_name)
        self.marker_genes = _try_sorting_dict_by_keys(marker_genes)

    def annotate_clusters(self, max_tokens: int | None = None, min_markers: int = 2):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        max_tokens : int
            Maximum number of tokens for OpenAI API.
        min_markers : int
            Minimum number of requires marker genes per cluster.

        Returns
        -------
        Nothing, writes annotation results to `self.annotation_df`.

        """
        if self.marker_genes is None:
            logger.debug(
                "Computing cluster marker genes using default parameters. Run `get_cluster_markers` for more control. "
            )
            self.get_cluster_markers()

        # parse expected markers into a string
        if self.expected_marker_genes:
            expected_markers_string = "\n".join(
                [f"{cell_type}: {', '.join(genes)}" for cell_type, genes in self.expected_marker_genes.items()]
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
                    max_tokens=max_tokens,
                )

        logger.debug("Writing annotation results to `self.sample_annotators['%s'].annotation_df`.", self.sample_name)
        self.annotation_df = DataFrame.from_dict(
            {k: v.model_dump() for k, v in self.annotation_dict.items()}, orient="index"
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
        self.sample_annotators = {}
        self.expected_cell_types = []
        self.harmonized_annotations = None

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

    def _initialize_sample_annotators(self):
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
            )

    def get_expected_cell_type_markers(self, n_markers: int = 5, max_tokens: int | None = 1000):
        """Get expected cell types and marker genes.

        Parameters
        ----------
        n_markers : int
            Number of marker genes per cell type.
        max_tokens : int
            Maximum number of tokens for OpenAI API.

        Returns
        -------
        Nothing, sets `self.expected_cell_types` and `self.expected_marker_genes`.
        """
        cell_type_prompt = Prompts.CELL_TYPE_PROMPT.format(species=self.species, tissue=self.tissue, stage=self.stage)

        logger.info("Querying cell types.")
        res_types = self._query_openai(
            instruction=cell_type_prompt,
            response_format=ExpectedCellTypeOutput,
            max_tokens=max_tokens,
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
            max_tokens=max_tokens,
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
        use_raw: bool = True,
    ):
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

    def annotate_clusters(self, max_tokens: int | None = None):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        max_tokens : int
            Maximum number of tokens for OpenAI API.

        Returns
        -------
        Nothing, writes annotation results to `self.annotation_df`.

        """
        logger.info("Iterating over samples to annotate clusters. ")
        for annotator in tqdm(self.sample_annotators.values()):
            # set expected marker genes only here, so that the user can manually filter
            annotator.expected_marker_genes = self.expected_marker_genes
            annotator.annotate_clusters(
                max_tokens=max_tokens,
            )

    def harmonize_annotations(self, max_tokens: int | None = None):
        """
        Harmonize annotations across samples.

        Parameters
        ----------
        max_tokens : int, optional
            Maximum number of tokens for OpenAI API.

        Returns
        -------
        pd.DataFrame
            A single DataFrame with harmonized annotations for all samples.
        """
        pass
