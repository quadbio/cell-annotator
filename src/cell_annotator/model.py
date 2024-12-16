import os

import numpy as np
import scanpy as sc
from dotenv import load_dotenv
from pandas import DataFrame
from tqdm.auto import tqdm

from cell_annotator._constants import ExpectedCellTypeOutput, ExpectedMarkerGeneOutput, PredictedCellTypeOutput, Prompts
from cell_annotator._logging import logger
from cell_annotator.utils import _get_auc, _get_specificity, _query_openai

MAX_MARKERS_RAW = 200
MIN_MARKERS_RAW = 15


class CellAnnotator:
    """Main class for cell annotation.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object
    species : str
        Species name
    tissue : str
        Tissue name
    stage : str
        Developmental stage. Default is 'adult'.
    cluster_key : str
        Key of the cluster column in adata.obs
    model : str
        OpenAI model name. Default is 'gpt-4o-mini'
    """

    def __init__(
        self,
        adata: sc.AnnData,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        model: str = "gpt-4o-mini",
    ):
        self.model = model
        self.adata = adata
        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.marker_gene_dfs = None
        self.marker_genes = None
        self.annotation_df = None
        self._expected_cell_types = None
        self._expected_marker_genes = None

        # laod environmental variables
        load_dotenv()

        # Check if the environment variable OPENAI_API_KEY is set
        if os.getenv("OPENAI_API_KEY"):
            logger.info("The environment variable `OPENAI_API_KEY` is set (that's good).")
        else:
            logger.warning(
                "The environment variable `OPENAI_API_KEY` is not set. Head over to https://platform.openai.com/api-keys to get a key and store it in an .env file."
            )

    def __repr__(self):
        return (
            f"CellAnnotator(model={self.model!r}, species={self.species!r}, "
            f"tissue={self.tissue!r}, stage={self.stage!r}, cluster_key={self.cluster_key!r})"
        )

    @property
    def expected_cell_types(self):
        """Display expected cell types."""
        if self._expected_cell_types is None:
            logger.warning("Expected cell types have not been queried yet.")
            return None
        print("Expected Cell Types:")
        for cell_type in self._expected_cell_types:
            print(f"- {cell_type}")

    @expected_cell_types.setter
    def expected_cell_types(self, value):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("Expected cell types must be a list of strings.")
        logger.info("Writing expected cell types to `self.expected_cell_types`.")
        self._expected_cell_types = value

    @property
    def expected_marker_genes(self):
        """Display expected marker genes."""
        if self._expected_marker_genes is None:
            logger.warning("Expected marker genes have not been queried yet.")
            return None
        print("Expected Marker Genes:")
        for cell_type, markers in self._expected_marker_genes.items():
            print(f"- {cell_type}: {', '.join(markers)}")

    @expected_marker_genes.setter
    def expected_marker_genes(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(item, str) for item in v)
            for k, v in value.items()
        ):
            raise ValueError(
                "Expected marker genes must be a dictionary with cell types as keys and lists of marker genes as values."
            )
        logger.info("Writing expected marker genes `self.expected_marker genes`.")
        self._expected_marker_genes = value

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
        agent_desc = Prompts.AGENT_DESCRIPTION.format(species=self.species)

        logger.info("Querying cell types.")
        res_types = _query_openai(
            agent_description=agent_desc,
            instruction=cell_type_prompt,
            response_format=ExpectedCellTypeOutput,
            model=self.model,
            max_tokens=max_tokens,
        )
        self.expected_cell_types = res_types.expected_cell_types

        marker_gene_prompt = [
            {"role": "assistant", "content": "; ".join(self._expected_cell_types) if self._expected_cell_types else ""},
            {"role": "user", "content": Prompts.CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)},
        ]

        logger.info("Querying cell type markers.")
        res_markers = _query_openai(
            agent_description=agent_desc,
            instruction=cell_type_prompt,
            other_messages=marker_gene_prompt,
            response_format=ExpectedMarkerGeneOutput,
            model=self.model,
            max_tokens=max_tokens,
        )
        self.expected_marker_genes = {
            cell_type_markers.cell_type_name: cell_type_markers.expected_marker_genes
            for cell_type_markers in res_markers.expected_markers_per_cell_type
        }

    def get_cluster_markers(
        self,
        method: str = "wilcoxon",
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
        logger.info("Computing marker genes per cluster using method `%s`.", method)
        sc.tl.rank_genes_groups(
            self.adata, groupby=self.cluster_key, method=method, use_raw=use_raw, n_genes=MAX_MARKERS_RAW
        )

        marker_dfs = {}
        logger.info("Iterating over clusters to compute specificity and AUC values.")
        for cli in tqdm(self.adata.obs[self.cluster_key].unique()):
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

        logger.info("Writing marker gene DataFrames to `self.marker_gene_dfs`.")
        self.marker_gene_dfs = marker_dfs

        # filter to the top markers
        self._filter_cluster_markers(min_auc=min_auc, max_markers=max_markers)

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

        # Attempt to sort the dictionary by cluster keys
        try:
            sorted_marker_genes = {k: marker_genes[k] for k in sorted(marker_genes, key=lambda x: int(x))}
        except ValueError:
            logger.warning("Cluster keys cannot be converted to integers. Keeping original order.")
            sorted_marker_genes = marker_genes

        logger.info("Writing top marker genes to `self.marker_genes`.")
        self.marker_genes = sorted_marker_genes

    def _update_adata_annotations(self, key_added: str):
        """Update cluster labels in adata object."""
        if self.annotation_df is None:
            raise ValueError("Run `annotate_clusters` first to get annotation results.")

        # map cluster labels to cell types
        logger.info("Writing updated cluster labels to `adata.obs['%s']`.", key_added)
        label_mapping = self.annotation_df["cell_type_annotation"].to_dict()
        self.adata.obs[key_added] = self.adata.obs[self.cluster_key].map(label_mapping)
        self.adata.obs[key_added] = self.adata.obs[key_added].astype("category")

        # update cluster ordering
        self.adata.obs[key_added] = self.adata.obs[key_added].cat.reorder_categories(
            self.annotation_df["cell_type_annotation"].unique()
        )

    def annotate_clusters(self, max_tokens: int | None = None, key_added: str = "cell_type_predicted"):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        max_tokens : int
            Maximum number of tokens for OpenAI API.
        key_added : str
            Key in `adata.obs` where predicted cell type labels will be written.

        Returns
        -------
        Nothing, writes annotation results to `self.annotation_df`.

        """
        answers = {}
        if self._expected_marker_genes is None:
            logger.info(
                "Querying expected cell type markers with default parameters. Run `get_expected_cell_type_markers` for more control."
            )
            self.get_expected_cell_type_markers()

        if self.marker_genes is None:
            logger.info(
                "Computing cluster marker genes using default parameters. Run `get_cluster_markers` for more control. "
            )
            self.get_cluster_markers()

        # parse expected markers into a string
        expected_markers_string = "\n".join(
            [f"{cell_type}: {', '.join(genes)}" for cell_type, genes in self._expected_marker_genes.items()]
        )

        actual_markers_all = "\n".join([f'- Cluster {i}: {", ".join(gs)}' for i, gs in self.marker_genes.items()])
        agent_desc = Prompts.AGENT_DESCRIPTION.format(species=self.species)

        # loop over clusters to annotate
        logger.info("Looping over clusters to annotate.")
        for cluster in tqdm(self.marker_genes):
            actual_markers_cluster = ", ".join(self.marker_genes[cluster])

            # fill in the annotation prompt
            annotation_prompt = Prompts.ANNOTATION_PROMPT.format(
                species=self.species,
                tissue=self.tissue,
                stage=self.stage,
                actual_markers_all=actual_markers_all,
                cluster_id=cluster,
                actual_markers_cluster=actual_markers_cluster,
                expected_markers=expected_markers_string,
            )

            answers[cluster] = _query_openai(
                agent_description=agent_desc,
                instruction=annotation_prompt,
                response_format=PredictedCellTypeOutput,
                model=self.model,
                max_tokens=max_tokens,
            )

        logger.info("Writing annotation results to `self.annotation_df`.")
        self.annotation_df = DataFrame.from_dict({k: v.model_dump() for k, v in answers.items()}, orient="index")

        # update the adata object with the annotations
        self._update_adata_annotations(key_added=key_added)

        return self
