"""Cell annotator class for annotating cell types across multiple samples."""

import scanpy as sc
from scanpy.tools._rank_genes_groups import _Method
from tqdm.auto import tqdm

from cell_annotator._constants import PackageConstants
from cell_annotator._docs import d
from cell_annotator._logging import logger
from cell_annotator._response_formats import CellTypeListOutput
from cell_annotator.model.base_annotator import BaseAnnotator
from cell_annotator.model.reference_providers import create_orchestrator
from cell_annotator.model.sample_annotator import SampleAnnotator
from cell_annotator.utils import _filter_marker_genes_to_adata, _format_annotation, _try_sorting_dict_by_keys


@d.dedent
class CellAnnotator(BaseAnnotator):
    """
    Main class for annotating cell types across multiple samples.

    Orchestrates the annotation workflow by creating SampleAnnotator instances for
    each sample, coordinating marker gene computation, cell type annotation, and
    harmonizing results across samples. Supports any LLM provider backend.

    Parameters
    ----------
    %(adata)s
    %(sample_key)s
    %(species)s
    %(tissue)s
    %(stage)s
    %(cluster_key)s
    %(model)s
    %(max_completion_tokens)s
    %(provider)s
    %(api_key)s
    """

    def __init__(
        self,
        adata: sc.AnnData,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = PackageConstants.default_cluster_key,
        sample_key: str | None = None,
        model: str | None = None,
        max_completion_tokens: int | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        reference_provider: str = "llm",
    ):
        super().__init__(species, tissue, stage, cluster_key, model, max_completion_tokens, provider, api_key)
        self.adata = adata
        self.sample_key = sample_key
        self._api_key = api_key  # Store API key for passing to SampleAnnotators
        self.reference_provider = reference_provider

        self.sample_annotators: dict[str, SampleAnnotator] = {}
        self.expected_cell_types: list[str] = []
        self.expected_marker_genes: dict[str, list[str]] | None = None
        self.annotated: bool = False
        self.cell_type_key: str = PackageConstants.cell_type_key
        self.global_cell_type_list: list[str] | None = None

        # Initialize SampleAnnotators for each batch
        self._initialize_sample_annotators()

    def __repr__(self) -> str:
        """Return a string representation of the CellAnnotator."""
        lines = []
        lines.append(f"🧬 {self.__class__.__name__}")
        lines.append("=" * (len(self.__class__.__name__) + 3))

        # Biological context
        lines.append(f"📋 Species: {self.species}")
        lines.append(f"🔬 Tissue: {self.tissue}")
        lines.append(f"⏳ Stage: {self.stage}")
        lines.append(f"🔗 Cluster key: {self.cluster_key}")
        lines.append(f"🔬 Sample key: {self.sample_key}")

        # Model configuration
        lines.append("")
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
            # Catch all exceptions during test (API errors, invalid models, etc.)
            logger.debug("Status check failed: %s", str(e))
            status = "⚠️ Unknown"
        lines.append(f"🔋 Status: {status}")

        # Sample information
        lines.append("")
        lines.append(f"📊 Samples: {len(self.sample_annotators)}")
        if self.sample_annotators:
            sample_list = list(self.sample_annotators.keys())
            if len(sample_list) <= 5:
                # Show all samples if 5 or fewer
                sample_summary = ", ".join(f"'{sample}'" for sample in sample_list)
            else:
                # Show first 3 and last 2 with ellipsis
                first_samples = ", ".join(f"'{sample}'" for sample in sample_list[:3])
                last_samples = ", ".join(f"'{sample}'" for sample in sample_list[-2:])
                sample_summary = f"{first_samples}, ..., {last_samples}"
            lines.append(f"🏷️  Sample IDs: {sample_summary}")

        return "\n".join(lines)

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
                provider=self._provider_name,
                api_key=self._api_key,  # Pass API key to SampleAnnotator
                _skip_validation=True,  # Skip validation since parent already validated
            )

        # sort by keys for visual pleasure
        self.sample_annotators = _try_sorting_dict_by_keys(self.sample_annotators)

    @d.dedent
    def get_expected_cell_type_markers(
        self,
        n_markers: int = 5,
        filter_to_var_names: bool = True,
        provide_var_names: bool = True,
    ) -> None:
        """Get expected cell types and marker genes using the configured reference provider.

        Parameters
        ----------
        %(n_markers)s
        filter_to_var_names
            Whether to filter marker genes to only include those present in `adata.var_names`
        provide_var_names
            If True, include the available gene names in the prompt and instruct the model to restrict itself to this set.
            Only used for LLM-based providers.

        Returns
        -------
        Updates the following attributes:
        - `self.expected_cell_types`
        - `self.expected_marker_genes`
        """
        logger.info("Creating reference provider orchestrator: %s", self.reference_provider)

        # Create the reference orchestrator based on the provider string
        orchestrator = create_orchestrator(self.reference_provider, annotator=self)

        # Get both cell types and markers from the orchestrator
        logger.info("Querying cell types and markers using %s", self.reference_provider)
        self.expected_cell_types, raw_marker_genes = orchestrator.get_cell_types_and_markers(
            tissue=self.tissue, species=self.species, stage=self.stage, n_markers=n_markers
        )

        logger.info("Found %d expected cell types: %s", len(self.expected_cell_types), self.expected_cell_types)

        # Filter marker genes to available genes if requested
        if filter_to_var_names:
            self.expected_marker_genes = _filter_marker_genes_to_adata(raw_marker_genes, self.adata)
        else:
            self.expected_marker_genes = raw_marker_genes

        logger.info("Finished getting expected cell types and markers using %s", self.reference_provider)

    @d.dedent
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
        %(method_rank_genes_groups)s
        %(min_specificity)s
        %(min_auc)s
        %(max_markers)s
        %(use_raw)s
        %(use_rapids)s

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

    @d.dedent
    def annotate_clusters(
        self, min_markers: int = 2, restrict_to_expected: bool = False, key_added: str = "cell_type_predicted"
    ):
        """Annotate clusters based on marker genes.

        Parameters
        ----------
        %(min_markers)s
        %(key_added)s
        %(restrict_to_expected)s

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

        # query llm
        logger.info("Querying cell-type label de-duplication.")
        response = self.query_llm(
            instruction=deduplication_prompt,
            response_format=CellTypeListOutput,
        )
        self.global_cell_type_list = response.cell_type_list
        logger.info("Removed %s/%s cell types.", len(cell_types) - len(self.global_cell_type_list), len(cell_types))

        # Step 2: map cell types to harmonized names for each sample annotator
        logger.info("Iterating over samples to harmonize cell type annotations.")
        for annotator in tqdm(self.sample_annotators.values()):
            annotator.harmonize_annotations(self.global_cell_type_list, unknown_key=unknown_key)
