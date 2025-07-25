"""Shared documentation for cell-annotator."""

from docrep import DocstringProcessor

__all__ = ["d"]

# Core biological parameters
_species = """\
species
    Species name (e.g., 'homo sapiens', 'mus musculus')."""

_tissue = """\
tissue
    Tissue name (e.g., 'brain', 'heart', 'lung')."""

_stage = """\
stage
    Developmental stage (e.g., 'adult', 'embryonic', 'fetal')."""

# Data structure parameters
_adata = """\
adata
    AnnData object containing single-cell data."""

_adata_sample = """\
adata
    AnnData object containing single-cell data for one sample/batch."""

_sample_key = """\
sample_key
    Key in :attr:`~anndata.AnnData.obs` indicating sample/batch membership.
    If None, treats the entire dataset as a single sample."""

_sample_name = """\
sample_name
    Identifier for this sample."""

_cluster_key = """\
cluster_key
    Key of the cluster column in adata.obs."""

# LLM configuration parameters
_model = """\
model
    Model name. If None, uses the default model for the selected or auto-detected provider.
    Examples: 'gpt-4o-mini', 'gemini-2.5-flash', 'claude-3-haiku'."""

_max_completion_tokens = """\
max_completion_tokens
    Maximum number of tokens the model is allowed to use for completion."""

_provider = """\
provider
    LLM provider name. If None, auto-detects from model name or uses the first
    available provider with a valid API key. See PackageConstants.supported_providers
    for the list of supported providers."""

_api_key = """\
api_key
    Optional API key for the selected provider. If None, uses environment variables.
    Useful for programmatically providing API keys or using different keys per instance."""

# Query and response parameters
_instruction = """\
instruction
    Instruction to provide to the model."""

_response_format = """\
response_format
    Response format class."""

_other_messages = """\
other_messages
    Additional messages to provide to the model."""

# Annotation workflow parameters
_n_markers = """\
n_markers
    Number of marker genes per cell type."""

_min_markers = """\
min_markers
    Minimal number of required marker genes per cluster."""

_key_added = """\
key_added
    Name of the key in .obs where updated annotations will be written."""

_restrict_to_expected = """\
restrict_to_expected
    If True, only use expected cell types for annotation."""

_method_rank_genes_groups = """\
method
    Method for `sc.tl.rank_genes_groups`."""

_method = """\
method
    Method for marker gene computation. See scanpy.tl.rank_genes_groups for details."""

_min_cells_per_cluster = """\
min_cells_per_cluster
    Include only clusters with at least this many cells."""

_min_specificity = """\
min_specificity
    Minimum specificity threshold for marker genes."""

_min_auc = """\
min_auc
    Minimum AUC threshold for marker genes."""

_max_markers = """\
max_markers
    Maximum number of marker genes per cluster."""

_use_raw = """\
use_raw
    Whether to use raw data for calculations."""

_use_rapids = """\
use_rapids
    Whether to use RAPIDS for GPU acceleration."""

# Common returns
_returns_none = """\
Returns
-------
None"""

_returns_parsed_response = """\
Returns
-------
Parsed response."""

_returns_list_str = """\
Returns
-------
list[str]
    List of available model names."""

d = DocstringProcessor(
    species=_species,
    tissue=_tissue,
    stage=_stage,
    adata=_adata,
    adata_sample=_adata_sample,
    sample_key=_sample_key,
    sample_name=_sample_name,
    cluster_key=_cluster_key,
    model=_model,
    max_completion_tokens=_max_completion_tokens,
    provider=_provider,
    api_key=_api_key,
    instruction=_instruction,
    response_format=_response_format,
    other_messages=_other_messages,
    n_markers=_n_markers,
    min_markers=_min_markers,
    key_added=_key_added,
    restrict_to_expected=_restrict_to_expected,
    method_rank_genes_groups=_method_rank_genes_groups,
    method=_method,
    min_cells_per_cluster=_min_cells_per_cluster,
    min_specificity=_min_specificity,
    min_auc=_min_auc,
    max_markers=_max_markers,
    use_raw=_use_raw,
    use_rapids=_use_rapids,
    returns_none=_returns_none,
    returns_parsed_response=_returns_parsed_response,
    returns_list_str=_returns_list_str,
)
