# Copilot Instructions for CellAnnotator

## Project Overview

**CellAnnotator** is an scverse ecosystem package for automated cell type
annotation in scRNA-seq data using Large Language Models (LLMs). It's
provider-agnostic, supporting OpenAI, Google Gemini, and Anthropic Claude.
The tool sends cluster marker genes (not expression values) to LLMs, which
return structured cell type annotations with confidence scores.

### Domain Context
- **Marker genes**: Differentially expressed genes that characterize cell
  types/clusters (computed via `scanpy.tl.rank_genes_groups()`).
- **LLM providers**: OpenAI (GPT), Google (Gemini), Anthropic (Claude).
  Uses Pydantic for structured outputs.
- **Workflow**: 1) Compute marker genes per cluster, 2) Send to LLM with
  biological context, 3) Get structured annotations, 4) Harmonize across samples.

### Key Dependencies
- **Core**: scanpy, pydantic, python-dotenv, rich
- **LLM providers**: openai, anthropic, google-genai (all optional)
- **Optional**: rapids-singlecell (GPU), colorspacious (colors)

## Architecture

### Core Components
1. **`src/cell_annotator/model/cell_annotator.py`**: Main `CellAnnotator` class
   - Orchestrates annotation across multiple samples
   - `annotate_clusters()`: Main entry point for annotation
2. **`src/cell_annotator/model/sample_annotator.py`**: `SampleAnnotator` class
   - Handles annotation for single sample
   - Computes marker genes, queries LLM, stores results
3. **`src/cell_annotator/model/base_annotator.py`**: `BaseAnnotator` abstract class
   - Shared LLM provider logic and validation
4. **`src/cell_annotator/_response_formats.py`**: Pydantic models for structured LLM outputs
5. **`src/cell_annotator/_prompts.py`**: LLM prompt templates
6. **`src/cell_annotator/utils.py`**: Helper functions (marker gene filtering, formatting)

## Project-Specific Patterns

### Basic Usage
```python
from cell_annotator import CellAnnotator

cell_ann = CellAnnotator(
    adata,
    species="human",
    tissue="heart",
    cluster_key="leiden",
    sample_key="batch",
    provider="openai",  # or "gemini", "anthropic"
).annotate_clusters()

# Results in adata.obs['cell_type_predicted']
```

### LLM Provider Selection
- Providers: `"openai"` (default), `"gemini"`, `"anthropic"`
- API keys via environment variables or `.env` file (loaded with python-dotenv)
- Models: `gpt-4o-mini`, `gemini-2.5-flash-lite`, `claude-haiku-4-5` (defaults)
- Anthropic is most expensive ($1/$5 per 1M tokens), minimize usage in tests
- All providers use model aliases that auto-update to latest snapshots

### Structured Outputs with Pydantic
- `CellTypeListOutput`: List of expected cell types
- `ExpectedMarkerGeneOutput`: Dict of cell type → marker genes
- Ensures reliable, parseable LLM responses

### AnnData Conventions
- Results stored in `adata.obs[cell_type_key]` (default: `"cell_type_predicted"`)
- Confidence scores in `adata.obs[f"{cell_type_key}_confidence"]`

## Common Gotchas

1. **API keys**: Must be set as env vars or in `.env` file. Package auto-loads via python-dotenv.
2. **Provider packages**: Install provider extras (`pip install cell-annotator[openai]`) to use specific LLMs.
3. **Real LLM tests**: Use `@pytest.mark.real_llm_query` and skip in CI unless explicitly enabled.
4. **Marker gene filtering**: Package automatically filters marker genes to genes present in `adata.var_names`.

## Related Resources

- **OpenAI structured outputs**: https://platform.openai.com/docs/guides/structured-outputs
- **scanpy docs**: https://scanpy.readthedocs.io/
- **Pydantic docs**: https://docs.pydantic.dev/
