# Copilot Instructions for CellAnnotator

## Important Notes
- Avoid drafting summary documents or endless markdown files. Just summarize in chat what you did, why, and any open questions.
- Don't update Jupyter notebooks - those are managed manually.
- When running terminal commands, activate the appropriate environment first (`mamba activate cell_annotator`).
- Rather than making assumptions, ask for clarification when uncertain.
- **GitHub workflows**: Use GitHub CLI (`gh`) when possible. For GitHub MCP server tools, ensure Docker Desktop is running first (`open -a "Docker Desktop"`).

## Project Overview

**CellAnnotator** is an scverse ecosystem package for automated cell type annotation in scRNA-seq data using Large Language Models (LLMs). It's provider-agnostic, supporting OpenAI, Google Gemini, and Anthropic Claude. The tool sends cluster marker genes (not expression values) to LLMs, which return structured cell type annotations with confidence scores.

### Domain Context (Brief)
- **AnnData**: Standard single-cell data structure. Contains `.X`, `.obs` (cell metadata), `.var` (gene metadata).
- **Marker genes**: Differentially expressed genes that characterize cell types/clusters (computed via scanpy).
- **LLM providers**: OpenAI (GPT), Google (Gemini), Anthropic (Claude). Uses Pydantic for structured outputs.
- **Workflow**: 1) Compute marker genes per cluster, 2) Send to LLM with biological context, 3) Get structured annotations, 4) Harmonize across samples.

### Key Dependencies`
- **Core**: scanpy, pydantic, python-dotenv, rich
- **LLM providers**: openai, anthropic, google-genai (all optional)
- **Optional**: rapids-singlecell (GPU), colorspacious (colors)

## Architecture & Code Organization

### Module Structure (follows scverse conventions)
- Use `AnnData` objects as primary data structure
- Type annotations use modern syntax: `str | None` instead of `Optional[str]`
- Supports Python 3.11, 3.12, 3.13 (see `pyproject.toml`)
- Avoid local imports unless necessary for circular import resolution

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

## Development Workflow

### Environment Management (Hatch-based)
```bash
# Testing - NEVER use pytest directly
hatch test                    # test with highest Python version
hatch test --all              # test all Python 3.11 & 3.13 + pre-release

# Documentation
hatch run docs:build          # build Sphinx docs
hatch run docs:open           # open in browser
hatch run docs:clean          # clean build artifacts

# Environment inspection
hatch env show                # list environments
```

### Testing Strategy
- Test matrix defined in `[[tool.hatch.envs.hatch-test.matrix]]` in `pyproject.toml`
- Tests Python 3.11 & 3.13 with stable deps, 3.13 with pre-release deps
- Tests live in `tests/`, use pytest with `@pytest.mark.real_llm_query` for actual LLM calls
- Run via `hatch test` to ensure proper environment isolation
- Optional dependencies tested via `features = ["test"]` which includes all providers

### Code Quality Tools
- **Ruff**: Linting and formatting (120 char line length)
- **Biome**: JSON/JSONC formatting with trailing commas
- **Pre-commit**: Auto-runs ruff, biome. Install with `pre-commit install`
- Use `git pull --rebase` if pre-commit.ci commits to your branch

## Key Configuration Files

### `pyproject.toml`
- **Build**: `hatchling` with `hatch-vcs` for git-based versioning
- **Dependencies**: Minimal core (scanpy, pydantic); provider packages are optional extras
- **Extras**: `[openai]`, `[anthropic]`, `[gemini]`, `[all-providers]`, `[test]`, `[doc]`
- **Ruff**: 120 char line length, NumPy docstring convention
- **Test matrix**: Python 3.11 & 3.13

### Version Management
- Version from git tags via `hatch-vcs`
- Release: Create GitHub release with tag `vX.X.X`
- Follows **Semantic Versioning**

## Project-Specific Patterns

### Basic Usage
```python
from cell_annotator import CellAnnotator

# Annotate across multiple samples
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
- Marker genes computed via `scanpy.tl.rank_genes_groups()`
- Results stored in `adata.obs[cell_type_key]` (default: `"cell_type_predicted"`)
- Confidence scores in `adata.obs[f"{cell_type_key}_confidence"]`

## Common Gotchas

1. **Hatch for testing**: Always use `hatch test`, never standalone `pytest`. CI matches hatch test matrix.
2. **API keys**: Must be set as env vars or in `.env` file. Package auto-loads via python-dotenv.
3. **Provider packages**: Install provider extras (`pip install cell-annotator[openai]`) to use specific LLMs.
4. **Real LLM tests**: Use `@pytest.mark.real_llm_query` and skip in CI unless explicitly enabled.
5. **Marker gene filtering**: Package automatically filters marker genes to genes present in `adata.var_names`.
6. **Pre-commit conflicts**: Use `git pull --rebase` to integrate pre-commit.ci fixes.
7. **Line length**: Ruff set to 120 chars, but keep docstrings readable (~80 chars per line).

## Related Resources

- **Contributing guide**: `docs/contributing.md`
- **Tutorials**: `docs/notebooks/tutorials/`
- **OpenAI structured outputs**: https://platform.openai.com/docs/guides/structured-outputs
- **scanpy docs**: https://scanpy.readthedocs.io/
- **Pydantic docs**: https://docs.pydantic.dev/
