# AGENTS.md — CellAnnotator

CellAnnotator is a Python package for automated cell type annotation of scRNA-seq data using Large Language Models. It is provider-agnostic (OpenAI, Anthropic Claude, Google Gemini) and part of the scverse ecosystem.
Key frameworks: scanpy/AnnData, Pydantic, python-dotenv, rich, and the provider SDKs `openai`, `anthropic`, `google-genai`.

## Trust Order

When sources disagree:
1. PR description and changed code
2. This file (`AGENTS.md`)
3. `REVIEW_GUIDE.md`
4. Tests and fixtures
5. Public docs in `docs/` and `README.md`

Every fact should have one owner. This file owns invariants and the reference table below.

## Where To Find What

| Topic | Source of truth |
|-------|----------------|
| User-facing overview, installation, quickstart | `README.md` |
| Contributor setup, environments, docs build, release workflow | `docs/contributing.md` |
| Public API reference | `docs/api.md` and autosummary under `docs/generated/` |
| Tutorials and example notebooks | `docs/notebooks/tutorials/` |
| PR review workflow and risk areas | `REVIEW_GUIDE.md` |
| Test fixtures (providers, example AnnData) | `tests/conftest.py`, `tests/utils.py` |
| LLM prompt templates | `src/cell_annotator/_prompts.py` |
| Structured LLM response schemas | `src/cell_annotator/_response_formats.py` |
| Provider wiring (OpenAI / Anthropic / Gemini) | `src/cell_annotator/model/_providers.py`, `llm_interface.py`, `base_annotator.py` |
| Release notes | GitHub Releases (<https://github.com/quadbio/cell-annotator/releases>) |

## Critical Invariants

- **Data privacy.** Only cluster marker gene *names*, plus `species`, `tissue`, and (optionally) `stage`, are sent to LLM providers. Raw expression values, cell-level data, and user metadata must never cross the provider boundary. Any PR that broadens what is sent is a hard red flag.
- **Structured outputs only.** All LLM calls round-trip through Pydantic models in `_response_formats.py` (e.g. `CellTypeListOutput`, `ExpectedMarkerGeneOutput`, `CellTypeMappingOutput`). Never parse free-form LLM text in production paths; renaming or removing fields breaks user-visible contracts.
- **Optional provider extras.** `openai`, `anthropic`, and `google-genai` are optional extras. Provider-specific imports must be guarded and must fail with a clear message when the extra is missing; they must not leak into import-time code paths of the top-level package.
- **API keys stay in the environment.** Keys are loaded from env vars or a local `.env` via python-dotenv (`APIKeyManager`). They must never be logged, serialized, or included in error messages / exceptions.
- **Marker gene filtering.** Expected marker gene lists returned by the LLM are filtered to genes present in `adata.var_names` before downstream use. Removing this step silently changes annotation behavior on datasets with limited feature sets (e.g. spatial).
- **AnnData output contract.** Results live in `adata.obs[cell_type_key]` (default `"cell_type_predicted"`) and `adata.obs[f"{cell_type_key}_confidence"]`. Renaming these keys breaks user notebooks and tutorials.
- **Real-LLM tests are gated.** Tests that hit a live provider are marked `@pytest.mark.real_llm_query` and are skipped in CI unless explicitly enabled. Prefer mocked provider fixtures; minimize Anthropic usage (it is the most expensive provider).
- **Tests mirror source layout.** `src/cell_annotator/X/` → `tests/X/`. New modules get matching test files.
- **Public API surface is minimal.** Only symbols in `src/cell_annotator/__init__.py`'s `__all__` are public. Internal helpers, Pydantic schemas, and prompt templates are private — do not re-export from the top-level package.

## Development Commands

Python 3.11 through 3.14.

```bash
hatch test                        # run tests (highest Python)
hatch test --all                  # full matrix
hatch run docs:build              # build Sphinx docs
hatch run docs:open               # open built docs
pre-commit run --all-files        # lint and format
```

Focused runs (with `uv`):
```bash
uv run pytest tests/model
uv run pytest tests/test_utils.py
uv run pytest -m "not real_llm_query"   # skip live-provider tests
```
