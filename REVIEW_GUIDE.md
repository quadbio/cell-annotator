# CellAnnotator Review Guide

This file is the canonical, agent-neutral source of truth for automated PR review in this repo.
It is written for **agents performing PR reviews on GitHub** — use the imperative voice and be concrete.

**Scope: review only.** Your job is to produce review comments and suggestions on the PR. Do **not** push commits, modify files, or apply fixes yourself. Any changes are the author's call. Flag issues, ask questions, and suggest concrete diffs in comments when helpful — but leave the decision and the edits to the user.

Use `AGENTS.md` for architecture, invariants, and commands. Use this guide for review workflow, risk areas, testing checks, documentation-impact checks, and test lookup.

## Review-First Workflow

1. Read the PR body first when it is present.
2. Check CI status (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate any test or lint failures before commenting.
3. Identify changed modules and map them to matching tests.
4. Check whether the change touches a repo invariant from `AGENTS.md`.
5. Prioritize data-privacy, provider-contract, and user-facing AnnData-key regressions over style feedback.
6. Verify that docs (human- and agent-facing) did not become stale — see [Documentation Impact](#documentation-impact).

## High-Risk Areas

- **Data privacy boundary.** Any change that broadens what is sent to an LLM provider beyond marker gene *names* + `species` / `tissue` / `stage` is a hard red flag. Audit new prompt construction code in `_prompts.py` and anything that builds payloads in `base_annotator.py` / `llm_interface.py`.
- **Prompt template wording** (`src/cell_annotator/_prompts.py`): silent wording changes measurably shift annotation outputs. Treat these like behavior changes; they warrant a GitHub-release note.
- **Pydantic response schemas** (`src/cell_annotator/_response_formats.py`): renaming, removing, or retyping fields breaks parsing of structured LLM outputs and is a serialized-contract change.
- **Provider SDK integration** (`src/cell_annotator/model/_providers.py`, `llm_interface.py`, `base_annotator.py`): auth, retry, structured-output parsing, and error surfacing differ per provider; regressions are easy to miss with only one provider mocked.
- **Optional-extras guards.** `openai`, `anthropic`, `google-genai`, `colorspacious`, `rapids-singlecell` are optional. Imports must be guarded and must not leak into top-level `__init__.py` paths. Flag any unguarded `import openai` (or similar) outside of provider modules.
- **Marker-gene filtering** (`src/cell_annotator/utils.py`): the step that filters expected marker genes to `adata.var_names` is load-bearing for spatial / limited-feature datasets. Regressions here silently change what reaches the LLM.
- **API key handling** (`src/cell_annotator/model/_api_keys.py`): keys must never be logged, included in exceptions, or serialized. Audit changes to error messages and `repr` / logging in this area.
- **AnnData output keys.** Changes to `cell_type_key` defaults or the confidence column naming break user notebooks and tutorials. Require explicit call-out in the PR body.
- **Public API surface.** New symbols re-exported from `src/cell_annotator/__init__.py` commit the project to an API; flag unnecessary re-exports and prefer keeping internals private.

## Changed-Path Test Lookup

Test directories mirror `src/cell_annotator/` — changes to `src/cell_annotator/X/` have matching tests under `tests/`.

| Changed path | Look at |
|---|---|
| `src/cell_annotator/model/` | `tests/model/` (per-class files: `test_cell_annotator.py`, `test_sample_annotator.py`, `test_base_annotator.py`, `test_llm_interface.py`, `test_obs_beautifier.py`) |
| `src/cell_annotator/utils.py` | `tests/test_utils.py` |
| `src/cell_annotator/check.py` | `tests/test_check.py` |
| `src/cell_annotator/_response_formats.py` or `_prompts.py` | downstream: `tests/model/test_base_annotator.py`, `tests/model/test_llm_interface.py` |
| Cross-cutting fixture changes | `tests/conftest.py`, `tests/utils.py` |

## Testing

Apply these checks whenever the PR touches code or tests.

**New code.** Confirm that new behavior is covered by tests.
- Reuse fixtures from `tests/conftest.py` and `tests/utils.py` rather than creating parallel ones.
- Prefer `pytest.mark.parametrize` over many near-identical tests; the existing `provider_name` fixture already parametrizes across available providers.
- Favor few meaningful tests over many redundant ones; flag low-value tests that only duplicate existing coverage.

**Real-LLM tests.** Keep live-provider usage minimal.
- Live tests must be marked `@pytest.mark.real_llm_query` and must be skipped by default in CI.
- Prefer mocked provider fixtures for unit tests.
- Anthropic is the most expensive provider — do not add Anthropic-only real-LLM tests without justification.

**Failing tests.** If CI is red, do not wave it through.
- Inspect which tests fail and why (`gh pr checks`, `gh run view --log-failed`).
- Distinguish critical regressions (data-privacy, provider-contract, AnnData-key, structured-output) from trivial or flaky failures.
- Surface critical failures back to the author and ask them to fix before merge.

**Modified tests.** Scrutinize *how* existing tests were changed.
- PRs that only relax thresholds, remove assertions, delete cases, or loosen `parametrize` matrices are a red flag — tests-working-around-tests defeats the purpose.
- Require an explicit justification in the PR body for any weakened assertion; do not accept silently.

## Documentation Impact

A single behavioral or API change often touches docs in multiple places. Check both audiences and ask the author to update what is stale. Point to the **owning file** for each topic rather than duplicating content in your review.

**Human-facing docs (`docs/`, Sphinx/RTD).**
- Public API signature or symbol changes → `docs/api.md` and autosummary entries under `docs/generated/`; any prose referencing the symbol.
- Contributor workflow, environment, or release changes → `docs/contributing.md`.
- Installation / quickstart changes → `README.md`.
- Tutorials under `docs/notebooks/tutorials/` → flag stale imports, outputs, or API usage (e.g. renamed kwargs, changed AnnData keys).

**Agent-facing docs (repo root and `.github/`).**
- Invariants or development commands changed → `AGENTS.md` (Critical Invariants, Development Commands).
- Review workflow, risk areas, or testing conventions changed → `REVIEW_GUIDE.md` (this file).
- Repo structure, new top-level docs, or moved pointers → `AGENTS.md` "Where To Find What" table, `CLAUDE.md`, `.github/copilot-instructions.md`.

**User-visible behavior changes** (prompt wording, default model, AnnData keys, public API) → note explicitly in the PR body so a GitHub Release entry can reference it. This repo does not maintain a hand-written `CHANGELOG.md`; release notes live in GitHub Releases.

If behavior changes but the relevant docs do not, call it out explicitly in the review and request the update.

## Review Checklist

- Does the change preserve the invariants in `AGENTS.md`?
- Does CI pass, and were any failures investigated? (See [Testing](#testing).)
- Is test coverage adequate and non-redundant, and are modified tests not simply weakened? (See [Testing](#testing).)
- Does it alter what is sent to LLM providers, prompt wording, or structured-output schemas?
- Does it change AnnData output keys or default `cell_type_key` behavior?
- Are optional provider imports still properly guarded?
- Are all affected human- and agent-facing docs updated? (See [Documentation Impact](#documentation-impact).)
- Is the PR scope tight — no unrelated changes bundled in — and is the public API surface kept minimal?

## PR Metadata

This repo uses a structured PR template.

Reviewers and agents should treat these sections as the preferred summary surface:
- summary
- behavior or invariants changed
- tests run
- reviewer focus
- context
- open questions or follow-ups
