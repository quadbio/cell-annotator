# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][]. Full commit history is available in the [commit logs][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
[commit logs]: https://github.com/quadbio/cell-annotator/commits

## Version 0.2

### Unreleased

### 0.2.0 (2025-07-26)

#### Added
- Added a generic LLM backend that supports OpenAI, Claude and Gemini models {pr}`53`
- Add the possibility to provide the current gene set when querying expected marker genes {pr}`53`
- Add the possibility to filter expected marker genes to those presend in AnnData {pr}`53`
- Add a new tutorial on spatial data annotation {pr}`53`
- Added and improved tests for the new classes (e.g. ObsBeautifier, LLMBackend, etc) {pr}`53`
- For each backend, add a small `test_query` method which can be used for diagnostics {pr}`53`

#### Changed
- Moved the `reorder_and_color` utility into a new class: `ObsBeautifier` {pr}`54`
- Improved class representations throughout the package  {pr}`53`

#### Fixed
- Fix the `ObsBeautifier` modifying cluster colors when only their order should be updated {pr}`54`

## Version 0.1

### 0.1.5 (2025-07-24)

#### Changed
- Update tutorials to use `gpt-4.1` {pr}`51`

### 0.1.4 (2025-03-28)

#### Added

- Use `rapids_singlecell`, `cupy` and `cuml` to accelerate cluster marker computation on GPUs {pr}`37`.
- Add the possibility to softly enforce adherence to expected cell types {pr}`42`.

#### Changed

- Run cluster label harmonization also for a single sample {pr}`37`.
- Re-format prompts into a dataclass {pr}`42`.

#### Fixed

- Fixed a bug with integer sample labels {pr}`37`.

### 0.1.3 (2025-02-07)

#### Added

- Added tests for the single-sample case {pr}`29`.
- Refer to issues and PRs with sphinx {pr}`30`.

#### Removed

- Removed `tenacity` for query retries {pr}`28`.

#### Fixed

- Fixed `_get_annotation_summary_string` for the single-sample case {pr}`29`.
- Fixed the expected cell type marker test by adding additional marker genes {pr}`28`.

### 0.1.2 (2025-01-29)

#### Added

- Update the documentation, in particular the installation instructions.

### 0.1.1 (2025-01-29)

#### Added

- Initial push to PyPI

### 0.1.0 (2025-01-29)

Initial package release
