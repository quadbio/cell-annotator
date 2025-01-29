# CellAnnotator

[![Tests][badge-tests]][tests]
[![Coverage][badge-coverage]][coverage]
[![Documentation][badge-docs]][documentation]
[![Pre-commit.ci][badge-pre-commit]][pre-commit]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/cell-annotator/test.yaml?branch=main
[badge-coverage]: https://codecov.io/gh/quadbio/cell-annotator/branch/main/graph/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/cell-annotator
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/quadbio/cell-annotator/main.svg

A tool to annotate cell types based on marker genes using OpenAI models.

## Key features

-   Automatically annotate cells including type, state and confidence fields.
-   Generate consistent annotations across samples of your study.
-   Optionally infuse prior knowledge by providing information about your biological system.
-   Retrieve reliable results thanks to [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs)
-   Use pre-integration cell type labels to either score your integration quality (e.g. [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/)) or to guide your integration effort (e.g. [scPoli](https://docs.scarches.org/en/latest/), [scANVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html))

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/cell-annotator.git@main
```

## Getting started

After installation, head over to OpenAI to generate your [API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

Keep this key private and don't share it with anyone. `CellAnnotator` will try to read the key as an environmental variable - either expose it to the environment yourself, or store it as an `.env` file anywhere within the repository where you conduct your analysis and plan to run `CellAnnotator`. The package will then use [dotenv](https://pypi.org/project/python-dotenv/) to export the key from the `env` file as an environmental variable.

## Credits

This tool was inspired by [Hou et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02235-4) and [https://github.com/VPetukhov/GPTCellAnnotator](https://github.com/VPetukhov/GPTCellAnnotator).

## Contact

If you found a bug, please use the [issue tracker][].

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[issue tracker]: https://github.com/quadbio/cell-annotator/issues
[tests]: https://github.com/quadbio/cell-annotator/actions/workflows/test.yaml
[coverage]: https://codecov.io/gh/quadbio/cell-annotator
[documentation]: https://cell-annotator.readthedocs.io
[pre-commit]: https://results.pre-commit.ci/latest/github/quadbio/cell-annotator/main
