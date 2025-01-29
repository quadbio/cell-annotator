# CellAnnotator

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/cell-annotator/test.yaml?branch=main

[badge-docs]: https://img.shields.io/readthedocs/{{ cookiecutter.project_name }}

A tool to annotate cell types based on marker genes using OpenAI models. Inspired by [Hou et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02235-4) and [https://github.com/VPetukhov/GPTCellAnnotator](https://github.com/VPetukhov/GPTCellAnnotator).

## Key features

-   Automatically annotate cells including type, state and confidence fields.
-   Generate consistent annotations across samples of your study.
-   Optionally infuse prior knowledge by providing information about your biological system.
-   Retrieve reliable results thanks to [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs)

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/cell-annotator.git@main
```

## Getting started

After installation, head over to OpenAI to generate your API key: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key

Keep this key private and don't share it with anyone. `CellAnnotator` will try to read the key as an environmental variable - either expose it to the enrivonment yourself, or store it as an `.env` file anywhere within the repository where you conduct your analysis and plan to run `CellAnnotator`. The package will then use [dotenv](https://pypi.org/project/python-dotenv/) to export the key from the `env` file as an environmental variable.

## Contact

If you found a bug, please use the [issue tracker][].

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[issue tracker]: https://github.com/quadbio/cell-annotator/issues
[tests]: https://github.com/quadbio/cell-annotator/actions/workflows/test.yml

[documentation]: https://{{ cookiecutter.project_name }}.readthedocs.io
