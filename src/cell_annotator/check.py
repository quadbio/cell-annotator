"""Dependency checking."""

import importlib
import types
from importlib.metadata import version

from packaging.version import Version, parse


class Checker:
    """
    Checks availability and version of a Python module dependency.

    Adapted from the scGLUE package: https://github.com/gao-lab/GLUE

    Parameters
    ----------
    name
        Name of the dependency
    package_name
        Name of the package to check version for (if different from module name)
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
        self, name: str, package_name: str | None = None, vmin: str | None = None, install_hint: str | None = None
    ) -> None:
        self.name = name
        self.package_name = package_name or name
        self.vmin: Version | None = parse(vmin) if vmin else None
        vreq = f" (>={self.vmin})" if self.vmin else ""
        self.vreq_hint = f"This function relies on {self.name}{vreq}."
        self.install_hint = install_hint

    def check(self) -> None:
        """Check if the dependency is available and meets the version requirement."""
        try:
            importlib.import_module(self.name)
        except ModuleNotFoundError as e:
            raise RuntimeError(" ".join(filter(None, [self.vreq_hint, self.install_hint]))) from e
        v = parse(version(self.package_name))
        if self.vmin and v < self.vmin:
            raise RuntimeError(
                " ".join(
                    [
                        self.vreq_hint,
                        f"Detected version is {v}.",
                        "Please install a newer version.",
                        self.install_hint or "",
                    ]
                )
            )


INSTALL_HINTS = types.SimpleNamespace(
    openai="To use OpenAI models, install with: pip install 'cell-annotator[openai]' or pip install openai>=1.66",
    google_genai="To use Google Gemini models, install with: pip install 'cell-annotator[gemini]' "
    "or pip install google-generativeai",
    anthropic="To use Anthropic Claude models, install with: pip install 'cell-annotator[anthropic]' "
    "or pip install anthropic",
    rapids_singlecell="To speed up analysis on GPU, install with: pip install 'cell-annotator[gpu]' "
    "or follow the guide from https://docs.rapids.ai/install/",
    cupy="To speed up GPU computations, install cuPy following the guide from https://docs.rapids.ai/install/",
    cuml="To speed up GPU machine learning, install cuML following the guide from https://docs.rapids.ai/install/",
)

CHECKERS = {
    "openai": Checker("openai", vmin="1.66", install_hint=INSTALL_HINTS.openai),
    "google-genai": Checker(
        "google.genai", package_name="google-generativeai", vmin=None, install_hint=INSTALL_HINTS.google_genai
    ),
    "anthropic": Checker("anthropic", vmin=None, install_hint=INSTALL_HINTS.anthropic),
    "rapids-singlecell": Checker(
        "rapids_singlecell", package_name="rapids-singlecell", vmin="0.12", install_hint=INSTALL_HINTS.rapids_singlecell
    ),
    "cupy": Checker("cupy", vmin=None, install_hint=INSTALL_HINTS.cupy),
    "cuml": Checker("cuml", vmin=None, install_hint=INSTALL_HINTS.cuml),
}


def check_deps(*args) -> None:
    """
    Check whether certain dependencies are installed.

    Parameters
    ----------
    args
        A list of dependencies to check
    """
    for item in args:
        if item not in CHECKERS:
            raise RuntimeError(f"Dependency '{item}' is not registered in CHECKERS.")
        CHECKERS[item].check()
