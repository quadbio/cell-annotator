from importlib.metadata import version

from .model import CellAnnotator

__all__ = ["CellAnnotator"]

__version__ = version("cell-annotator")
