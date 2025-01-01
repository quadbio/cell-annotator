from importlib.metadata import version

from .cell_annotator import CellAnnotator
from .sample_annotator import SampleAnnotator

__all__ = ["CellAnnotator", "SampleAnnotator"]

__version__ = version("cell-annotator")
