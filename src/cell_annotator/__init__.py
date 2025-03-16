from importlib.metadata import version

from ._logging import logger
from .base_annotator import BaseAnnotator
from .cell_annotator import CellAnnotator
from .sample_annotator import SampleAnnotator

__all__ = ["CellAnnotator", "SampleAnnotator", "BaseAnnotator", "logger"]
__version__ = version("cell-annotator")
