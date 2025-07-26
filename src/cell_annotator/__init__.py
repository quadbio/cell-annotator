from importlib.metadata import version

from ._logging import logger
from .check import check_deps
from .model._api_keys import APIKeyManager
from .model.base_annotator import BaseAnnotator
from .model.cell_annotator import CellAnnotator
from .model.sample_annotator import SampleAnnotator

__all__ = ["CellAnnotator", "SampleAnnotator", "BaseAnnotator", "logger", "check_deps", "APIKeyManager"]
__version__ = version("cell-annotator")
