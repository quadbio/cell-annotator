"""Reference providers for cell type and marker gene annotation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cell_annotator._logging import logger
from cell_annotator._response_formats import CellTypeListOutput, ExpectedMarkerGeneOutput
from cell_annotator.check import check_deps
from cell_annotator.model.base_annotator import BaseAnnotator


@dataclass
class ProviderCapabilities:
    """Declares what a provider can do."""

    provides_cell_types: bool
    provides_markers: bool
    requires_cell_types_input: bool  # True if provider needs cell types to find markers


@dataclass
class CellTypeInfo:
    """Information about a cell type from a reference provider."""

    name: str
    ontology_id: str | None = None
    description: str | None = None
    synonyms: list[str] | None = None
    parent_types: list[str] | None = None
    markers: list[str] | None = None  # May be None if provider doesn't support markers


class CellTypeReferenceProvider(ABC):
    """Abstract base class for cell type reference providers."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Declare what this provider can do."""
        pass

    @abstractmethod
    def get_cell_types(self, tissue: str, species: str, stage: str) -> list[CellTypeInfo]:
        """Get cell types for the given context.

        Parameters
        ----------
        tissue : str
            The tissue type
        species : str
            The species
        stage : str
            The developmental stage

        Returns
        -------
        list[CellTypeInfo]
            List of cell type information
        """
        ...

    @abstractmethod
    def get_markers(
        self, cell_types: list[str], tissue: str, species: str, stage: str, n_markers: int = 50
    ) -> dict[str, list[str]]:
        """Get marker genes for the specified cell types.

        Parameters
        ----------
        cell_types : list[str]
            List of cell type names
        tissue : str
            The tissue type
        species : str
            The species
        stage : str
            The developmental stage
        n_markers : int, optional
            Maximum number of markers per cell type, by default 50

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping cell type names to marker gene lists
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available/installed.

        Returns
        -------
        bool
            True if the provider is available
        """
        ...


class LLMProvider(CellTypeReferenceProvider):
    """LLM-based cell type and marker gene provider (current approach)."""

    def __init__(self, annotator: BaseAnnotator):
        self.annotator = annotator

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this LLM provider.

        Returns
        -------
        ProviderCapabilities
            Provider capabilities supporting both cell types and markers
        """
        return ProviderCapabilities(provides_cell_types=True, provides_markers=True, requires_cell_types_input=False)

    def get_cell_types(self, tissue: str, species: str, stage: str) -> list[CellTypeInfo]:
        """Does the full current LLM workflow: cell types + markers."""
        logger.info("Querying cell types via LLM.")
        res_types = self.annotator.query_llm(
            instruction=self.annotator.prompts.get_cell_type_prompt(),
            response_format=CellTypeListOutput,
        )

        cell_types = res_types.cell_type_list

        # Get markers for these cell types
        logger.info("Querying cell type markers via LLM.")
        marker_gene_prompt = self.annotator.prompts.get_cell_type_marker_prompt(
            n_markers=5,  # Default, will be overridden by caller if needed
            cell_types=cell_types,
            var_names=None,  # Can be enhanced later
        )

        res_markers = self.annotator.query_llm(
            instruction=marker_gene_prompt,
            response_format=ExpectedMarkerGeneOutput,
        )

        # Create CellTypeInfo objects with both names and markers
        marker_dict = {
            cell_type_markers.cell_type_name: cell_type_markers.expected_marker_genes
            for cell_type_markers in res_markers.expected_markers_per_cell_type
        }

        return [CellTypeInfo(name=cell_type, markers=marker_dict.get(cell_type, [])) for cell_type in cell_types]

    def get_markers(
        self, cell_types: list[str], tissue: str, species: str, stage: str, n_markers: int = 50
    ) -> dict[str, list[str]]:
        """Get marker genes for the specified cell types."""
        return self.get_markers_for_cell_types(cell_types, tissue, species, n_markers)

    def get_markers_for_cell_types(
        self, cell_types: list[str], tissue: str, species: str, n_markers: int = 5
    ) -> dict[str, list[str]]:
        """Just gets markers for provided cell types (for composition scenarios)."""
        logger.info("Querying markers for provided cell types via LLM.")
        marker_gene_prompt = self.annotator.prompts.get_cell_type_marker_prompt(
            n_markers=n_markers,
            cell_types=cell_types,
            var_names=None,  # Can be enhanced later
        )

        res_markers = self.annotator.query_llm(
            instruction=marker_gene_prompt,
            response_format=ExpectedMarkerGeneOutput,
        )

        return {
            cell_type_markers.cell_type_name: cell_type_markers.expected_marker_genes
            for cell_type_markers in res_markers.expected_markers_per_cell_type
        }

    def is_available(self) -> bool:
        """Check if LLM provider is available."""
        try:
            # Check if annotator has basic required methods
            return hasattr(self.annotator, "query_llm") and hasattr(self.annotator, "prompts")
        except (AttributeError, TypeError):
            return False


class BiontyCellOntologyProvider(CellTypeReferenceProvider):
    """Uses bionty Cell Ontology for cell type names only."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this Cell Ontology provider.

        Returns
        -------
        ProviderCapabilities
            Provider capabilities supporting cell types but not markers
        """
        return ProviderCapabilities(provides_cell_types=True, provides_markers=False, requires_cell_types_input=False)

    def _ensure_lamindb_initialized(self) -> None:
        """Ensure lamindb is initialized with bionty module for this session."""
        try:
            # First, just try to use bionty directly
            import bionty as bt

            # Test if we can create a simple public instance
            bt.CellType.public(organism="all")
            return  # Already working!
        except (ImportError, AttributeError, RuntimeError, ValueError):
            pass  # Need to initialize

        try:
            import tempfile

            import lamindb as ln

            # Create a temporary directory for lamindb if needed
            if not hasattr(self, "_lamindb_storage"):
                self._lamindb_storage = tempfile.mkdtemp(prefix="cell_annotator_bionty_")

            # Initialize lamindb with bionty module
            ln.setup.init(storage=self._lamindb_storage, modules="bionty")
            logger.info("Initialized lamindb with bionty module")

        except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to initialize lamindb with bionty: %s", str(e))
            # Don't raise - let the actual method handle the failure
            pass

    def get_cell_types(self, tissue: str, species: str, stage: str) -> list[CellTypeInfo]:
        """Get cell types from Cell Ontology via bionty."""
        check_deps("bionty")

        # Ensure lamindb is properly initialized
        self._ensure_lamindb_initialized()

        try:
            import bionty as bt
        except ImportError as e:
            raise RuntimeError("bionty is required for Cell Ontology queries.") from e

        logger.info("Querying cell types from Cell Ontology via bionty.")

        # Get Cell Ontology public interface
        celltype_pub = bt.CellType.public(organism="all")

        # Search for tissue-related cell types
        search_terms = [tissue, f"{tissue} cell", f"{species} {tissue}"]

        results = []
        for term in search_terms:
            try:
                search_results = celltype_pub.search(term, limit=50)
                if hasattr(search_results, "df") and not search_results.df.empty:
                    df = search_results.df
                    for _, row in df.iterrows():
                        cell_type_info = CellTypeInfo(
                            name=row.get("name", ""),
                            ontology_id=row.get("ontology_id", None),
                            description=row.get("description", None),
                            synonyms=row.get("synonyms", "").split("|") if row.get("synonyms") else None,
                        )
                        if cell_type_info.name:  # Only add if name is not empty
                            results.append(cell_type_info)

                    if results:  # If we found results, break
                        break
            except (AttributeError, KeyError, ValueError) as e:
                logger.debug("Search term '%s' failed: %s", term, str(e))
                continue

        # Remove duplicates while preserving order
        seen_names = set()
        unique_results = []
        for result in results:
            if result.name not in seen_names:
                seen_names.add(result.name)
                unique_results.append(result)

        if not unique_results:
            logger.warning("No cell types found in Cell Ontology for tissue '%s' and species '%s'", tissue, species)
        else:
            logger.info("Found %d cell types from Cell Ontology.", len(unique_results))

        return unique_results

    def get_markers(
        self, cell_types: list[str], tissue: str, species: str, stage: str, n_markers: int = 50
    ) -> dict[str, list[str]]:
        """Cell Ontology doesn't provide markers."""
        raise NotImplementedError("Cell Ontology provider doesn't support marker genes")

    def get_markers_for_cell_types(
        self, cell_types: list[str], tissue: str, species: str, n_markers: int = 5
    ) -> dict[str, list[str]]:
        """Cell Ontology doesn't provide markers."""
        raise NotImplementedError("Cell Ontology provider doesn't support marker genes")

    def is_available(self) -> bool:
        """Check if bionty is available."""
        try:
            check_deps("bionty")
            return True
        except RuntimeError:
            return False


class BiontyCellMarkerProvider(CellTypeReferenceProvider):
    """Uses bionty CellMarker database for both cell types and markers."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this CellMarker provider.

        Returns
        -------
        ProviderCapabilities
            Provider capabilities supporting both cell types and markers
        """
        return ProviderCapabilities(provides_cell_types=True, provides_markers=True, requires_cell_types_input=False)

    def _ensure_lamindb_initialized(self) -> None:
        """Ensure lamindb is initialized with bionty module for this session."""
        try:
            # First, just try to use bionty directly
            import bionty as bt

            # Test if we can create a simple public instance
            bt.CellType.public(organism="all")
            return  # Already working!
        except (ImportError, AttributeError, RuntimeError, ValueError):
            pass  # Need to initialize

        try:
            import tempfile

            import lamindb as ln

            # Create a temporary directory for lamindb if needed
            if not hasattr(self, "_lamindb_storage"):
                self._lamindb_storage = tempfile.mkdtemp(prefix="cell_annotator_bionty_")

            # Initialize lamindb with bionty module
            ln.setup.init(storage=self._lamindb_storage, modules="bionty")
            logger.info("Initialized lamindb with bionty module")

        except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to initialize lamindb with bionty: %s", str(e))
            # Don't raise - let the actual method handle the failure
            pass

    def get_cell_types(self, tissue: str, species: str, stage: str) -> list[CellTypeInfo]:
        """Get cell types and markers from CellMarker database via bionty."""
        check_deps("bionty")

        # Ensure lamindb is properly initialized
        self._ensure_lamindb_initialized()

        try:
            import bionty as bt
        except ImportError as e:
            raise RuntimeError("bionty is required for CellMarker database queries.") from e

        logger.info("Querying cell types and markers from CellMarker database via bionty.")

        # Get CellMarker public interface for the specified organism
        from typing import Literal, cast

        organism_map = {"human": "human", "mouse": "mouse", "homo sapiens": "human", "mus musculus": "mouse"}
        organism_name = cast(Literal["human", "mouse"], organism_map.get(species.lower(), "human"))  # Default to human

        try:
            marker_pub = bt.CellMarker.public(organism=organism_name)
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            logger.warning("Failed to initialize CellMarker for organism '%s': %s", organism_name, str(e))
            return []

        # Search for tissue-related markers
        search_terms = [tissue, f"{tissue} cell"]

        results = []
        cell_type_markers = {}  # Track markers per cell type

        for term in search_terms:
            try:
                search_results = marker_pub.search(term, limit=100)
                if hasattr(search_results, "df") and not search_results.df.empty:
                    df = search_results.df
                    for _, row in df.iterrows():
                        cell_type_name = row.get("name", "")
                        gene_symbol = row.get("gene_symbol", "")

                        if cell_type_name and gene_symbol:
                            # Group markers by cell type
                            if cell_type_name not in cell_type_markers:
                                cell_type_markers[cell_type_name] = []
                            cell_type_markers[cell_type_name].append(gene_symbol)
            except (AttributeError, KeyError, ValueError) as e:
                logger.debug("Search term '%s' failed: %s", term, str(e))
                continue

        # Convert to CellTypeInfo objects
        for cell_type_name, markers in cell_type_markers.items():
            # Remove duplicates and limit markers
            unique_markers = list(dict.fromkeys(markers))  # Preserves order
            cell_type_info = CellTypeInfo(name=cell_type_name, markers=unique_markers)
            results.append(cell_type_info)

        if not results:
            logger.warning(
                "No cell types found in CellMarker database for tissue '%s' and species '%s'", tissue, species
            )
        else:
            logger.info("Found %d cell types with markers from CellMarker database.", len(results))

        return results

    def get_markers(
        self, cell_types: list[str], tissue: str, species: str, stage: str, n_markers: int = 50
    ) -> dict[str, list[str]]:
        """Get marker genes for the specified cell types."""
        return self.get_markers_for_cell_types(cell_types, tissue, species, n_markers)

    def get_markers_for_cell_types(
        self, cell_types: list[str], tissue: str, species: str, n_markers: int = 5
    ) -> dict[str, list[str]]:
        """Get markers for specific cell types from CellMarker database."""
        check_deps("bionty")

        # Ensure lamindb is properly initialized
        self._ensure_lamindb_initialized()

        try:
            import bionty as bt
        except ImportError as e:
            raise RuntimeError("bionty is required for CellMarker database queries.") from e

        logger.info("Querying markers for specific cell types from CellMarker database.")

        # Get CellMarker public interface
        from typing import Literal, cast

        organism_map = {"human": "human", "mouse": "mouse", "homo sapiens": "human", "mus musculus": "mouse"}
        organism_name = cast(Literal["human", "mouse"], organism_map.get(species.lower(), "human"))  # Default to human

        try:
            marker_pub = bt.CellMarker.public(organism=organism_name)
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            logger.warning("Failed to initialize CellMarker for organism '%s': %s", organism_name, str(e))
            return {}

        marker_dict = {}

        for cell_type in cell_types:
            try:
                # Search for this specific cell type
                search_results = marker_pub.search(cell_type, limit=n_markers * 2)  # Get extra to filter
                markers = []

                if hasattr(search_results, "df") and not search_results.df.empty:
                    df = search_results.df
                    for _, row in df.iterrows():
                        gene_symbol = row.get("gene_symbol", "")
                        if gene_symbol and gene_symbol not in markers:
                            markers.append(gene_symbol)
                            if len(markers) >= n_markers:
                                break

                marker_dict[cell_type] = markers[:n_markers]

            except (AttributeError, KeyError, ValueError) as e:
                logger.debug("Failed to get markers for cell type '%s': %s", cell_type, str(e))
                marker_dict[cell_type] = []

        return marker_dict

    def is_available(self) -> bool:
        """Check if bionty is available."""
        try:
            check_deps("bionty")
            return True
        except RuntimeError:
            return False


class ReferenceQueryOrchestrator:
    """Orchestrates queries across different reference providers."""

    def __init__(
        self,
        primary_provider: CellTypeReferenceProvider,
        marker_provider: CellTypeReferenceProvider | None = None,
    ):
        self.primary_provider = primary_provider
        self.marker_provider = marker_provider

    def get_cell_types_and_markers(
        self, tissue: str, species: str, stage: str, n_markers: int = 5
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Get cell types and markers using the configured providers."""
        # Get cell types from primary provider
        cell_type_infos = self.primary_provider.get_cell_types(tissue, species, stage)
        cell_types = [info.name for info in cell_type_infos]

        # Get markers
        if self.primary_provider.capabilities.provides_markers:
            # Primary provider has markers already
            markers = {info.name: info.markers or [] for info in cell_type_infos}
        elif self.marker_provider and self.marker_provider.capabilities.provides_markers and cell_types:
            # Use secondary provider for markers (only if we have cell types to query)
            markers = self.marker_provider.get_markers(cell_types, tissue, species, stage, n_markers)
        else:
            # No marker information available or no cell types to query
            markers = {cell_type: [] for cell_type in cell_types}

        return cell_types, markers


def get_reference_provider(provider_name: str, annotator: BaseAnnotator | None = None) -> CellTypeReferenceProvider:
    """Factory function to create reference providers."""
    if provider_name == "llm":
        if annotator is None:
            raise ValueError("LLM provider requires an annotator instance")
        return LLMProvider(annotator)
    elif provider_name == "bionty_cellontology":
        return BiontyCellOntologyProvider()
    elif provider_name == "bionty_cellmarker":
        return BiontyCellMarkerProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def create_orchestrator(reference_provider: str, annotator: BaseAnnotator | None = None) -> ReferenceQueryOrchestrator:
    """Create orchestrator for the specified reference provider strategy."""
    if "+" in reference_provider:
        # Composite provider (e.g., "bionty_cellontology+llm")
        parts = reference_provider.split("+", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid composite provider format: {reference_provider}")

        primary_name, marker_name = parts
        primary_provider = get_reference_provider(primary_name, annotator)
        marker_provider = get_reference_provider(marker_name, annotator)

        return ReferenceQueryOrchestrator(primary_provider, marker_provider)
    else:
        # Single provider
        provider = get_reference_provider(reference_provider, annotator)
        return ReferenceQueryOrchestrator(provider)
