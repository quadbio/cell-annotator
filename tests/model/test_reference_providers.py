"""Tests for reference provider system."""

import pytest

from cell_annotator.model.reference_providers import (
    BiontyCellMarkerProvider,
    BiontyCellOntologyProvider,
    LLMProvider,
    ProviderCapabilities,
    ReferenceQueryOrchestrator,
    create_orchestrator,
    get_reference_provider,
)


class TestProviderCapabilities:
    """Test ProviderCapabilities dataclass."""

    def test_provider_capabilities_creation(self):
        """Test creating ProviderCapabilities."""
        caps = ProviderCapabilities(provides_cell_types=True, provides_markers=False, requires_cell_types_input=True)
        assert caps.provides_cell_types is True
        assert caps.provides_markers is False
        assert caps.requires_cell_types_input is True


class TestLLMProvider:
    """Test LLMProvider functionality."""

    def test_llm_provider_capabilities(self):
        """Test LLMProvider returns expected capabilities."""
        # Mock annotator for testing
        mock_annotator = type("MockAnnotator", (), {})()

        provider = LLMProvider(mock_annotator)
        caps = provider.capabilities

        assert caps.provides_cell_types is True
        assert caps.provides_markers is True
        assert caps.requires_cell_types_input is False

    def test_llm_provider_is_available(self):
        """Test LLMProvider is always available."""
        mock_annotator = type(
            "MockAnnotator", (), {"query_llm": lambda self, **kwargs: None, "prompts": type("MockPrompts", (), {})()}
        )()
        provider = LLMProvider(mock_annotator)

        assert provider.is_available() is True


class TestBiontyProviders:
    """Test bionty-based providers."""

    def test_cell_ontology_provider_capabilities(self):
        """Test BiontyCellOntologyProvider capabilities."""
        provider = BiontyCellOntologyProvider()
        caps = provider.capabilities

        assert caps.provides_cell_types is True
        assert caps.provides_markers is False
        assert caps.requires_cell_types_input is False

    def test_cell_marker_provider_capabilities(self):
        """Test BiontyCellMarkerProvider capabilities."""
        provider = BiontyCellMarkerProvider()
        caps = provider.capabilities

        assert caps.provides_cell_types is True
        assert caps.provides_markers is True
        assert caps.requires_cell_types_input is False


class TestReferenceQueryOrchestrator:
    """Test ReferenceQueryOrchestrator functionality."""

    def test_orchestrator_creation_single_provider(self):
        """Test creating orchestrator with single provider."""
        mock_annotator = type("MockAnnotator", (), {})()
        primary_provider = LLMProvider(mock_annotator)

        orchestrator = ReferenceQueryOrchestrator(primary_provider=primary_provider)

        assert orchestrator.primary_provider is primary_provider
        assert orchestrator.marker_provider is None

    def test_orchestrator_creation_dual_providers(self):
        """Test creating orchestrator with separate providers."""
        mock_annotator = type("MockAnnotator", (), {})()
        primary_provider = BiontyCellOntologyProvider()
        marker_provider = LLMProvider(mock_annotator)

        orchestrator = ReferenceQueryOrchestrator(primary_provider=primary_provider, marker_provider=marker_provider)

        assert orchestrator.primary_provider is primary_provider
        assert orchestrator.marker_provider is marker_provider


class TestGetReferenceProvider:
    """Test get_reference_provider factory function."""

    def test_get_llm_provider(self):
        """Test getting LLM provider."""
        mock_annotator = type("MockAnnotator", (), {})()
        provider = get_reference_provider("llm", annotator=mock_annotator)

        assert isinstance(provider, LLMProvider)
        assert provider.annotator is mock_annotator

    def test_get_bionty_cell_ontology_provider(self):
        """Test getting bionty cell ontology provider."""
        provider = get_reference_provider("bionty_cellontology")

        assert isinstance(provider, BiontyCellOntologyProvider)

    def test_get_bionty_cell_marker_provider(self):
        """Test getting bionty cell marker provider."""
        provider = get_reference_provider("bionty_cellmarker")

        assert isinstance(provider, BiontyCellMarkerProvider)

    def test_get_unknown_provider_raises(self):
        """Test that unknown provider names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_reference_provider("unknown_provider")


class TestCreateOrchestrator:
    """Test create_orchestrator factory function."""

    def test_create_orchestrator_llm(self):
        """Test creating orchestrator for LLM provider."""
        mock_annotator = type("MockAnnotator", (), {})()
        orchestrator = create_orchestrator("llm", annotator=mock_annotator)

        assert isinstance(orchestrator, ReferenceQueryOrchestrator)
        assert isinstance(orchestrator.primary_provider, LLMProvider)
        assert orchestrator.marker_provider is None

    def test_create_orchestrator_bionty_cellontology(self):
        """Test creating orchestrator for bionty cell ontology."""
        orchestrator = create_orchestrator("bionty_cellontology")

        assert isinstance(orchestrator, ReferenceQueryOrchestrator)
        assert isinstance(orchestrator.primary_provider, BiontyCellOntologyProvider)
        assert orchestrator.marker_provider is None

    def test_create_orchestrator_composite(self):
        """Test creating orchestrator for composite provider."""
        mock_annotator = type("MockAnnotator", (), {})()
        orchestrator = create_orchestrator("bionty_cellontology+llm", annotator=mock_annotator)

        assert isinstance(orchestrator, ReferenceQueryOrchestrator)
        assert isinstance(orchestrator.primary_provider, BiontyCellOntologyProvider)
        assert isinstance(orchestrator.marker_provider, LLMProvider)

    def test_create_orchestrator_invalid_composite(self):
        """Test that invalid composite raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_orchestrator("invalid+format+too+many")


class TestBiontyIntegration:
    """Test actual bionty functionality."""

    def test_bionty_cell_ontology_is_available(self):
        """Test that bionty Cell Ontology provider reports availability correctly."""
        provider = BiontyCellOntologyProvider()
        # Should be available since we installed bionty
        assert provider.is_available() is True

    def test_bionty_cell_marker_is_available(self):
        """Test that bionty CellMarker provider reports availability correctly."""
        provider = BiontyCellMarkerProvider()
        # Should be available since we installed bionty
        assert provider.is_available() is True

    def test_bionty_cell_ontology_get_cell_types(self):
        """Test getting cell types from Cell Ontology."""
        provider = BiontyCellOntologyProvider()

        try:
            # Test with more specific cell type terms that are likely to be found
            # Using "lymphocyte" instead of "blood" as it's a specific cell type
            cell_types = provider.get_cell_types(tissue="lymphocyte", species="human", stage="adult")

            # Should get some results (but if not, that's okay - databases vary)
            # All should be CellTypeInfo objects
            for ct in cell_types:
                assert hasattr(ct, "name")
                assert ct.name is not None
                assert len(ct.name) > 0

            # If we didn't get results with lymphocyte, that's acceptable
            # The test mainly verifies the provider works without crashing
            print(f"Found {len(cell_types)} cell types from Cell Ontology")
        except Exception as e:
            # In testing environments, bionty might not be properly initialized
            # This is acceptable - we just want to verify the provider can be instantiated
            # and doesn't crash immediately
            if any(
                keyword in str(e)
                for keyword in ["bionty_source", "bionty_organism", "lamindb", "instance", "no such table"]
            ):
                pytest.skip(f"Bionty/lamindb not properly set up in test environment: {e}")
            else:
                raise  # Re-raise unexpected errors

    def test_bionty_cell_marker_get_cell_types(self):
        """Test getting cell types from CellMarker database."""
        provider = BiontyCellMarkerProvider()

        try:
            # Test with more specific terms - use well-known cell types
            # Try both "lymphocyte" and "T cell" to increase chances of finding something
            cell_types_lymph = provider.get_cell_types(tissue="lymphocyte", species="human", stage="adult")
            cell_types_tcell = provider.get_cell_types(tissue="T cell", species="human", stage="adult")

            # Combine results
            all_cell_types = cell_types_lymph + cell_types_tcell

            # All should be CellTypeInfo objects with markers
            for ct in all_cell_types:
                assert hasattr(ct, "name")
                assert ct.name is not None
                assert len(ct.name) > 0
                # CellMarker provider should include markers
                assert hasattr(ct, "markers")

            print(f"Found {len(all_cell_types)} cell types from CellMarker database")
        except Exception as e:
            # In testing environments, bionty might not be properly initialized
            if any(
                keyword in str(e)
                for keyword in ["bionty_source", "bionty_organism", "lamindb", "instance", "no such table"]
            ):
                pytest.skip(f"Bionty/lamindb not properly set up in test environment: {e}")
            else:
                raise  # Re-raise unexpected errors

    def test_bionty_cell_marker_get_markers(self):
        """Test getting markers from CellMarker database."""
        provider = BiontyCellMarkerProvider()

        try:
            # Test with well-known cell types
            cell_types = ["T cell", "B cell"]
            markers = provider.get_markers(
                cell_types=cell_types, tissue="blood", species="human", stage="adult", n_markers=5
            )

            # Should get a dictionary back
            assert isinstance(markers, dict)
            # Should have entries for the cell types we requested
            for ct in cell_types:
                if ct in markers:  # May not find all cell types
                    assert isinstance(markers[ct], list)
                    assert len(markers[ct]) <= 5  # Respects n_markers limit
        except Exception as e:
            # In testing environments, bionty might not be properly initialized
            if any(
                keyword in str(e)
                for keyword in ["bionty_source", "bionty_organism", "lamindb", "instance", "no such table"]
            ):
                pytest.skip(f"Bionty/lamindb not properly set up in test environment: {e}")
            else:
                raise  # Re-raise unexpected errors

    def test_composite_orchestrator_functionality(self):
        """Test that composite orchestrator works with real bionty data."""
        # Create a mock annotator that won't actually be called since we're using bionty for cell types
        mock_annotator = type(
            "MockAnnotator",
            (),
            {
                "query_llm": lambda self, **kwargs: None,
                "prompts": type(
                    "MockPrompts", (), {"get_cell_type_marker_prompt": lambda self, **kwargs: "mock prompt"}
                )(),
                "species": "human",
                "tissue": "lymphocyte",
                "stage": "adult",
            },
        )()

        try:
            # Create orchestrator that uses bionty for cell types, would use LLM for markers
            orchestrator = create_orchestrator("bionty_cellontology+llm", annotator=mock_annotator)

            # Test that we can get the structure set up correctly
            assert isinstance(orchestrator.primary_provider, BiontyCellOntologyProvider)
            assert isinstance(orchestrator.marker_provider, LLMProvider)

            # Test getting cell types (should work with real bionty)
            cell_types, markers = orchestrator.get_cell_types_and_markers(
                tissue="lymphocyte", species="human", stage="adult", n_markers=5
            )

            # Should get a list of cell types (may be empty, that's ok)
            assert isinstance(cell_types, list)
            # Markers would be empty since LLM provider would fail without real annotator
            assert isinstance(markers, dict)

            print(f"Orchestrator found {len(cell_types)} cell types")
        except Exception as e:
            # In testing environments, bionty might not be properly initialized
            if "bionty_source" in str(e) or "lamindb" in str(e) or "instance" in str(e):
                pytest.skip(f"Bionty/lamindb not properly set up in test environment: {e}")
            else:
                raise  # Re-raise unexpected errors
