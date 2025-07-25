"""Global test configuration and fixtures."""

import pytest
from dotenv import load_dotenv

from cell_annotator._api_keys import APIKeyManager
from cell_annotator.base_annotator import BaseAnnotator
from cell_annotator.cell_annotator import CellAnnotator
from cell_annotator.sample_annotator import SampleAnnotator

from .utils import get_example_data

# Load environment variables at import time, before test collection
# This ensures that @pytest.mark.skipif decorators can see the environment variables
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def load_environment():
    """Ensure environment variables are loaded for all tests."""
    # Environment is already loaded above, this just ensures it's available
    yield


def get_available_providers() -> list[str]:
    """Get list of providers with valid API keys."""
    manager = APIKeyManager()
    return manager.get_available_providers()


# Parametrize fixture for all available providers
@pytest.fixture(params=get_available_providers(), ids=lambda x: f"provider_{x}")
def provider_name(request) -> str:
    """Parametrize tests across all available LLM providers."""
    return request.param


@pytest.fixture
def base_annotator(provider_name: str) -> BaseAnnotator:
    """Create BaseAnnotator instance with specified provider."""
    # Use specific model names that work reliably for testing
    model_map = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }

    return BaseAnnotator(
        species="human",
        tissue="brain",
        stage="adult",
        cluster_key="leiden",
        model=model_map.get(provider_name),
        max_completion_tokens=300,
        provider=provider_name,
    )


@pytest.fixture
def sample_annotator(provider_name: str) -> SampleAnnotator:
    """Create SampleAnnotator instance with specified provider."""
    # Use specific model names that work reliably for testing
    model_map = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }

    adata = get_example_data(n_cells=100, n_samples=1)
    return SampleAnnotator(
        adata=adata,
        sample_name="sample_0",
        species="human",
        tissue="In vitro neurons and fibroblasts",
        stage="adult",
        cluster_key="leiden",
        model=model_map.get(provider_name),
        max_completion_tokens=1500,
        provider=provider_name,
    )


@pytest.fixture
def cell_annotator_single(provider_name: str) -> CellAnnotator:
    """Create CellAnnotator instance with single sample and specified provider."""
    # Use specific model names that work reliably for testing
    model_map = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }

    adata = get_example_data(n_cells=200, n_samples=1)
    return CellAnnotator(
        adata=adata,
        species="human",
        tissue="In vitro neurons and fibroblasts",
        stage="adult",
        cluster_key="leiden",
        sample_key=None,
        model=model_map.get(provider_name),
        max_completion_tokens=1500,
        provider=provider_name,
    )


@pytest.fixture
def cell_annotator_multi(provider_name: str) -> CellAnnotator:
    """Create CellAnnotator instance with multiple samples and specified provider."""
    # Use specific model names that work reliably for testing
    model_map = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }

    adata = get_example_data(n_cells=200, n_samples=2)
    return CellAnnotator(
        adata=adata,
        species="human",
        tissue="In vitro neurons and fibroblasts",
        stage="adult",
        cluster_key="leiden",
        sample_key="sample",
        model=model_map.get(provider_name),
        max_completion_tokens=1500,
        provider=provider_name,
    )
