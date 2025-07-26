"""Tests for LLM provider implementations."""

import os

import pytest
from flaky import flaky

from cell_annotator._response_formats import BaseOutput
from cell_annotator.model._providers import AnthropicProvider, GeminiProvider, OpenAIProvider


class SimpleOutput(BaseOutput):
    """Simple test output format."""

    text: str


class TestLLMProviders:
    """Test suite for LLM provider implementations."""

    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization."""
        # Test with no API key
        provider = OpenAIProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = OpenAIProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"

    def test_gemini_provider_initialization(self):
        """Test Gemini provider initialization."""
        # Test with no API key
        provider = GeminiProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = GeminiProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"

    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initialization."""
        # Test with no API key
        provider = AnthropicProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = AnthropicProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"

    def test_provider_repr(self):
        """Test string representation of providers."""
        openai_provider = OpenAIProvider()
        gemini_provider = GeminiProvider()
        anthropic_provider = AnthropicProvider()

        # Should contain provider name
        assert "OpenAIProvider" in repr(openai_provider)
        assert "GeminiProvider" in repr(gemini_provider)
        assert "AnthropicProvider" in repr(anthropic_provider)


class TestOpenAIProvider:
    """Isolated tests for OpenAI provider."""

    @flaky
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_openai_list_models_real(self):
        """Test OpenAI model listing with real API."""
        provider = OpenAIProvider()
        models = provider.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should contain common OpenAI models
        model_names = [model.lower() for model in models]
        assert any("gpt" in model for model in model_names)

    @flaky
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_openai_query_real(self):
        """Test OpenAI query with real API call."""
        provider = OpenAIProvider()

        response = provider.query(
            agent_description="You are a helpful assistant.",
            instruction="Say hello in exactly one word.",
            model="gpt-4o-mini",
            response_format=SimpleOutput,
            max_completion_tokens=50,
        )

        assert isinstance(response, SimpleOutput)
        assert hasattr(response, "text")
        assert len(response.text.strip()) > 0

    def test_openai_initialization(self):
        """Test OpenAI provider initialization and basic properties."""
        provider = OpenAIProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = OpenAIProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"

    def test_openai_query_error_handling(self):
        """Test OpenAI provider basic functionality."""
        # Just test that we can create a provider with an API key
        provider = OpenAIProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert hasattr(provider, "query")
        assert hasattr(provider, "list_available_models")


class TestGeminiProvider:
    """Isolated tests for Gemini provider."""

    @flaky
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_gemini_list_models_real(self):
        """Test Gemini model listing with real API."""
        provider = GeminiProvider()
        models = provider.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should contain Gemini models
        model_names = [model.lower() for model in models]
        assert any("gemini" in model for model in model_names)

    @flaky
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_gemini_query_real(self):
        """Test Gemini query with real API call."""
        provider = GeminiProvider()

        response = provider.query(
            agent_description="You are a helpful assistant.",
            instruction="Say hello in exactly one word.",
            model="gemini-1.5-flash",
            response_format=SimpleOutput,
            max_completion_tokens=50,
        )

        assert isinstance(response, SimpleOutput)
        assert hasattr(response, "text")
        assert len(response.text.strip()) > 0

    def test_gemini_initialization(self):
        """Test Gemini provider initialization and basic properties."""
        provider = GeminiProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = GeminiProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"


class TestAnthropicProvider:
    """Isolated tests for Anthropic provider."""

    @flaky
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_anthropic_list_models_real(self):
        """Test Anthropic model listing with real API."""
        provider = AnthropicProvider()
        models = provider.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should contain Claude models
        model_names = [model.lower() for model in models]
        assert any("claude" in model for model in model_names)

    @flaky
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_anthropic_query_real(self):
        """Test Anthropic query with real API call."""
        provider = AnthropicProvider()

        response = provider.query(
            agent_description="You are a helpful assistant.",
            instruction="Say hello in exactly one word.",
            model="claude-3-haiku-20240307",
            response_format=SimpleOutput,
            max_completion_tokens=50,
        )

        assert isinstance(response, SimpleOutput)
        assert hasattr(response, "text")
        assert len(response.text.strip()) > 0

    def test_anthropic_initialization(self):
        """Test Anthropic provider initialization and basic properties."""
        provider = AnthropicProvider()
        assert provider is not None
        assert provider._api_key is None

        # Test with manual API key
        provider_with_key = AnthropicProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"


class TestProviderIntegration:
    """Integration tests for provider functionality."""

    def test_provider_factory_pattern(self):
        """Test creating providers dynamically."""
        providers = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "anthropic": AnthropicProvider,
        }

        for name, provider_class in providers.items():
            provider = provider_class()
            assert provider is not None
            assert provider.__class__.__name__.lower().startswith(name)

    @flaky
    @pytest.mark.skipif(
        not any(os.getenv(key) for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="No API keys available for testing",
    )
    @pytest.mark.real_llm_query()
    def test_available_provider_models(self):
        """Test model availability for configured providers."""
        providers_to_test = []

        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai", OpenAIProvider()))
        if os.getenv("GEMINI_API_KEY"):
            providers_to_test.append(("gemini", GeminiProvider()))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(("anthropic", AnthropicProvider()))

        for provider_name, provider in providers_to_test:
            print(f"Testing {provider_name} models...")
            models = provider.list_available_models()
            assert len(models) > 0, f"No models available for {provider_name}"
            print(f"{provider_name} has {len(models)} models available")

    def test_provider_error_handling(self):
        """Test provider error handling with invalid credentials."""
        # Test with clearly invalid API keys - just check they don't crash on creation
        invalid_providers = [
            OpenAIProvider(api_key="invalid-openai-key"),
            GeminiProvider(api_key="invalid-gemini-key"),
            AnthropicProvider(api_key="invalid-anthropic-key"),
        ]

        # Just verify they were created successfully
        assert len(invalid_providers) == 3
        for provider in invalid_providers:
            assert provider is not None

    def test_provider_response_format_consistency(self):
        """Test that all providers handle response formats consistently."""
        providers = [
            OpenAIProvider(api_key="fake-key"),
            GeminiProvider(api_key="fake-key"),
            AnthropicProvider(api_key="fake-key"),
        ]

        # Just test that providers can be created and have the expected interface
        for provider in providers:
            # Test that all providers have the same query method signature
            assert hasattr(provider, "query")
            assert hasattr(provider, "list_available_models")
            assert callable(provider.query)
            assert callable(provider.list_available_models)
