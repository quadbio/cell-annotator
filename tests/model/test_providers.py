"""Tests for LLM provider implementations."""

import os

import pytest
from flaky import flaky

from cell_annotator._constants import PackageConstants
from cell_annotator._response_formats import BaseOutput
from cell_annotator.model._providers import AnthropicProvider, GeminiProvider, OpenAIProvider, OpenRouterProvider


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

    def test_openrouter_provider_initialization(self):
        """Test OpenRouter provider initialization."""
        # Test with no API key — constructor resolves OPENROUTER_API_KEY from env
        # (so the SDK doesn't fall through to OPENAI_API_KEY when both are set).
        provider = OpenRouterProvider()
        assert provider is not None

        # Test with manual API key — explicit override wins over env.
        provider_with_key = OpenRouterProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"

    def test_provider_repr(self):
        """Test string representation of providers."""
        openai_provider = OpenAIProvider()
        gemini_provider = GeminiProvider()
        anthropic_provider = AnthropicProvider()
        openrouter_provider = OpenRouterProvider()

        # Should contain provider name
        assert "OpenAIProvider" in repr(openai_provider)
        assert "GeminiProvider" in repr(gemini_provider)
        assert "AnthropicProvider" in repr(anthropic_provider)
        assert "OpenRouterProvider" in repr(openrouter_provider)


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
            model=PackageConstants.default_models["openai"],
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
            model=PackageConstants.default_models["gemini"],
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
            model=PackageConstants.default_models["anthropic"],
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


class TestOpenRouterProvider:
    """Isolated tests for OpenRouter provider."""

    @flaky
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_openrouter_list_models_real(self):
        """Test OpenRouter model listing with real API."""
        provider = OpenRouterProvider()
        models = provider.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        model_names = [model.lower() for model in models]
        assert any("/" in model for model in model_names)

    @flaky
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_openrouter_query_real(self):
        """Test OpenRouter query with real API call."""
        provider = OpenRouterProvider()

        response = provider.query(
            agent_description="You are a helpful assistant.",
            instruction="Say hello in exactly one word.",
            model=PackageConstants.default_models["openrouter"],
            response_format=SimpleOutput,
            max_completion_tokens=50,
        )

        assert isinstance(response, SimpleOutput)
        assert hasattr(response, "text")
        assert len(response.text.strip()) > 0

    def test_openrouter_initialization(self):
        """Test OpenRouter provider initialization and basic properties."""
        provider = OpenRouterProvider()
        assert provider is not None

        # Test with manual API key — explicit override wins over env.
        provider_with_key = OpenRouterProvider(api_key="test-key")
        assert provider_with_key._api_key == "test-key"


class TestProviderIntegration:
    """Integration tests for provider functionality."""

    def test_provider_factory_pattern(self):
        """Test creating providers dynamically."""
        providers = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "anthropic": AnthropicProvider,
            "openrouter": OpenRouterProvider,
        }

        for name, provider_class in providers.items():
            provider = provider_class()
            assert provider is not None
            assert provider.__class__.__name__.lower().startswith(name)

    @flaky
    @pytest.mark.skipif(
        not any(
            os.getenv(key) for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
        ),
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
        if os.getenv("OPENROUTER_API_KEY"):
            providers_to_test.append(("openrouter", OpenRouterProvider()))

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
            OpenRouterProvider(api_key="invalid-openrouter-key"),
        ]

        # Just verify they were created successfully
        assert len(invalid_providers) == 4
        for provider in invalid_providers:
            assert provider is not None

    def test_provider_response_format_consistency(self):
        """Test that all providers handle response formats consistently."""
        providers = [
            OpenAIProvider(api_key="fake-key"),
            GeminiProvider(api_key="fake-key"),
            AnthropicProvider(api_key="fake-key"),
            OpenRouterProvider(api_key="fake-key"),
        ]

        # Just test that providers can be created and have the expected interface
        for provider in providers:
            # Test that all providers have the same query method signature
            assert hasattr(provider, "query")
            assert hasattr(provider, "list_available_models")
            assert callable(provider.query)
            assert callable(provider.list_available_models)


class _FallbackOutput(BaseOutput):
    """Test response format whose required field has a default, so ``default_failure`` can construct it."""

    text: str = ""


class TestJSONFallback:
    """Tests for the tiered JSON-fallback chain in OpenAIProvider/OpenRouterProvider."""

    @pytest.mark.parametrize(
        ("provider_cls", "expected"),
        [(OpenAIProvider, False), (OpenRouterProvider, True)],
    )
    def test_text_repair_default(self, provider_cls, expected):
        """``enable_text_repair`` is off for OpenAI, on for OpenRouter — the structured-outputs invariant."""
        assert provider_cls()._enable_text_repair is expected

    @staticmethod
    def _build_mock_client(*, tier_responses):
        """Build a mock OpenAI client whose .parse() raises and whose .create() is programmable.

        Each entry in ``tier_responses`` is either a string (returned as ``message.content``)
        or ``None`` (raises ``RuntimeError`` to fall through to the next tier).
        """
        from unittest.mock import MagicMock

        import openai

        client = MagicMock()
        client.chat.completions.parse.side_effect = openai.OpenAIError("simulated parse failure")

        call_idx = {"i": 0}

        def fake_create(**kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            response = tier_responses[idx] if idx < len(tier_responses) else None
            if response is None:
                raise RuntimeError(f"tier {idx + 1} simulated failure")
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = response
            return mock_completion

        client.chat.completions.create.side_effect = fake_create
        return client

    def test_chain_tries_extra_body_first_then_json_object(self, monkeypatch):
        """Tier 1 uses ``extra_body`` json_schema; on failure tier 2 uses plain ``json_object``."""
        provider = OpenAIProvider(api_key="fake", enable_text_repair=False)
        # Tier 1 fails, tier 2 returns valid JSON.
        client = self._build_mock_client(tier_responses=[None, '{"text": "ok"}'])
        monkeypatch.setattr(provider, "_client", client)

        result = provider.query(
            agent_description="x",
            instruction="y",
            model="m",
            response_format=_FallbackOutput,
        )

        assert isinstance(result, _FallbackOutput)
        assert result.text == "ok"
        calls = client.chat.completions.create.call_args_list
        assert len(calls) == 2
        # Tier 1: extra_body json_schema
        assert "extra_body" in calls[0].kwargs
        assert calls[0].kwargs["extra_body"]["response_format"]["type"] == "json_schema"
        # Tier 2: plain json_object
        assert calls[1].kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.parametrize("enable_text_repair", [False, True])
    def test_text_repair_flag_gates_tier_three(self, monkeypatch, enable_text_repair):
        """Tier 3 (text-repair) only runs when ``enable_text_repair`` is True."""
        provider = OpenAIProvider(api_key="fake", enable_text_repair=enable_text_repair)
        # Tier 1 fails, tier 2 returns prose without JSON braces, tier 3 returns valid JSON.
        client = self._build_mock_client(
            tier_responses=[None, "no json here at all", '{"text": "repaired"}'],
        )
        monkeypatch.setattr(provider, "_client", client)

        result = provider.query(
            agent_description="x",
            instruction="y",
            model="m",
            response_format=_FallbackOutput,
        )

        if enable_text_repair:
            assert isinstance(result, _FallbackOutput)
            assert result.text == "repaired"
            assert client.chat.completions.create.call_count == 3
        else:
            assert result.reason_for_failure is not None
            assert client.chat.completions.create.call_count == 2

    def test_query_reraises_local_error_when_text_repair_disabled(self):
        """ValueError raised before any API call must propagate when text-repair is off."""
        from unittest.mock import MagicMock

        provider = OpenAIProvider(api_key="fake", enable_text_repair=False)
        client = MagicMock()
        client.chat.completions.parse.side_effect = ValueError("local validation issue")
        provider._client = client

        with pytest.raises(ValueError, match="local validation issue"):
            provider.query(
                agent_description="x",
                instruction="y",
                model="m",
                response_format=_FallbackOutput,
            )


class TestOpenRouterFallback:
    """Live tests exercising the fallback chain against a real OpenRouter slug."""

    @flaky
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not available")
    @pytest.mark.real_llm_query()
    def test_openrouter_query_real_with_fallback(self):
        """Pick a slug whose upstream model commonly fails ``.parse()`` to exercise the new ``extra_body`` tier."""
        provider = OpenRouterProvider()
        # anthropic-via-OpenRouter consistently triggered .parse() failure during PR #71 development;
        # haiku is the cheapest known-to-need-fallback option.
        # If model behavior on OpenRouter ever drifts and this slug succeeds via .parse(), the
        # underlying chain still returns a valid response so the assertion holds — mark xfail
        # only if a different failure mode emerges.
        response = provider.query(
            agent_description="You are a helpful assistant.",
            instruction="Say hello in exactly one word.",
            model="anthropic/claude-haiku-4-5",
            response_format=SimpleOutput,
            max_completion_tokens=50,
        )

        assert isinstance(response, SimpleOutput)
        assert hasattr(response, "text")
        assert len(response.text.strip()) > 0
