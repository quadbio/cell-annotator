"""Tests for LLMInterface class."""

from unittest.mock import patch

import pytest
from flaky import flaky

from cell_annotator._response_formats import OutputForTesting
from cell_annotator.model.llm_interface import LLMInterface


class TestLLMInterface:
    def test_repr(self, provider_name):
        """Test __repr__ method produces expected format."""
        interface = LLMInterface(provider=provider_name, max_completion_tokens=300)
        repr_str = repr(interface)

        # Should contain class name
        assert "LLMInterface" in repr_str

        # Should contain model configuration
        assert interface._provider_name in repr_str
        assert interface.model in repr_str
        assert "300" in repr_str  # max_completion_tokens

        # Should contain status
        assert "Status:" in repr_str

    @pytest.mark.parametrize(
        ("model_name", "expected_provider"),
        [
            ("gpt-4o-mini", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-preview", "openai"),
            ("gemini-1.5-flash", "gemini"),
            ("gemini-pro", "gemini"),
            ("text-bison", "gemini"),
            ("claude-3-5-sonnet-20241022", "anthropic"),
            ("claude-3-haiku-20240307", "anthropic"),
            ("claude-3-opus-20240229", "anthropic"),
            ("unknown-model", "openai"),  # Should default to openai
        ],
    )
    def test_detect_provider_from_model(self, model_name, expected_provider):
        """Test automatic provider detection from model names."""
        interface = LLMInterface(_skip_validation=True)
        detected_provider = interface._detect_provider_from_model(model_name)
        assert detected_provider == expected_provider

    @patch("cell_annotator.model.llm_interface.LLMInterface.query_llm")
    def test_query_success(self, mock_query_llm, provider_name):
        """Test successful test_query scenario."""
        # Mock successful response
        mock_response = OutputForTesting(parsed_response="test successful")
        mock_query_llm.return_value = mock_response

        interface = LLMInterface(provider=provider_name)

        # Test boolean return
        result = interface.test_query()
        assert result is True

        # Test detailed return
        result_with_details = interface.test_query(return_details=True)
        assert isinstance(result_with_details, tuple)
        assert result_with_details[0] is True
        assert "Successfully queried" in result_with_details[1]
        assert provider_name in result_with_details[1]

    @patch("cell_annotator.model.llm_interface.LLMInterface.query_llm")
    def test_query_failure_with_reason(self, mock_query_llm, provider_name):
        """Test test_query when LLM returns failure reason."""
        # Mock response with failure reason
        mock_response = OutputForTesting(parsed_response="test", reason_for_failure="API rate limit exceeded")
        mock_query_llm.return_value = mock_response

        interface = LLMInterface(provider=provider_name)

        # Test boolean return
        result = interface.test_query()
        assert result is False

        # Test detailed return
        result_with_details = interface.test_query(return_details=True)
        assert isinstance(result_with_details, tuple)
        assert result_with_details[0] is False
        assert "Query failed:" in result_with_details[1]
        assert "API rate limit exceeded" in result_with_details[1]

    @patch("cell_annotator.model.llm_interface.LLMInterface.query_llm")
    def test_query_exception_handling(self, mock_query_llm, provider_name):
        """Test test_query exception handling for different error types."""
        test_cases = [
            (ValueError("Model 'unknown-model' not found"), "Model", "not found"),
            (RuntimeError("401 Unauthorized"), "Invalid API key", provider_name),
            (ConnectionError("Network timeout"), "Network connection error", ""),
            (Exception("Generic error"), "Error: Generic error", ""),
        ]

        interface = LLMInterface(provider=provider_name)

        for exception, expected_msg_part, expected_provider_part in test_cases:
            mock_query_llm.side_effect = exception

            # Test boolean return
            result = interface.test_query()
            assert result is False

            # Test detailed return
            result_with_details = interface.test_query(return_details=True)
            assert isinstance(result_with_details, tuple)
            assert result_with_details[0] is False
            assert expected_msg_part in result_with_details[1]
            if expected_provider_part:
                assert expected_provider_part in result_with_details[1]

    @flaky
    @pytest.mark.real_llm_query()
    def test_query_real(self, provider_name):
        """Test actual query_llm call across all available providers."""
        interface = LLMInterface(provider=provider_name, max_completion_tokens=300)

        response = interface.query_llm(
            agent_description="You are a helpful assistant.",
            instruction="Respond with a simple greeting message.",
            response_format=OutputForTesting,
        )

        assert response is not None
        assert isinstance(response, OutputForTesting)
        assert hasattr(response, "parsed_response")
        print(f"✅ {interface._provider_name} provider test passed with model: {interface.model}")

    def test_list_available_models(self, provider_name):
        """Test list_available_models returns expected format."""
        interface = LLMInterface(provider=provider_name)
        models = interface.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)
