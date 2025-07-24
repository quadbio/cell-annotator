from unittest.mock import patch

import pytest

from cell_annotator._response_formats import OutputForTesting
from cell_annotator.base_annotator import BaseAnnotator


class TestBaseAnnotator:
    @pytest.fixture
    def base_annotator(self):
        return BaseAnnotator(
            species="human",
            tissue="brain",
            stage="adult",
            cluster_key="leiden",
            model="gpt-4o-mini",
            max_completion_tokens=300,
            provider="openai",  # Explicitly use OpenAI provider
        )

    @patch("cell_annotator.base_annotator.BaseAnnotator.query_llm")
    def test_query_llm_openai(self, mock_query_llm, base_annotator):
        """Test query_llm specifically with OpenAI provider."""
        mock_response = OutputForTesting(parsed_response="parsed_response")
        mock_query_llm.return_value = mock_response

        # Verify we're using OpenAI provider
        assert base_annotator._provider_name == "openai"

        agent_description = base_annotator.prompts.get_cell_type_prompt()
        response = base_annotator.query_llm(instruction="Test instruction", response_format=OutputForTesting)

        print("Provider:", base_annotator._provider_name)
        print("Model:", base_annotator.model)
        print("Agent Description:", agent_description)
        print("Instruction:", "Test instruction")
        print("Expected Response:", "parsed_response")
        print("Actual Response:", response.parsed_response)

        assert response.parsed_response == "parsed_response"
        mock_query_llm.assert_called_once_with(
            instruction="Test instruction",
            response_format=OutputForTesting,
        )

    @pytest.mark.openai()
    def test_query_llm_openai_actual(self, base_annotator):
        """Test actual query_llm call with OpenAI provider."""
        # Verify we're using OpenAI provider
        assert base_annotator._provider_name == "openai"
        assert "gpt" in base_annotator.model.lower()

        response = base_annotator.query_llm(instruction="Test instruction", response_format=OutputForTesting)

        assert response is not None
        assert isinstance(response, OutputForTesting)
        print(f"✅ OpenAI provider test passed with model: {base_annotator.model}")
        print(f"Response: {response.parsed_response}")

    def test_provider_auto_detection(self):
        """Test that provider is auto-detected from model name."""
        # Test OpenAI model detection
        annotator_openai = BaseAnnotator(species="human", tissue="brain", model="gpt-4o-mini")
        assert annotator_openai._provider_name == "openai"

        # Test model without explicit provider (should detect from name)
        annotator_gpt = BaseAnnotator(species="human", tissue="brain", model="gpt-3.5-turbo")
        assert annotator_gpt._provider_name == "openai"

    def test_explicit_provider_selection(self):
        """Test explicit provider selection."""
        annotator = BaseAnnotator(
            species="human",
            tissue="brain",
            provider="openai",  # Explicit provider, should use default model
        )
        assert annotator._provider_name == "openai"
        assert annotator.model == "gpt-4o-mini"  # Should use default OpenAI model

    def test_provider_model_combination(self):
        """Test explicit provider and model combination."""
        annotator = BaseAnnotator(species="human", tissue="brain", provider="openai", model="gpt-4o-mini")
        assert annotator._provider_name == "openai"
        assert annotator.model == "gpt-4o-mini"
