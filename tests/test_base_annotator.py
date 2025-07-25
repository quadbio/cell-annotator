from unittest.mock import patch

import pytest

from cell_annotator._response_formats import OutputForTesting
from cell_annotator.base_annotator import BaseAnnotator


class TestBaseAnnotator:
    @patch("cell_annotator.base_annotator.BaseAnnotator.query_llm")
    def test_query_llm_mock(self, mock_query_llm, base_annotator):
        """Test query_llm with mocked response across all providers."""
        mock_response = OutputForTesting(parsed_response="parsed_response")
        mock_query_llm.return_value = mock_response

        response = base_annotator.query_llm(
            instruction="Test instruction",
            response_format=OutputForTesting,
        )

        assert response.parsed_response == "parsed_response"
        mock_query_llm.assert_called_once_with(
            instruction="Test instruction",
            response_format=OutputForTesting,
        )

    @pytest.mark.real_llm_query()
    def test_query_llm_real(self, base_annotator):
        """Test actual query_llm call across all available providers."""
        response = base_annotator.query_llm(instruction="Test instruction", response_format=OutputForTesting)

        assert response is not None
        assert isinstance(response, OutputForTesting)
        assert hasattr(response, "parsed_response")
        print(f"✅ {base_annotator._provider_name} provider test passed with model: {base_annotator.model}")
        print(f"Response: {response.parsed_response}")

    def test_provider_auto_detection(self):
        """Test automatic provider detection when none specified."""
        annotator = BaseAnnotator(
            species="human",
            tissue="brain",
            stage="adult",
            cluster_key="leiden",
        )

        # Should have auto-detected a provider
        assert annotator._provider_name is not None
        assert annotator._provider is not None

    def test_explicit_provider_selection(self):
        """Test explicit provider selection."""
        for provider_name in ["openai", "gemini", "anthropic"]:
            try:
                annotator = BaseAnnotator(
                    species="human",
                    tissue="brain",
                    stage="adult",
                    cluster_key="leiden",
                    provider=provider_name,
                )
                # If provider is available, should be selected
                assert annotator._provider_name == provider_name
            except (ValueError, RuntimeError):
                # Provider not available, which is fine
                pass

    def test_provider_model_combination(self):
        """Test provider and model combinations."""
        # Test cases with different provider/model combinations
        test_cases = [
            ("openai", "gpt-4o-mini"),
            ("gemini", "gemini-1.5-flash"),
            ("anthropic", "claude-3-haiku-20240307"),
        ]

        for provider_name, model in test_cases:
            try:
                annotator = BaseAnnotator(
                    species="human",
                    tissue="brain",
                    stage="adult",
                    cluster_key="leiden",
                    provider=provider_name,
                    model=model,
                )
                assert annotator._provider_name == provider_name
                assert annotator.model == model
            except (ValueError, RuntimeError):
                # Provider not available, which is fine
                pass
