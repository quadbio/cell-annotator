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
        )

    @patch("cell_annotator.base_annotator.BaseAnnotator.query_openai")
    def test_query_openai(self, mock_query_openai, base_annotator):
        mock_response = OutputForTesting(parsed_response="parsed_response")
        mock_query_openai.return_value = mock_response

        agent_description = base_annotator.prompts.get_agent_description()
        response = base_annotator.query_openai(instruction="Test instruction", response_format=OutputForTesting)

        print("Agent Description:", agent_description)
        print("Instruction:", "Test instruction")
        print("Expected Response:", "parsed_response")
        print("Actual Response:", response.parsed_response)

        assert response.parsed_response == "parsed_response"
        mock_query_openai.assert_called_once_with(
            instruction="Test instruction",
            response_format=OutputForTesting,
        )

    @pytest.mark.openai()
    def test_query_openai_actual(self, base_annotator):
        response = base_annotator.query_openai(instruction="Test instruction", response_format=OutputForTesting)

        assert response is not None
        assert isinstance(response, OutputForTesting)
