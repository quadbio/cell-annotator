"""Base model class to query openAI models."""

import time

from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import BaseOutput
from cell_annotator.utils import _query_openai


class BaseAnnotator:
    """
    Shared base class for annotation-related functionality.

    Parameters
    ----------
    species
        Species name.
    tissue
        Tissue name.
    stage
        Developmental stage.
    cluster_key
        Key of the cluster column in adata.obs.
    model
        OpenAI model name.
    max_tokens
        Maximum number of tokens the model is allowed to use.
    """

    def __init__(
        self,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = "leiden",
        model: str = "gpt-4o-mini",
        max_tokens: int | None = None,
    ):
        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.model = model
        self.max_tokens = max_tokens

    def query_openai(
        self,
        instruction: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> BaseOutput:
        """
        Query OpenAI to retrieve structured output based on the provided instruction.

        Parameters
        ----------
        instruction
            Instruction to provide to the model.
        response_format
            Response format class.
        other_messages
            Additional messages to provide to the model.
        max_retries
            Maximum number of retries for rate limit errors.
        backoff_factor
            Factor by which the delay increases with each retry.

        Returns
        -------
        Parsed response.
        """
        agent_description = Prompts.AGENT_DESCRIPTION.format(species=self.species)
        retries = 0

        while retries < max_retries:
            try:
                response = _query_openai(
                    agent_description=agent_description,
                    instruction=instruction,
                    model=self.model,
                    response_format=response_format,
                    other_messages=other_messages,
                    max_tokens=self.max_tokens,
                )
                return response
            except Exception as e:
                if "rate limit" in str(e).lower() or "invalid_api_key" in str(e).lower():
                    retries += 1
                    delay = backoff_factor**retries
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise e

        raise Exception("Max retries exceeded for OpenAI API request.")
