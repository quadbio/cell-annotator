"""Base model class to query openAI models."""

from tenacity import retry, stop_after_attempt, wait_random_exponential

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

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def query_openai(
        self,
        instruction: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
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

        Returns
        -------
        Parsed response.
        """
        agent_description = Prompts.AGENT_DESCRIPTION.format(species=self.species)

        response = _query_openai(
            agent_description=agent_description,
            instruction=instruction,
            model=self.model,
            response_format=response_format,
            other_messages=other_messages,
            max_tokens=self.max_tokens,
        )

        return response
