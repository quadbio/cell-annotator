"""Base model class to query LLM models."""

from cell_annotator._constants import PackageConstants
from cell_annotator._docs import d
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._response_formats import BaseOutput
from cell_annotator.model.llm_interface import LLMInterface


class BaseAnnotator(LLMInterface):
    """
    Shared base class for annotation-related functionality.

    Provides a unified interface for querying different LLM providers to annotate
    cell types based on marker genes. Supports automatic provider detection from
    model names and handles API key management.

    Parameters
    ----------
    %(species)s
    %(tissue)s
    %(stage)s
    %(cluster_key)s
    %(model)s
    %(max_completion_tokens)s
    %(provider)s
    %(api_key)s
    """

    def __init__(
        self,
        species: str,
        tissue: str,
        stage: str = "adult",
        cluster_key: str = PackageConstants.default_cluster_key,
        model: str | None = None,
        max_completion_tokens: int | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        _skip_validation: bool = False,
    ):
        super().__init__(
            model=model,
            max_completion_tokens=max_completion_tokens,
            provider=provider,
            api_key=api_key,
            _skip_validation=_skip_validation,
        )

        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.prompts = Prompts(species=species, tissue=tissue, stage=stage)

    def __repr__(self) -> str:
        """Return a string representation of the BaseAnnotator."""
        lines = []
        lines.append(f"🧬 {self.__class__.__name__}")
        lines.append("=" * (len(self.__class__.__name__) + 3))

        # Biological context
        lines.append(f"📋 Species: {self.species}")
        lines.append(f"🔬 Tissue: {self.tissue}")
        lines.append(f"⏳ Stage: {self.stage}")
        lines.append(f"🔗 Cluster key: {self.cluster_key}")

        # Model configuration
        lines.append("")
        lines.append(f"🤖 Provider: {self._provider_name}")
        lines.append(f"🧠 Model: {self.model}")
        if self.max_completion_tokens:
            lines.append(f"🎚️ Max tokens: {self.max_completion_tokens}")

        # Status
        lines.append("")
        try:
            test_result = self.test_query()
            status = "✅ Ready" if test_result else "❌ Not working"
        except Exception as e:  # noqa: BLE001
            # Catch all exceptions during test (API errors, invalid models, etc.)
            logger.debug("Status check failed: %s", str(e))
            status = "⚠️ Unknown"
        lines.append(f"🔋 Status: {status}")

        return "\n".join(lines)

    @d.dedent
    def query_llm(
        self, instruction: str, response_format: type[BaseOutput], other_messages: list | None = None
    ) -> BaseOutput:
        """
        Query the LLM with a given instruction.

        Parameters
        ----------
        %(instruction)s
        %(response_format)s
        %(other_messages)s

        %(returns_parsed_response)s
        """
        agent_description = self.prompts.get_cell_type_prompt()

        response = self._provider.query(
            agent_description=agent_description,
            instruction=instruction,
            model=self.model,
            response_format=response_format,
            other_messages=other_messages,
            max_completion_tokens=self.max_completion_tokens,
        )

        return response
