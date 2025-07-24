"""Base model class to query LLM models."""

from cell_annotator._api_keys import APIKeyMixin
from cell_annotator._constants import PackageConstants
from cell_annotator._logging import logger
from cell_annotator._prompts import Prompts
from cell_annotator._providers import get_provider
from cell_annotator._response_formats import BaseOutput


class BaseAnnotator(APIKeyMixin):
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
        Model name (e.g., 'gpt-4o-mini', 'gemini-2.5-flash'). If None, uses default model for the provider.
    max_completion_tokens
        Maximum number of tokens the model is allowed to use.
    provider
        LLM provider ('openai', 'gemini', or 'anthropic'). If None, auto-detects from model name or uses first available provider.
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
        _skip_validation: bool = False,
    ):
        super().__init__()  # Initialize APIKeyMixin

        self.species = species
        self.tissue = tissue
        self.stage = stage
        self.cluster_key = cluster_key
        self.max_completion_tokens = max_completion_tokens
        self.prompts = Prompts(species=species, tissue=tissue, stage=stage)

        # Determine provider and model
        if provider is None and model is None:
            # Auto-select the first available provider and its default model
            available_providers = self.api_keys.get_available_providers()
            if not available_providers:
                raise ValueError(
                    "No API keys found. Please set up at least one provider. "
                    "Run APIKeyManager().print_status() for setup instructions."
                )
            provider = available_providers[0]
            model = PackageConstants.default_models[provider]
        elif provider is None and model is not None:
            # Model specified, auto-detect provider
            provider = self._detect_provider_from_model(model)
        elif provider is not None and model is None:
            # Provider specified, use default model for that provider
            model = PackageConstants.default_models.get(provider, PackageConstants.default_model)

        # At this point, both provider and model should be strings
        assert provider is not None, "Provider should not be None at this point"
        assert model is not None, "Model should not be None at this point"

        # Validate provider and set up (skip if already validated by parent)
        if not _skip_validation:
            if not self.check_api_access(provider=provider):
                raise ValueError(f"Cannot use model '{model}': missing API key for provider '{provider}'")

        self._provider = get_provider(provider)
        self._provider_name = provider
        self.model = model

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

    def _detect_provider_from_model(self, model: str) -> str:
        """
        Auto-detect provider from model name.

        Parameters
        ----------
        model
            Model name.

        Returns
        -------
        Provider name.
        """
        if any(keyword in model.lower() for keyword in ["gpt", "o1"]):
            return "openai"
        elif any(keyword in model.lower() for keyword in ["gemini", "bison"]):
            return "gemini"
        elif any(keyword in model.lower() for keyword in ["claude", "sonnet", "haiku", "opus"]):
            return "anthropic"
        else:
            # Default to OpenAI for unknown models
            return "openai"

    def query_llm(
        self,
        instruction: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
    ) -> BaseOutput:
        """
        Query LLM to retrieve structured output based on the provided instruction.

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

    def test_query(self, return_details: bool = False) -> bool | tuple[bool, str]:
        """
        Test if the LLM setup is working correctly.

        Performs a simple query to verify that the API key is valid
        and the model can be accessed successfully.

        Parameters
        ----------
        return_details
            If True, returns (success, message) tuple with detailed information.
            If False, returns only boolean success status.

        Returns
        -------
        If return_details=False: True if the test query succeeds, False otherwise.
        If return_details=True: Tuple of (success, message) with detailed status.

        Examples
        --------
        >>> annotator = BaseAnnotator(species="human", tissue="brain")
        >>> if annotator.test_query():
        ...     print("Setup is working!")
        ... else:
        ...     print("Setup failed - check API keys and model access")
        >>>
        >>> # Get detailed information
        >>> success, message = annotator.test_query(return_details=True)
        >>> print(f"Status: {success}, Details: {message}")
        """
        try:
            # Use a simple test response format with default values

            class TestResponse(BaseOutput):
                """Simple response format for testing."""

                message: str = "test"  # Provide default to avoid validation errors

            # Make a simple test query
            response = self._provider.query(
                agent_description="You are a helpful assistant.",
                instruction="Respond with a simple greeting message.",
                model=self.model,
                response_format=TestResponse,
                max_completion_tokens=1000,
            )

            # Check if we got a valid response
            if response.reason_for_failure is None:
                if return_details:
                    return True, f"✅ Successfully queried {self._provider_name} model '{self.model}'"
                return True
            else:
                if return_details:
                    return False, f"❌ Query failed: {response.reason_for_failure}"
                return False

        except Exception as e:  # noqa: BLE001
            # Catch all exceptions (API errors, network issues, etc.)
            error_msg = str(e)
            logger.debug("Test query failed: %s", error_msg)

            if return_details:
                # Provide more helpful error messages based on error type
                error_type = type(e).__name__
                if "NotFound" in error_type or "404" in error_msg:
                    return False, f"❌ Model '{self.model}' not found or not accessible"
                elif "Unauthorized" in error_type or "401" in error_msg:
                    return False, f"❌ Invalid API key for {self._provider_name}"
                elif "RateLimited" in error_type or "429" in error_msg:
                    return False, f"❌ Rate limited by {self._provider_name} API"
                elif "Connection" in error_type or "Network" in error_type:
                    return False, "❌ Network connection error"
                else:
                    return False, f"❌ Error: {error_msg}"

            return False
