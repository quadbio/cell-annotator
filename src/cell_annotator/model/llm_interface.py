"""LLM interface class for querying models."""

from cell_annotator._constants import PackageConstants
from cell_annotator._docs import d
from cell_annotator._logging import logger
from cell_annotator._response_formats import BaseOutput
from cell_annotator.model._api_keys import APIKeyMixin, detect_provider_from_model
from cell_annotator.model._providers import get_provider


@d.dedent
class LLMInterface(APIKeyMixin):
    """
    Base class for interfacing with Large Language Models (LLMs).

    Provides a unified interface for querying different LLM providers, supporting
    automatic provider detection from model names and handling API key management.
    This class is designed to be a generic gateway to LLMs without any specific
    application context.

    Parameters
    ----------
    %(model)s
    %(max_completion_tokens)s
    %(provider)s
    %(api_key)s
    _skip_validation
        For internal use. If True, skips API key validation.
    """

    def __init__(
        self,
        model: str | None = None,
        max_completion_tokens: int | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        _skip_validation: bool = False,
    ):
        super().__init__()  # Initialize APIKeyMixin

        self.max_completion_tokens = max_completion_tokens

        # Determine provider and model
        if provider is None and model is None:
            # Auto-select the first available provider and its default model.
            # When ``_skip_validation`` is set or a manual API key is supplied,
            # we must not raise on missing env keys — fork-PR CI runs and
            # ``LLMInterface(_skip_validation=True)`` callers both rely on this.
            if api_key is None and not _skip_validation:
                available_providers = self.api_keys.get_available_providers()
                if not available_providers:
                    raise ValueError(
                        "No API keys found. Please set up at least one provider. "
                        "Run APIKeyManager().print_status() for setup instructions."
                    )
                provider = available_providers[0]
            else:
                provider = "openai"
            model = PackageConstants.default_models[provider]
        elif provider is None and model is not None:
            # Model specified, auto-detect provider
            provider = self._detect_provider_from_model(model)
        elif provider is not None and model is None:
            # Provider specified, use default model for that provider
            if provider not in PackageConstants.default_models:
                raise ValueError(
                    f"Unknown provider '{provider}'. Supported providers: {PackageConstants.supported_providers}"
                )
            model = PackageConstants.default_models[provider]

        # At this point, both provider and model should be strings
        assert provider is not None, "Provider should not be None at this point"
        assert model is not None, "Model should not be None at this point"

        # Validate provider and set up (skip if already validated by parent)
        if not _skip_validation and api_key is None:
            # Only check environment API keys if no manual key provided
            if not self.check_api_access(provider=provider):
                raise ValueError(f"Cannot use model '{model}': missing API key for provider '{provider}'")

        self._provider = get_provider(provider, api_key=api_key)
        self._provider_name = provider
        self.model = model

    def __repr__(self) -> str:
        """Return a string representation of the LLMInterface."""
        lines = []
        lines.append(f"🧬 {self.__class__.__name__}")
        lines.append("=" * (len(self.__class__.__name__) + 3))

        # Model configuration
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
        """Auto-detect provider from model name. Thin wrapper around the shared helper."""
        return detect_provider_from_model(model)

    @d.dedent
    def query_llm(
        self,
        instruction: str,
        response_format: type[BaseOutput],
        agent_description: str = "You are a helpful assistant.",
        other_messages: list | None = None,
    ) -> BaseOutput:
        """
        Query the LLM with a given instruction.

        Parameters
        ----------
        %(instruction)s
        %(response_format)s
        agent_description
            A description of the agent's role or persona.
        %(other_messages)s

        %(returns_parsed_response)s
        """
        response = self._provider.query(
            agent_description=agent_description,
            instruction=instruction,
            model=self.model,
            response_format=response_format,
            other_messages=other_messages,
            max_completion_tokens=self.max_completion_tokens,
        )

        return response

    @d.dedent
    def list_available_models(self) -> list[str]:
        """
        List available models for the current provider.

        %(returns_list_str)s
        """
        return self._provider.list_available_models()

    def test_query(self, return_details: bool = False) -> bool | tuple[bool, str]:
        """
        Test if the LLM setup is working correctly.

        Performs a simple structured-output query against the configured model.
        For OpenRouter slugs whose upstream model does not implement OpenAI's
        ``.parse()`` endpoint, the provider's fallback chain
        (``extra_body`` json_schema → plain ``json_object`` → optional text-repair)
        carries the request, so the same code path works for every provider.

        Parameters
        ----------
        return_details
            If True, returns (success, message) tuple with detailed information.
            If False, returns only boolean success status.

        Returns
        -------
        If return_details=False: True if the test query succeeds, False otherwise.
        If return_details=True: Tuple of (success, message) with detailed status.
        """
        try:
            # Use a simple test response format with default values

            class TestResponse(BaseOutput):
                """Simple response format for testing."""

                message: str = "test"  # Provide default to avoid validation errors

            # Make a simple test query
            response = self.query_llm(
                agent_description="You are a helpful assistant.",
                instruction="Respond with a simple greeting message.",
                response_format=TestResponse,
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
