"""LLM provider abstraction layer."""

from abc import ABC, abstractmethod

from dotenv import load_dotenv

from cell_annotator._logging import logger
from cell_annotator._response_formats import BaseOutput
from cell_annotator.check import check_deps


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def query(
        self,
        agent_description: str,
        instruction: str,
        model: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
        max_completion_tokens: int | None = None,
    ) -> BaseOutput:
        """
        Query the LLM provider with structured output.

        Parameters
        ----------
        agent_description
            Description of the agent/system prompt.
        instruction
            User instruction.
        model
            Model name.
        response_format
            Pydantic response format class.
        other_messages
            Additional messages.
        max_completion_tokens
            Token limit.

        Returns
        -------
        Parsed structured response.
        """

    def list_available_models(self) -> list[str]:
        """
        List models available with current API key and usage tier.

        Returns
        -------
        List of available model names.

        Raises
        ------
        RuntimeError
            If unable to retrieve model list (e.g., invalid API key, network issues).
        """
        return self._list_models_impl()

    @abstractmethod
    def _list_models_impl(self) -> list[str]:
        """Provider-specific implementation for listing models."""


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize OpenAI provider with dependency check.

        Parameters
        ----------
        api_key
            Optional API key. If None, uses environment variable.
        """
        check_deps("openai")
        self._client = None
        self._api_key = api_key

    @property
    def client(self):
        """Lazy-initialized OpenAI client."""
        if self._client is None:
            # Ensure environment variables are loaded (unless manual API key provided)
            if self._api_key is None:
                load_dotenv()
            from openai import OpenAI

            # Use manual API key if provided, otherwise use environment/default
            self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client

    def __repr__(self) -> str:
        """Return a string representation of the OpenAI provider."""
        try:
            models = self.list_available_models()[:5]  # Show first 5 models
            if models:
                model_preview = ", ".join(models)
                if len(self.list_available_models()) > 5:
                    model_preview += ", ..."
                return f"OpenAIProvider(models: {model_preview}). Call .list_available_models() for complete list."
            else:
                return "OpenAIProvider(models: none available). Call .list_available_models() for complete list."
        except Exception:  # noqa: BLE001
            return "OpenAIProvider(models: unavailable). Call .list_available_models() for details."

    def _list_models_impl(self) -> list[str]:
        """List available OpenAI models."""
        models = self.client.models.list()

        # Filter to only chat models (exclude embeddings, TTS, etc.)
        chat_models = []
        for model in models.data:
            model_id = model.id.lower()
            if (
                any(prefix in model_id for prefix in ["gpt", "o1"])
                and "embedding" not in model_id
                and "tts" not in model_id
                and "whisper" not in model_id
                and "dall" not in model_id
            ):
                chat_models.append(model.id)

        return sorted(chat_models)

    def query(
        self,
        agent_description: str,
        instruction: str,
        model: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
        max_completion_tokens: int | None = None,
    ) -> BaseOutput:
        """Query OpenAI API."""
        import openai

        if other_messages is None:
            other_messages = []

        try:
            messages = [{"role": "user", "content": instruction}]
            if other_messages:
                messages.extend(other_messages)

            completion = self.client.chat.completions.parse(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
            )

            response = completion.choices[0].message
            if response.parsed:
                return response.parsed
            elif response.refusal:
                failure_reason = "Model refused to respond: %s"
                logger.warning(failure_reason, response.refusal)
                return response_format.default_failure(failure_reason=failure_reason % response.refusal)
            else:
                failure_reason = "Unknown model failure."
                logger.warning(failure_reason)
                return response_format.default_failure(failure_reason=failure_reason)
        except openai.LengthFinishReasonError:
            failure_reason = "Maximum number of tokens exceeded. Try increasing `max_completion_tokens`."
            logger.warning(failure_reason)
            return response_format.default_failure(failure_reason=failure_reason)
        except openai.OpenAIError as e:
            raise e


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize Gemini provider with dependency check.

        Parameters
        ----------
        api_key
            Optional API key. If None, uses environment variable.
        """
        check_deps("google-genai")
        self._client = None
        self._api_key = api_key

    @property
    def client(self):
        """Lazy-initialized Gemini client."""
        if self._client is None:
            # Ensure environment variables are loaded (unless manual API key provided)
            if self._api_key is None:
                load_dotenv()
            from google import genai

            # Use manual API key if provided, otherwise use environment/default
            if self._api_key:
                self._client = genai.Client(api_key=self._api_key)
            else:
                self._client = genai.Client()
        return self._client

    def __repr__(self) -> str:
        """Return a string representation of the Gemini provider."""
        try:
            models = self.list_available_models()[:5]  # Show first 5 models
            if models:
                model_preview = ", ".join(models)
                if len(self.list_available_models()) > 5:
                    model_preview += ", ..."
                return f"GeminiProvider(models: {model_preview}). Call .list_available_models() for complete list."
            else:
                return "GeminiProvider(models: none available). Call .list_available_models() for complete list."
        except Exception:  # noqa: BLE001
            return "GeminiProvider(models: unavailable). Call .list_available_models() for details."

    def _list_models_impl(self) -> list[str]:
        """List available Gemini models."""
        try:
            models = self.client.models.list()

            # Filter to only generative models (exclude embeddings, etc.)
            chat_models = []
            for model in models:
                if hasattr(model, "name") and model.name:
                    model_name = model.name.replace("models/", "")  # Remove 'models/' prefix if present
                    if "embed" not in model_name.lower() and "text" not in model_name.lower():
                        chat_models.append(model_name)

            return sorted(chat_models)
        except Exception as e:
            logger.error("Failed to list Gemini models: %s", str(e))
            raise RuntimeError("Unable to retrieve Gemini model list. Please check your API key and connection.") from e

    def query(
        self,
        agent_description: str,
        instruction: str,
        model: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
        max_completion_tokens: int | None = None,
    ) -> BaseOutput:
        """Query Gemini API."""
        # Combine agent description with instruction
        if agent_description:
            full_instruction = f"{agent_description}\n\n{instruction}"
        else:
            full_instruction = instruction

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=full_instruction,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_format,
                    "max_output_tokens": max_completion_tokens,
                },
            )

            # Return parsed response
            if hasattr(response, "parsed") and response.parsed:
                return response.parsed
            else:
                # Fallback if parsing fails
                failure_reason = "Gemini failed to parse structured response"
                return response_format.default_failure(failure_reason=failure_reason)

        except (ValueError, TypeError, KeyError) as e:
            failure_reason = f"Gemini API error: {str(e)}"
            return response_format.default_failure(failure_reason=failure_reason)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize Anthropic provider with dependency check.

        Parameters
        ----------
        api_key
            Optional API key. If None, uses environment variable.
        """
        check_deps("anthropic")
        self._client = None
        self._api_key = api_key

    @property
    def client(self):
        """Lazy-initialized Anthropic client."""
        if self._client is None:
            # Ensure environment variables are loaded (unless manual API key provided)
            if self._api_key is None:
                load_dotenv()
            import anthropic

            # Use manual API key if provided, otherwise use environment/default
            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        return self._client

    def __repr__(self) -> str:
        """Return a string representation of the Anthropic provider."""
        try:
            models = self.list_available_models()[:5]  # Show first 5 models
            if models:
                model_preview = ", ".join(models)
                if len(self.list_available_models()) > 5:
                    model_preview += ", ..."
                return f"AnthropicProvider(models: {model_preview}). Call .list_available_models() for complete list."
            else:
                return "AnthropicProvider(models: none available). Call .list_available_models() for complete list."
        except Exception:  # noqa: BLE001
            return "AnthropicProvider(models: unavailable). Call .list_available_models() for details."

    def _list_models_impl(self) -> list[str]:
        """List available Anthropic models."""
        try:
            models = self.client.models.list()
            model_list = [model.id for model in models.data]
            return sorted(model_list)
        except Exception as e:
            logger.error("Failed to list Anthropic models: %s", str(e))
            raise RuntimeError(
                "Unable to retrieve Anthropic model list. Please check your API key and connection."
            ) from e

    def query(
        self,
        agent_description: str,
        instruction: str,
        model: str,
        response_format: type[BaseOutput],
        other_messages: list | None = None,
        max_completion_tokens: int | None = None,
    ) -> BaseOutput:
        """Query Anthropic API."""
        # Combine agent description with instruction
        if agent_description:
            full_instruction = f"{agent_description}\n\n{instruction}"
        else:
            full_instruction = instruction

        try:
            # Note: Anthropic doesn't have native structured output yet
            # So we'll ask for JSON and parse manually
            json_instruction = f"{full_instruction}\n\nPlease respond with valid JSON matching this format: {response_format.model_json_schema()}"

            response = self.client.messages.create(
                model=model,
                max_tokens=max_completion_tokens or 4096,
                messages=[{"role": "user", "content": json_instruction}],
            )

            # Parse the JSON response
            import json

            response_text = response.content[0].text
            parsed_data = json.loads(response_text)
            return response_format(**parsed_data)

        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            failure_reason = f"Anthropic API error: {str(e)}"
            return response_format.default_failure(failure_reason=failure_reason)


# Provider registry - initialize lazily to avoid import errors
_PROVIDERS = {}


def get_provider(provider_name: str, api_key: str | None = None) -> LLMProvider:
    """
    Get LLM provider by name.

    Parameters
    ----------
    provider_name
        Name of the provider ('openai', 'gemini', or 'anthropic').
    api_key
        Optional API key. If provided, creates a new provider instance with this key.
        If None, uses cached provider instance with environment variables.

    Returns
    -------
    Provider instance.
    """
    # If API key is provided, always create a new instance (don't cache)
    if api_key is not None:
        if provider_name == "openai":
            return OpenAIProvider(api_key=api_key)
        elif provider_name == "gemini":
            return GeminiProvider(api_key=api_key)
        elif provider_name == "anthropic":
            return AnthropicProvider(api_key=api_key)
        else:
            available = ["openai", "gemini", "anthropic"]
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(available)}")

    # Use cached provider instance for environment-based keys
    if provider_name not in _PROVIDERS:
        if provider_name == "openai":
            _PROVIDERS[provider_name] = OpenAIProvider()
        elif provider_name == "gemini":
            _PROVIDERS[provider_name] = GeminiProvider()
        elif provider_name == "anthropic":
            _PROVIDERS[provider_name] = AnthropicProvider()
        else:
            available = ["openai", "gemini", "anthropic"]
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(available)}")

    return _PROVIDERS[provider_name]
