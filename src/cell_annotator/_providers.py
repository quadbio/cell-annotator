"""LLM provider abstraction layer."""

from abc import ABC, abstractmethod

from cell_annotator._logging import logger
from cell_annotator._response_formats import BaseOutput
from cell_annotator.check import check_deps


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

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


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self) -> None:
        """Initialize OpenAI provider with dependency check."""
        check_deps("openai")

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
        from openai import OpenAI

        client = OpenAI()

        if other_messages is None:
            other_messages = []

        try:
            messages = [{"role": "user", "content": instruction}]
            if other_messages:
                messages.extend(other_messages)

            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
            )

            response = completion.choices[0].message
            if response.parsed:
                return response.parsed
            elif response.refusal:
                failure_reason = f"Model refused to respond: {response.refusal}"
                logger.warning(failure_reason)
                return response_format.default_failure(failure_reason=failure_reason)
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

    def __init__(self) -> None:
        """Initialize Gemini provider with dependency check."""
        check_deps("google-genai")

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
        from google import genai

        client = genai.Client()

        # Combine agent description with instruction
        if agent_description:
            full_instruction = f"{agent_description}\n\n{instruction}"
        else:
            full_instruction = instruction

        try:
            response = client.models.generate_content(
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

    def __init__(self) -> None:
        """Initialize Anthropic provider with dependency check."""
        check_deps("anthropic")

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
        import anthropic

        client = anthropic.Anthropic()

        # Combine agent description with instruction
        if agent_description:
            full_instruction = f"{agent_description}\n\n{instruction}"
        else:
            full_instruction = instruction

        try:
            # Note: Anthropic doesn't have native structured output yet
            # So we'll ask for JSON and parse manually
            json_instruction = f"{full_instruction}\n\nPlease respond with valid JSON matching this format: {response_format.model_json_schema()}"

            response = client.messages.create(
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


def get_provider(provider_name: str) -> LLMProvider:
    """
    Get LLM provider by name.

    Parameters
    ----------
    provider_name
        Name of the provider ('openai', 'gemini', or 'anthropic').

    Returns
    -------
    Provider instance.
    """
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
