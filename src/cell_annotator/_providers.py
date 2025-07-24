"""LLM provider abstraction layer."""

from abc import ABC, abstractmethod

import openai
from openai import OpenAI

from cell_annotator._logging import logger
from cell_annotator._response_formats import BaseOutput


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
        try:
            from google import genai
        except ImportError as err:
            raise ImportError(
                "Google GenAI library not installed. Install with: pip install google-generativeai"
            ) from err

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


# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
}


def get_provider(provider_name: str) -> LLMProvider:
    """
    Get LLM provider by name.

    Parameters
    ----------
    provider_name
        Name of the provider ('openai' or 'gemini').

    Returns
    -------
    Provider instance.
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDERS[provider_name]
