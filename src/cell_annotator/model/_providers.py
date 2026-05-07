"""LLM provider abstraction layer."""

import json
import os
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
        self._base_url = None
        self._default_headers = None

    @property
    def client(self):
        """Lazy-initialized OpenAI client."""
        if self._client is None:
            # Ensure environment variables are loaded (unless manual API key provided)
            if self._api_key is None:
                load_dotenv()
            from openai import OpenAI

            # Use manual API key if provided, otherwise use environment/default
            client_kwargs = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            if self._default_headers:
                client_kwargs["default_headers"] = self._default_headers
            self._client = OpenAI(**client_kwargs)
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
        return self._filter_chat_model_ids(models.data)

    def _filter_chat_model_ids(self, model_data: list) -> list[str]:
        """Filter a model list to chat-capable OpenAI models."""
        chat_models = []
        for model in model_data:
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

        messages = self._build_messages(
            agent_description=agent_description,
            instruction=instruction,
            other_messages=other_messages,
        )

        try:
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
            logger.debug(
                "Structured parse failed for model '%s'. Falling back to JSON-mode query. Error: %s", model, str(e)
            )
            return self._query_with_json_fallback(
                model=model,
                response_format=response_format,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                fallback_error=str(e),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "Non-OpenAI parse failure for model '%s'. Falling back to JSON-mode query. Error: %s", model, str(e)
            )
            return self._query_with_json_fallback(
                model=model,
                response_format=response_format,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                fallback_error=str(e),
            )

    def _build_messages(self, agent_description: str, instruction: str, other_messages: list) -> list[dict[str, str]]:
        """Build chat messages with system prompt and optional history."""
        messages: list[dict[str, str]] = []
        if agent_description:
            messages.append({"role": "system", "content": agent_description})
        if other_messages:
            messages.extend(other_messages)
        messages.append({"role": "user", "content": instruction})
        return messages

    def _query_with_json_fallback(
        self,
        model: str,
        response_format: type[BaseOutput],
        messages: list[dict[str, str]],
        max_completion_tokens: int | None,
        fallback_error: str,
    ) -> BaseOutput:
        """
        Fallback for providers/models that do not support `.parse(...)`.

        Requests plain JSON output and validates it with the Pydantic response model.
        """
        schema = response_format.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=True)
        fallback_instruction = (
            "Return only valid JSON that matches this schema exactly. "
            "Do not include markdown fences or extra text.\n"
            f"JSON schema: {schema_json}"
        )
        fallback_messages = [*messages, {"role": "user", "content": fallback_instruction}]

        request_kwargs: dict = {
            "model": model,
            "messages": fallback_messages,  # type: ignore[arg-type]
            "response_format": {"type": "json_object"},
        }
        if max_completion_tokens is not None:
            # `max_tokens` is the most widely supported field across OpenAI-compatible APIs.
            request_kwargs["max_tokens"] = max_completion_tokens

        try:
            completion = self.client.chat.completions.create(**request_kwargs)
            raw_content = completion.choices[0].message.content
            text = self._coerce_text_content(raw_content)
            if not text:
                return response_format.default_failure(
                    failure_reason=(
                        f"Model returned empty content during JSON fallback. Original parse error: {fallback_error}"
                    )
                )

            # First, try strict JSON parsing directly.
            try:
                return response_format.model_validate_json(text)
            except Exception:  # noqa: BLE001
                pass

            # Then try extracting a JSON object from surrounding text.
            json_candidate = self._extract_json_candidate(text)
            if json_candidate is not None:
                parsed = json.loads(json_candidate)
                return response_format(**parsed)

            # Last attempt: ask the same model to repair plain text into valid JSON.
            repaired_text = self._repair_text_to_json(
                model=model,
                raw_text=text,
                schema_json=schema_json,
                max_completion_tokens=max_completion_tokens,
            )
            if repaired_text:
                try:
                    return response_format.model_validate_json(repaired_text)
                except Exception:  # noqa: BLE001
                    repaired_candidate = self._extract_json_candidate(repaired_text)
                    if repaired_candidate is not None:
                        return response_format(**json.loads(repaired_candidate))

            return response_format.default_failure(
                failure_reason=(
                    "Could not parse structured JSON response from model output. "
                    f"Original parse error: {fallback_error}"
                )
            )
        except Exception as fallback_exception:  # noqa: BLE001
            return response_format.default_failure(
                failure_reason=(
                    "Fallback JSON query failed. "
                    f"Original parse error: {fallback_error}. "
                    f"Fallback error: {str(fallback_exception)}"
                )
            )

    def _coerce_text_content(self, content) -> str:
        """Coerce OpenAI-compatible response content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
                elif hasattr(item, "text"):
                    chunks.append(str(item.text))
            return "\n".join(chunks).strip()
        return str(content).strip()

    def _extract_json_candidate(self, text: str) -> str | None:
        """Extract the first JSON object-like substring from text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _repair_text_to_json(
        self,
        model: str,
        raw_text: str,
        schema_json: str,
        max_completion_tokens: int | None,
    ) -> str:
        """Ask the model to convert plain text into schema-valid JSON."""
        repair_instruction = (
            "Convert the following assistant output into valid JSON matching this schema exactly. "
            "Return JSON only, with no markdown or explanation.\n"
            f"Schema: {schema_json}\n"
            f"Assistant output: {raw_text}"
        )
        repair_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": repair_instruction}],
            "response_format": {"type": "json_object"},
        }
        if max_completion_tokens is not None:
            repair_kwargs["max_tokens"] = max_completion_tokens
        repair_completion = self.client.chat.completions.create(**repair_kwargs)
        repair_content = repair_completion.choices[0].message.content
        return self._coerce_text_content(repair_content)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider implementation (OpenAI-compatible API)."""

    def __init__(self, api_key: str | None = None) -> None:
        # When no manual key is supplied, resolve the OpenRouter key from the
        # environment explicitly. The underlying OpenAI client otherwise picks
        # up ``OPENAI_API_KEY`` (since both providers share the SDK), causing
        # 401s against ``https://openrouter.ai/api/v1`` whenever both keys
        # are configured side-by-side.
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key=api_key)
        self._base_url = "https://openrouter.ai/api/v1"

        # Optional headers recommended by OpenRouter for request attribution.
        referer = os.getenv("OPENROUTER_SITE_URL")
        title = os.getenv("OPENROUTER_APP_NAME")
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self._default_headers = headers if headers else None

    def __repr__(self) -> str:
        """Return a string representation of the OpenRouter provider."""
        try:
            models = self.list_available_models()[:5]  # Show first 5 models
            if models:
                model_preview = ", ".join(models)
                if len(self.list_available_models()) > 5:
                    model_preview += ", ..."
                return f"OpenRouterProvider(models: {model_preview}). Call .list_available_models() for complete list."
            else:
                return "OpenRouterProvider(models: none available). Call .list_available_models() for complete list."
        except Exception:  # noqa: BLE001
            return "OpenRouterProvider(models: unavailable). Call .list_available_models() for details."

    def _list_models_impl(self) -> list[str]:
        """
        List available OpenRouter models.

        OpenRouter exposes models from many upstream providers, so model IDs
        are not restricted to OpenAI prefixes like "gpt" and "o1".
        """
        models = self.client.models.list()

        filtered_models = []
        for model in models.data:
            model_id = model.id.lower()
            if any(keyword in model_id for keyword in ["embedding", "tts", "whisper", "dall", "moderation", "rerank"]):
                continue
            filtered_models.append(model.id)

        return sorted(filtered_models)


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
        """Query Anthropic API using tool use for structured output."""
        try:
            import anthropic

            # Use Anthropic's tool use system for structured output
            # This is the recommended approach for getting structured JSON responses
            tool_schema = {
                "name": "structured_response",
                "description": "Provide a structured response in the specified format",
                "input_schema": response_format.model_json_schema(),
            }

            # Prepare messages
            messages = []

            # Add system message if provided
            system_content = agent_description if agent_description else "You are a helpful assistant."

            # Add other messages if provided
            if other_messages:
                messages.extend(other_messages)

            # Add the main instruction
            messages.append({"role": "user", "content": instruction})

            response = self.client.messages.create(
                model=model,
                max_tokens=max_completion_tokens or 4096,
                system=system_content,
                tools=[tool_schema],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "structured_response"},  # type: ignore[arg-type]
                messages=messages,
            )

            # Extract the structured response from tool use
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if hasattr(content_block, "type") and content_block.type == "tool_use":
                        if hasattr(content_block, "input") and content_block.input:
                            # The input contains our structured data
                            input_data = content_block.input
                            if isinstance(input_data, dict):
                                return response_format(**input_data)

            # Fallback if no tool use found
            failure_reason = "No structured response found in tool use output"
            return response_format.default_failure(failure_reason=failure_reason)

        except anthropic.AnthropicError as e:
            failure_reason = f"Anthropic API error: {str(e)}"
            return response_format.default_failure(failure_reason=failure_reason)
        except (ValueError, TypeError, KeyError) as e:
            failure_reason = f"Response parsing error: {str(e)}"
            return response_format.default_failure(failure_reason=failure_reason)


# Provider registry - initialize lazily to avoid import errors
_PROVIDERS = {}


def get_provider(provider_name: str, api_key: str | None = None) -> LLMProvider:
    """
    Get LLM provider by name.

    Parameters
    ----------
    provider_name
        Name of the provider ('openai', 'gemini', 'anthropic', or 'openrouter').
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
        elif provider_name == "openrouter":
            return OpenRouterProvider(api_key=api_key)
        else:
            available = ["openai", "gemini", "anthropic", "openrouter"]
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(available)}")

    # Use cached provider instance for environment-based keys
    if provider_name not in _PROVIDERS:
        if provider_name == "openai":
            _PROVIDERS[provider_name] = OpenAIProvider()
        elif provider_name == "gemini":
            _PROVIDERS[provider_name] = GeminiProvider()
        elif provider_name == "anthropic":
            _PROVIDERS[provider_name] = AnthropicProvider()
        elif provider_name == "openrouter":
            _PROVIDERS[provider_name] = OpenRouterProvider()
        else:
            available = ["openai", "gemini", "anthropic", "openrouter"]
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(available)}")

    return _PROVIDERS[provider_name]
