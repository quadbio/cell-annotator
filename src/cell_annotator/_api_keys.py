"""API key management for different LLM providers."""

import os

from dotenv import load_dotenv

from cell_annotator._logging import logger


class APIKeyManager:
    """
    Manages API keys for different LLM providers.

    Provides utilities to check availability, validate keys, and guide users
    on setup for different providers.
    """

    # Provider configurations
    PROVIDER_CONFIG = {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "setup_url": "https://platform.openai.com/api-keys",
            "description": "OpenAI GPT models",
        },
        "gemini": {
            "env_var": "GOOGLE_API_KEY",
            "setup_url": "https://aistudio.google.com/apikey",
            "description": "Google Gemini models",
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY",
            "setup_url": "https://console.anthropic.com/settings/keys",
            "description": "Anthropic Claude models",
        },
    }

    def __init__(self, auto_load_env: bool = True):
        """
        Initialize API key manager.

        Parameters
        ----------
        auto_load_env
            Whether to automatically load .env file.
        """
        if auto_load_env:
            load_dotenv()

    def check_key_availability(self) -> dict[str, bool]:
        """
        Check which API keys are available.

        Returns
        -------
        Dictionary mapping provider names to availability status.
        """
        availability = {}
        for provider, config in self.PROVIDER_CONFIG.items():
            availability[provider] = bool(os.getenv(config["env_var"]))
        return availability

    def get_available_providers(self) -> list[str]:
        """
        Get list of providers with valid API keys.

        Returns
        -------
        List of available provider names.
        """
        return [provider for provider, available in self.check_key_availability().items() if available]

    def validate_provider(self, provider: str) -> bool:
        """
        Check if a provider has a valid API key.

        Parameters
        ----------
        provider
            Provider name to validate.

        Returns
        -------
        True if provider has valid API key.
        """
        if provider not in self.PROVIDER_CONFIG:
            return False
        return provider in self.get_available_providers()

    def get_setup_instructions(self, provider: str) -> str:
        """
        Get setup instructions for a specific provider.

        Parameters
        ----------
        provider
            Provider name.

        Returns
        -------
        Setup instruction string.
        """
        if provider not in self.PROVIDER_CONFIG:
            return f"Unknown provider: {provider}"

        config = self.PROVIDER_CONFIG[provider]
        return (
            f"To use {config['description']}, set the environment variable "
            f"`{config['env_var']}` with your API key from {config['setup_url']}"
        )

    def print_status(self) -> None:
        """Print a comprehensive status report of all providers."""
        print("🔑 API Key Status Report")
        print("=" * 50)

        availability = self.check_key_availability()

        for provider, config in self.PROVIDER_CONFIG.items():
            status = "✅" if availability[provider] else "❌"
            print(f"\n{status} {provider.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Environment Variable: {config['env_var']}")

            if availability[provider]:
                print("   Status: Ready to use")
            else:
                print(f"   Setup URL: {config['setup_url']}")
                print(f"   Note: {self.get_setup_instructions(provider)}")

        print("\n📊 Summary:")
        total_providers = len(self.PROVIDER_CONFIG)
        available_count = len(self.get_available_providers())

        print(f"   • {available_count}/{total_providers} providers configured")

        if available_count == 0:
            print("\n⚠️  No API keys found! Please set up at least one provider.")
        elif available_count < total_providers:
            missing = set(self.PROVIDER_CONFIG.keys()) - set(self.get_available_providers())
            print(f"   • Missing providers: {', '.join(missing)}")
        else:
            print("   • All providers configured! 🎉")

    def validate_model_access(self, model: str) -> tuple[bool, str | None]:
        """
        Check if a specific model is accessible by detecting its provider.

        Uses heuristics to detect provider from model name.

        Parameters
        ----------
        model
            Model name to check.

        Returns
        -------
        Tuple of (is_accessible, provider_name)
        """
        # Detect provider from model name using heuristics
        model_lower = model.lower()

        if any(gemini_name in model_lower for gemini_name in ["gemini", "bison"]):
            provider = "gemini"
        elif any(claude_name in model_lower for claude_name in ["claude", "anthropic"]):
            provider = "anthropic"
        elif any(openai_name in model_lower for openai_name in ["gpt", "o1", "davinci", "curie", "babbage", "ada"]):
            provider = "openai"
        else:
            # Default to OpenAI for unknown models (most common)
            provider = "openai"

        if self.validate_provider(provider):
            return True, provider
        else:
            return False, provider

    def check_and_warn(self, provider: str | None = None, model: str | None = None) -> bool:
        """
        Check API key availability and log appropriate warnings.

        Parameters
        ----------
        provider
            Specific provider to check. If None, checks all.
        model
            Specific model to check access for.

        Returns
        -------
        True if requested provider/model is accessible.
        """
        if model:
            accessible, detected_provider = self.validate_model_access(model)
            if accessible:
                logger.info(f"✅ Model '{model}' is accessible via {detected_provider}")
                return True
            elif detected_provider:
                logger.warning(
                    f"❌ Model '{model}' requires {detected_provider} API key. "
                    f"{self.get_setup_instructions(detected_provider)}"
                )
                return False
            else:
                logger.warning(f"❌ Unknown model '{model}'")
                return False

        if provider:
            if self.validate_provider(provider):
                logger.info(f"✅ {provider} API key is configured")
                return True
            else:
                logger.warning(f"❌ {self.get_setup_instructions(provider)}")
                return False

        # Check all providers
        available = self.get_available_providers()
        if available:
            logger.info(f"✅ Available providers: {', '.join(available)}")
            return True
        else:
            logger.warning("❌ No API keys found! Please configure at least one provider.")
            for prov in self.PROVIDER_CONFIG:
                logger.info(f"   {self.get_setup_instructions(prov)}")
            return False


class APIKeyMixin:
    """Mixin class to add API key management capabilities to other classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api_key_manager = APIKeyManager()

    @property
    def api_keys(self) -> APIKeyManager:
        """Access to API key manager."""
        return self._api_key_manager

    def check_api_access(self, provider: str | None = None, model: str | None = None) -> bool:
        """Check API access and log warnings if needed."""
        return self._api_key_manager.check_and_warn(provider=provider, model=model)
