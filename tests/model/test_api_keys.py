"""Tests for API key management functionality."""

import os
from unittest.mock import patch

from cell_annotator.model._api_keys import APIKeyManager, APIKeyMixin


class TestAPIKeyManager:
    """Test APIKeyManager functionality."""

    def test_initialization(self):
        """Test APIKeyManager initialization."""
        manager = APIKeyManager()
        assert manager is not None
        assert hasattr(manager, "PROVIDER_CONFIG")
        assert "openai" in manager.PROVIDER_CONFIG
        assert "gemini" in manager.PROVIDER_CONFIG
        assert "anthropic" in manager.PROVIDER_CONFIG

    def test_supported_providers(self):
        """Test that all expected providers are supported."""
        manager = APIKeyManager()
        expected_providers = {"openai", "gemini", "anthropic"}
        assert set(manager.PROVIDER_CONFIG.keys()) == expected_providers

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}, clear=False)
    def test_openai_key_detection(self):
        """Test OpenAI key detection from environment."""
        manager = APIKeyManager()
        availability = manager.check_key_availability()
        assert availability["openai"] is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_gemini_key"}, clear=False)
    def test_gemini_key_detection(self):
        """Test Gemini key detection from environment."""
        manager = APIKeyManager()
        availability = manager.check_key_availability()
        assert availability["gemini"] is True

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_anthropic_key"}, clear=False)
    def test_anthropic_key_detection(self):
        """Test Anthropic key detection from environment."""
        manager = APIKeyManager()
        availability = manager.check_key_availability()
        assert availability["anthropic"] is True

    @patch.dict(os.environ, {}, clear=True)
    @patch("cell_annotator.model._api_keys.load_dotenv")
    def test_no_keys_available(self, _mock_load_dotenv):
        """Test behavior when no API keys are available."""
        manager = APIKeyManager()
        availability = manager.check_key_availability()
        assert availability["openai"] is False
        assert availability["gemini"] is False
        assert availability["anthropic"] is False
        assert manager.get_available_providers() == []

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai", "ANTHROPIC_API_KEY": "test_anthropic"}, clear=True)
    @patch("cell_annotator.model._api_keys.load_dotenv")
    def test_partial_keys_available(self, _mock_load_dotenv):
        """Test behavior when only some API keys are available."""
        manager = APIKeyManager()
        available = manager.get_available_providers()
        assert "openai" in available
        assert "anthropic" in available
        assert "gemini" not in available
        assert len(available) == 2

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test_openai", "GEMINI_API_KEY": "test_gemini", "ANTHROPIC_API_KEY": "test_anthropic"},
        clear=True,
    )
    @patch("cell_annotator.model._api_keys.load_dotenv")
    def test_all_keys_available(self, _mock_load_dotenv):
        """Test behavior when all API keys are available."""
        manager = APIKeyManager()
        available = manager.get_available_providers()
        expected = {"openai", "gemini", "anthropic"}
        assert set(available) == expected

    def test_validate_provider(self):
        """Test provider validation functionality."""
        manager = APIKeyManager()

        # Test with non-existent provider
        assert not manager.validate_provider("invalid_provider")

        # Test with valid providers (will depend on environment)
        for provider in manager.PROVIDER_CONFIG.keys():
            result = manager.validate_provider(provider)
            assert isinstance(result, bool)

    def test_get_setup_instructions(self):
        """Test setup instructions generation."""
        manager = APIKeyManager()

        # Test with valid provider
        instructions = manager.get_setup_instructions("openai")
        assert "OPENAI_API_KEY" in instructions
        assert "platform.openai.com" in instructions

        # Test with invalid provider
        instructions = manager.get_setup_instructions("invalid")
        assert "Unknown provider" in instructions

    def test_print_status_output(self, capsys):
        """Test that print_status produces output."""
        manager = APIKeyManager()
        manager.print_status()
        captured = capsys.readouterr()
        assert "API Key Status Report" in captured.out
        assert "OPENAI" in captured.out
        assert "GEMINI" in captured.out
        assert "ANTHROPIC" in captured.out

    def test_repr_output(self):
        """Test string representation of manager."""
        manager = APIKeyManager()
        repr_str = repr(manager)
        assert "APIKeyManager Status" in repr_str
        assert "providers configured" in repr_str

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=False)
    def test_env_file_loading(self):
        """Test that environment variables are properly loaded."""
        manager = APIKeyManager()
        availability = manager.check_key_availability()
        assert availability["openai"] is True

    def test_auto_load_env_parameter(self):
        """Test auto_load_env parameter."""
        # Test with auto_load_env=False
        manager = APIKeyManager(auto_load_env=False)
        assert manager is not None

        # Test with auto_load_env=True (default)
        manager = APIKeyManager(auto_load_env=True)
        assert manager is not None


class TestAPIKeyMixin:
    """Test APIKeyMixin functionality."""

    def test_mixin_initialization(self):
        """Test that APIKeyMixin initializes correctly."""
        mixin = APIKeyMixin()
        assert hasattr(mixin, "api_keys")
        assert isinstance(mixin.api_keys, APIKeyManager)

    def test_mixin_check_api_access(self):
        """Test mixin check_api_access method."""
        mixin = APIKeyMixin()

        # Test with valid providers
        for provider in ["openai", "gemini", "anthropic"]:
            result = mixin.check_api_access(provider)
            assert isinstance(result, bool)

        # Test with invalid provider
        assert not mixin.check_api_access("invalid_provider")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=False)
    def test_mixin_inheritance(self):
        """Test APIKeyMixin when inherited by another class."""

        class TestClass(APIKeyMixin):
            def __init__(self):
                super().__init__()
                self.name = "test"

        instance = TestClass()
        assert instance.name == "test"
        assert hasattr(instance, "api_keys")
        # Should detect the OpenAI key we set in the environment
        assert instance.check_api_access("openai") is True


class TestAPIKeyIntegration:
    """Integration tests for API key functionality."""

    def test_real_api_key_detection(self):
        """Test with real environment API keys if available."""
        manager = APIKeyManager()
        available_providers = manager.get_available_providers()

        # At least log what we found for debugging
        print(f"Available providers in test environment: {available_providers}")

        # Basic sanity checks
        assert isinstance(available_providers, list)
        assert all(provider in manager.PROVIDER_CONFIG for provider in available_providers)

    def test_manager_independence(self):
        """Test that multiple manager instances work independently."""
        manager1 = APIKeyManager()
        manager2 = APIKeyManager()

        # They should be separate instances
        assert manager1 is not manager2

        # But should have the same availability status
        availability1 = manager1.check_key_availability()
        availability2 = manager2.check_key_availability()
        assert availability1 == availability2

    def test_provider_config_structure(self):
        """Test that provider config has expected structure."""
        manager = APIKeyManager()

        for _provider, config in manager.PROVIDER_CONFIG.items():
            assert "env_var" in config
            assert "setup_url" in config
            assert "description" in config
            assert isinstance(config["env_var"], str)
            assert isinstance(config["setup_url"], str)
            assert isinstance(config["description"], str)
