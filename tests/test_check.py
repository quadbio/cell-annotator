"""Test dependency checking functionality."""

import pytest

from cell_annotator.check import check_deps


def test_check_deps_with_available_dependency():
    """Test that check_deps works with an available dependency."""
    # This should not raise any error since packaging is a core dependency
    check_deps("openai")  # OpenAI is in test dependencies


def test_check_deps_with_unavailable_dependency():
    """Test that check_deps raises RuntimeError for unavailable dependencies."""
    with pytest.raises(RuntimeError, match="This function relies on rapids_singlecell"):
        check_deps("rapids-singlecell")


def test_check_deps_with_multiple_dependencies():
    """Test checking multiple dependencies at once."""
    # This should work for available deps
    check_deps("openai")

    # This should fail for unavailable deps
    with pytest.raises(RuntimeError):
        check_deps("rapids-singlecell", "cupy")


def test_check_deps_with_unknown_dependency():
    """Test that check_deps raises RuntimeError for unknown dependencies."""
    with pytest.raises(RuntimeError, match="Dependency 'unknown-package' is not registered"):
        check_deps("unknown-package")
