import cell_annotator
from cell_annotator._constants import PackageConstants


def test_package_has_version():
    assert cell_annotator.__version__ is not None


def test_list_all_available_models():
    """Test that list_all_available_models returns expected structure."""
    all_models = PackageConstants.list_all_available_models()

    # Should return a dictionary
    assert isinstance(all_models, dict)

    # All keys should be valid provider names
    for provider in all_models.keys():
        assert provider in PackageConstants.supported_providers

    # All values should be non-empty lists of strings
    for _provider, models in all_models.items():
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)
