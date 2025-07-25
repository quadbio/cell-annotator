"""Constants used throughout the package."""


class PackageConstants:
    """Constants used througout the package."""

    unknown_name: str = "Unknown"
    unknown_color: str = "#D3D3D3"
    max_markers: int = 200
    min_markers: int = 15
    use_raw: bool = False
    default_model: str = "gpt-4o-mini"  # Legacy default for backward compatibility
    default_models: dict[str, str] = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }
    # Supported LLM providers
    supported_providers: list[str] = ["openai", "gemini", "anthropic"]
    default_cluster_key: str = "leiden"
    cell_type_key: str = "cell_type_harmonized"

    @classmethod
    def list_all_available_models(cls) -> dict[str, list[str]]:
        """
        List all available models across all providers.

        Returns
        -------
        Dictionary mapping provider names to lists of available models.
        """
        from cell_annotator._providers import get_provider

        all_models = {}
        for provider_name in cls.supported_providers:
            try:
                provider = get_provider(provider_name)
                models = provider.list_available_models()
                if models:  # Only include providers with available models
                    all_models[provider_name] = models
            except Exception:  # noqa: BLE001
                # Skip providers that can't be initialized (missing dependencies, API keys, etc.)
                continue

        return all_models


class PromptExamples:
    """Examples to be used in prompts."""

    unordered_cell_types = [
        "Myeloid",
        "Ventricular_Cardiomyocyte",
        "Fibroblast",
        "Endothelial",
        "Adipocytes",
        "Pericytes",
        "Atrial_Cardiomyocyte",
        "Smooth_muscle_cells",
        "Neuronal",
        "Lymphoid",
        "Mesothelial",
        "T cells",
        "Macrophages",
        "Cardiomyocytes",
        "Cardiac fibroblasts",
        "Endothelial cells",
        "Neurons",
        "Oligodendrocytes",
        "Immune cells",
        "Neural crest-derived cells",
    ]
    ordered_cell_types = [
        "Immune cells",
        "Myeloid",
        "Lymphoid",
        "T cells",
        "Macrophages",
        "Cardiomyocytes",
        "Ventricular_Cardiomyocyte",
        "Atrial_Cardiomyocyte",
        "Smooth_muscle_cells",
        "Pericytes",
        "Endothelial",
        "Endothelial cells",
        "Fibroblast",
        "Cardiac fibroblasts",
        "Mesothelial",
        "Adipocytes",
        "Neuronal",
        "Neurons",
        "Neural crest-derived cells",
        "Oligodendrocytes",
    ]

    color_mapping_example = {  # This example is from Pijuan-Sala et al., Nature 2019
        "Epiblast": "#504337",
        "Primitive streak": "#D0B188",
        "Caudal epiblast": "#8B5250",
        "Primordial germ cells": "#F7C212",
        "Anterior primitive streak": "#B48E5D",
        "Notochord": "#0E368B",
        "Def. endoderm": "#EE80B3",
        "Gut": "#E83E8B",
        "Nascent mesoderm": "#B87FB1",
        "Mixed mesoderm": "#D7C1DD",
        "Intermediate mesoderm": "#178980",
        "Caudal mesoderm": "#337099",
        "Paraxial mesoderm": "#7BA6C2",
        "Somitic mesoderm": "#074366",
        "Pharyngeal mesoderm": "#BFE6FA",
        "Cardiomyocytes": "#A3007A",
        "Allantois": "#411A77",
        "ExE mesoderm": "#745A9D",
        "Mesenchyme": "#BE6414",
        "Haemato-endothelial prog.": "#F9B080",
        "Blood progenitors 1": "#F6D6C4",
        "Blood progenitors 2": "#BD9885",
        "Erythroid 1": "#B80D1F",
        "Erythroid 2": "#F37B70",
        "Erythroid 3": "#E8371B",
        "Endothelium": "#FC7417",
        "Neuromesodermal progenitors": "#7DBD80",
        "Rostral neuroectoderm": "#559B30",
        "Caudal neuroectoderm": "#283E1A",
        "Neural crest": "#B6B876",
        "Forebrain/midbrain/hindbrain": "#52683E",
        "Spinal cord": "#C3DB75",
        "Surface ectoderm": "#F5F88D",
        "Visceral endoderm": "#F2AFC0",
        "ExE endoderm": "#6B5460",
    }
