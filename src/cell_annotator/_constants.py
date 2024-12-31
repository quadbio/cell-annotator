"""Constants used throughout the package."""


class PackageConstants:
    """Constants used througout the package."""

    unknown_name: str = "Unknown"
    unknown_color: str = "#D3D3D3"
    max_markers: int = 200
    min_markers: int = 15
    use_raw: bool = False


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
