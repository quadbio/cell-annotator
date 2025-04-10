"""Constants used throughout the package."""


class PackageConstants:
    """Constants used througout the package."""

    unknown_name: str = "Unknown"
    unknown_color: str = "#D3D3D3"
    max_markers: int = 200
    min_markers: int = 15
    use_raw: bool = False
    default_model: str = "gpt-4o-mini"
    default_cluster_key: str = "leiden"
    cell_type_key: str = "cell_type_harmonized"


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
