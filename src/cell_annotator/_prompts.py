class Prompts:
    """Prompts for scRNA-seq cell annotation tasks.

    These were heavily inspired by https://github.com/VPetukhov/GPTCellAnnotator.
    """

    CELL_TYPE_PROMPT = (
        "Provide me a comprehensive list of cell types that are expected in {species} {tissue} at stage `{stage}`."
    )
    CELL_TYPE_MARKER_PROMPT = "Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Make sure that you provided markers for **each** cell type you mentioned above."

    ANNOTATION_PROMPT = """
    You need to annotate a {species} {tissue} dataset at stage `{stage}`. You found gene markers for each cluster and need to determine clusters' identities.
    Below is a short list of markers for each cluster:

    {actual_markers_all}

    We expect the following cell types (with associated marker genes) to be present in the dataset, however, some might be absent, and additionall cell types may also be present:
    {expected_markers}

    Determine cell type and state for cluster {cluster_id} (markers {actual_markers_cluster})
    Only output data in the following format:
    ```
    - marker_gene_description: description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y
    - cell_type_annotation: name of this cell type. If unknown, use 'Unknown'.
    - cell_state_annotation: cell state if there is anything specific, 'Normal' otherwise
    - annotation_confidence: one of 'Low', 'Medium', 'High'. If unknown, use 'Unknown'.
    - reason_for_confidence_estimate: reason for the confidence estimate
    ```
    """.strip()

    ORDER_PROMPT = """
    You are tasked with reordering cell type annotations for global consistency across all annotation sets. The goal is to align the order of labels based on biological relationships and similarities.

    Here are the rules you must follow:
    1. For each annotation set, you must include all original elements in the reordered list.
    2. You cannot change the names of any labels. Use the labels exactly as they are provided.
    3. The order must be biologically meaningful and consistent across annotation sets.

    Below are the complete cell type annotations across all sets, in their original order:
    {current_annotation_sets}

    For each annotation set, reorder these labels into a biologically meaningful order that is consistent with the global context.

    ### Example:
    If you are provides with the following sets of annotations:

    - annotation name: cell_type
    - current label ordering: Myeloid, Ventricular_Cardiomyocyte, Fibroblast, Endothelial, Adipocytes, Pericytes, Atrial_Cardiomyocyte, Smooth_muscle_cells, Neuronal, Lymphoid, Mesothelial

    - annotation name: cell_type_predicted
    - current label ordering: Macrophages, Cardiomyocytes, Fibroblasts, Endothelial cells, Adipocytes, Pericytes, Smooth muscle cells, Neurons, T cells, Epicardial cells

    A possible re-ordering could be:
    - annotation name: cell_type
    - new label ordering: Myeloid, Lymphoid, Ventricular_Cardiomyocyte, Atrial_Cardiomyocyte, Smooth_muscle_cells, Pericytes, Endothelial, Fibroblast, Mesothelial, Adipocytes, Neuronal

    - annotation name: cell_type_predicted
    - new label ordering: T cells, Macrophages, Cardiomyocytes, Epicardial cells, Smooth muscle cells, Pericytes, Endothelial cells, Fibroblasts, Adipocytes, Neurons

    ### Output format:
    For each annotation set, provide output in the following format
    ```
    - annotation_name: the original annotation name provided
    - new_label_ordering: the new, globally consistent ordering of cell type names
    ```
    """.strip()

    COLOR_PROMPT = """
    You are tasked with assigning meaningful colors to cell type labels. Below are cell type annotations, already reordered in a biologically meaningful manner:

    {cluster_names}

    Now assign colors to these cell the labels. Follow these rules:
    1. Use colors that are biologically meaningful: similar cell types should have related colors (e.g., shades of the same color family).
    2. All labels must have distinct colors to avoid confusion.
    3. Use hexadecimal color codes (e.g., "#1f77b4").
    4. Do not use white, black, or grey colors.
    5. Do not modify the order of the labels.
    6. Include all labels in the color assignmeent, and do not modify them in any way.

    ### Example:
    If the cell type annotations are:
        [T cells, NK cells, B cells, Macrophages, Dendritic cells]
    A possible color assignment could be:
        "T cells": "#1f77b4",
        "NK cells": "#aec7e8",a
        "B cells": "#ff7f0e",
        "Macrophages": "#2ca02c",
        "Dendritic cells": "#98df8a"

    ### Output format:
    For each cell type, provide output in the following format:
    ```
    - original_cell_type_label: the original cell type label
    - assigned_color: the color assigned to this cell type
    ```
    """.strip()

    AGENT_DESCRIPTION = (
        "You're an expert bioinformatician, proficient in scRNA-seq analysis with background in {species} cell biology."
    )
