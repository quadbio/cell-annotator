"""Prompts for scRNA-seq cell annotation tasks."""


class Prompts:
    """Prompts for scRNA-seq cell annotation tasks.

    These were in parts heavily inspired by https://github.com/VPetukhov/GPTCellAnnotator.
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

    ### Output format:
    ```
    - marker_gene_description: description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y
    - cell_type: name of this cell type. If unknown, use 'Unknown'.
    - cell_state: cell state if there is anything specific, 'Normal' otherwise
    - annotation_confidence: one of 'Low', 'Medium', 'High'. If unknown, use 'Unknown'.
    - reason_for_confidence_estimate: reason for the confidence estimate
    ```
    """.strip()

    ORDER_PROMPT = """
    You are tasked with reordering cell type annotations for global consistency. The goal is to align the order of labels based on biological relationships and similarities.

    Here are the rules you must follow:
    1. You must include all original elements in the reordered list.
    2. You cannot change the names of any labels. Use the labels exactly as they are provided. Keep all duplicates.
    3. The order must be biologically meaningful.

    Biologically meaningful order means grouping similar cell types together based on their lineage, function, or tissue origin. For example, immune cells should be grouped together, and different types of muscle cells should be grouped together.

    Below are the current cell type annotations, in their original, random order:
    {unique_cell_types}

    Reorder these labels into a biologically meaningful order.

    ### Example:
    If you are provided with the following list of cell type labels:
    {example_unordered}

    A possible re-ordering could be:
    {example_ordered}

    ### Output format:
    Provide the reordered cell type annotations as a list with no additional commnets.
    ```
    """.strip()

    DUPLICATE_REMOVAL_PROMPT = """
    You need to remove duplicates from a list of cell type annotations. The same cell type might be included multiple times in the list, for example, with different capitalization, abbreviations, or synonyms. Thus, by duplicates, we mean the same, or an extremely simlar cell type, being represented by two or more elements in the list. The goal is to ensure that each cell type is only represented once in the list. Below are the current cell type annotations, which may contain duplicates:

    {list_with_duplicates}

    Remove any such duplicates from the list.

    ### Example:
    If you are provided with the following list of cell type labels:
    ["Natural killer cells", "natural killer cells", "NK cells", "Natural killer cells (NK cells) "T cells", "T-cells", "B cells", "B-cells"]

    A possible de-duplicated list could be:
    ["Natural killer cells (NK cells)", "T cells", "B cells"]

    ### Output format:
    Provide the updated cell type annotations as a list with no additional comments.
    """.strip()

    MAPPING_PROMPT = """
    Now, you need to map cell type annotations ('cell_labels_user') to the unique set of annotations you provided earlier ('cell_labels_global'). Here is the list of cell type annotations you need to map ('cell_labels_user'):
    {cell_type_list}

    Follow these rules:
    1. You must include all elements from the 'cell_labels_user' list exactly once.
    2. You must map each element from the 'cell_labels_user' list to a unique element from the 'cell_labels_global' list. Be careful with capitalization, spelling, and abbreviations.
    2. You cannot modify the names of any labels in either list. Use the labels exactly as they are provided.

    Your task it to find the mapping, not to modify the labels themselves.

     ### Output format:
    ```
    - original_name: name of the cell type as provided in 'cell_labels_user'.
    - unique_name: name of the cell type from 'cell_labels_global' that corresponds to the original name.
    ```
    """.strip()

    COLOR_PROMPT = """
    You need to assign meaningful colors to the following cell type labels:

    {cluster_names}

    Follow these rules:
    1. Use colors that are biologically meaningful: similar cell types should have similar colors (e.g., shades of the same color family), and unrelated cell types should have distinct colors.
    3. Use hexadecimal color codes (e.g., "#1f77b4").
    4. Do not use white, black, or grey colors.
    5. Do not modify the order of the cell type labels.
    6. Include all labels in the color assignmeent, and do not modify them in any way.

    ### Example:
    If the cell type annotations are:
        {example_cell_types}
    A possible color assignment could be:
        {example_color_assignment}

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
