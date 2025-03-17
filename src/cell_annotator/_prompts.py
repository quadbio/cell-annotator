from dataclasses import dataclass

from cell_annotator._constants import PromptExamples


@dataclass
class Prompts:
    """Prompts for scRNA-seq cell annotation tasks."""

    species: str = ""
    tissue: str = ""
    stage: str = ""

    def get_cell_type_prompt(self) -> str:
        """Generate the cell type prompt."""
        return f"Provide me a comprehensive list of cell types that are expected in {self.species} {self.tissue} at stage `{self.stage}`."

    def get_cell_type_marker_prompt(self, n_markers: int) -> str:
        """Generate the cell type marker prompt."""
        return f"Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Make sure that you provided markers for **each** cell type you mentioned above."

    def get_annotation_prompt(
        self,
        actual_markers_all: str,
        expected_markers: str,
        cluster_id: str,
        actual_markers_cluster: str,
        restrict_to_expected: bool = False,
    ) -> str:
        """Generate the annotation prompt."""
        restriction_clause = (
            "You must only use labels from the expected cell types listed above."
            if restrict_to_expected
            else "Some expected cell types might be absent, and additional cell types may also be present."
        )

        return f"""
        You need to annotate a {self.species} {self.tissue} dataset at stage `{self.stage}`. You found gene markers for each cluster and need to determine clusters' identities.
        Below is a short list of markers for each cluster:

        {actual_markers_all}

        We expect the following cell types (with associated marker genes) to be present in the dataset:
        {expected_markers}

        {restriction_clause}

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

    def get_order_prompt(
        self,
        unique_cell_types: str,
        example_unordered: str | None = None,
        example_ordered: str | None = None,
    ) -> str:
        """Generate the order prompt."""
        if example_unordered is None:
            example_unordered = ", ".join(PromptExamples.unordered_cell_types)
        if example_ordered is None:
            example_ordered = ", ".join(PromptExamples.ordered_cell_types)

        return f"""
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
        Provide the reordered cell type annotations as a list with no additional comments.
        """.strip()

    def get_duplicate_removal_prompt(self, list_with_duplicates: str) -> str:
        """Generate the duplicate removal prompt."""
        return f"""
        You need to remove duplicates from a list of cell type annotations. The same cell type might be included multiple times in the list, for example, with different capitalization, abbreviations, or synonyms. Thus, by duplicates, we mean the same, or an extremely similar cell type, being represented by two or more elements in the list. The goal is to ensure that each cell type is only represented once in the list. Below are the current cell type annotations, which may contain duplicates:

        {list_with_duplicates}

        Remove any such duplicates from the list.

        ### Output format:
        Provide the updated cell type annotations as a list with no additional comments.
        """.strip()

    def get_mapping_prompt(self, local_cell_type_list: str, global_cell_type_list: str, current_cell_type: str) -> str:
        """Generate the mapping prompt."""
        return f"""
        You're given two lists of cell type annotations. The first list, called 'cell_types_local', contains the cell type annotations from a local dataset. The second list, called 'cell_types_global', contains a unique set of cell type annotations that are used globally. Your task is to map the local cell type annotations to the global cell type annotations. Here are both lists:

        Local cell type annotations:
        {local_cell_type_list}

        Global cell type annotations:
        {global_cell_type_list}

        Now, map the following item from the local list to the global list: {current_cell_type}.

        Follow these rules:
        1. Map the local cell type annotation to the global cell type annotation that best represents it.
        2. Do not modify the corresponding entry from the global cell type annotation list in any way. Use the labels exactly as they are provided.
        """.strip()

    def get_color_prompt(
        self, cluster_names: str, example_cell_types: str | None = None, example_color_assignment: str | None = None
    ) -> str:
        """Generate the color assignment prompt."""
        if example_cell_types is None:
            example_cell_types = ", ".join(PromptExamples.color_mapping_example.keys())
        if example_color_assignment is None:
            example_color_assignment = "; ".join(
                f"{key}: {value}" for key, value in PromptExamples.color_mapping_example.items()
            )
        return f"""
        You need to assign meaningful colors to the following cell type labels:

        {cluster_names}

        Follow these rules:
        1. Use colors that are biologically meaningful: similar cell types should have similar colors (e.g., shades of the same color family, which are still easy to distinguish by eye), and unrelated cell types should have distinct colors.
        3. Use hexadecimal color codes (e.g., "#1f77b4").
        4. Do not use white, black, or grey colors.
        5. Do not modify the order of the cell type labels.
        6. Include all labels in the color assignment, and do not modify them in any way.

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

    def get_agent_description(self) -> str:
        """Generate the agent description."""
        return f"You're an expert bioinformatician, proficient in scRNA-seq analysis with background in {self.species} cell biology."
