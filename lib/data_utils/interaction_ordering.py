from typing import List, Dict, Tuple, Any

import numpy as np

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.interaction_graph import InteractionGraph
from lib.data_utils.interaction_hashable import CellInteraction


class AllInteractionsOrdering:
    '''
    Data structure to keep track of the ordering of all of the
        pairwise interactions included in the analysis

    This is a write-once, read-many data structure. Does not support making
        changes to it after creation.

    Functionality of this data structure (what it needs to be good at):
    (1) Getting the indices of the interacting cells, so they can easily
        be selected from the cell-type-concatenated spike vector
    (2) Querying the interaction by type, so that we can easily fetch
        attributes about all interactions of a particular kind

    '''

    def __init__(self,
                 interactions_graph: InteractionGraph,
                 cell_type_ordering: OrderedMatchedCellsStruct):
        # keeps track of the interaction ordering (useful for reinflating trained models)
        self.interaction_ordering = interactions_graph.get_possible_interactions()

        # keeps track of the order of interactions of a given type, so that we can manually inspect
        # cell pairs after the fact
        self.interaction_cell_ids_by_type = {}  # type: Dict[CellInteraction, Tuple[List[int], List[int]]]

        self.interaction_attrs_by_type = {} # type: Dict[CellInteraction, List[Dict[str, Any]]]

        self.interaction_specific_index_select = {}  # type: Dict[CellInteraction, np.ndarray]

        cat_index_select = []
        cell_type_offsets = cell_type_ordering.compute_concatenated_cell_type_index_offset()
        for interaction_hashable in self.interaction_ordering:
            type1, type2 = interaction_hashable.interaction_types()

            type1_cell_id, type2_cell_id, interaction_attrs = interactions_graph.query_type_interaction_edges(
                type1,
                type2,
                return_attrs=True
            )

            unadjusted_type1_indices = np.array(
                cell_type_ordering.get_idx_for_same_type_cell_id_list(type1, type1_cell_id),
                dtype=np.int64)
            unadjusted_type2_indices = np.array(
                cell_type_ordering.get_idx_for_same_type_cell_id_list(type2, type2_cell_id),
                dtype=np.int64)

            adjusted_type1_indices = unadjusted_type1_indices + cell_type_offsets[type1]
            adjusted_type2_indices = unadjusted_type2_indices + cell_type_offsets[type2]

            # shape (n_interactions, 2)
            stacked_indices = np.stack([adjusted_type1_indices, adjusted_type2_indices], axis=1)
            cat_index_select.append(stacked_indices)

            self.interaction_cell_ids_by_type[interaction_hashable] = (type1_cell_id, type2_cell_id)
            self.interaction_attrs_by_type[interaction_hashable] = interaction_attrs
            self.interaction_specific_index_select[interaction_hashable] = stacked_indices

        # shape (n_total_interactions, 2)
        self.cat_index_select = np.concatenate(cat_index_select, axis=0)

    def get_type_specific_interaction_ids(self, interaction_type: CellInteraction) \
            -> Tuple[List[int], List[int]]:
        return self.interaction_cell_ids_by_type[interaction_type]

    def get_type_specific_interaction_attrs(self, interaction_type: CellInteraction) \
            -> List[Dict[str, Any]]:
        return self.interaction_attrs_by_type[interaction_type]

    def get_type_specific_interaction_attr_vals(self,
                                                interaction_type: CellInteraction,
                                                attr_key : str) -> np.ndarray:
        return np.array([x[attr_key] for x in self.interaction_attrs_by_type[interaction_type]], dtype=np.float64)

    def get_type_specific_index_select(self, interaction_type: CellInteraction) -> np.ndarray:
        return self.interaction_specific_index_select[interaction_type]

    def get_cat_index_select(self) -> np.ndarray:
        return self.cat_index_select.copy()

    def get_interaction_type_ordering(self) -> List[CellInteraction]:
        return [x.clone() for x in self.interaction_ordering]
