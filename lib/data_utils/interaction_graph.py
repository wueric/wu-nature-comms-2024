import numpy as np

from typing import List, Dict, Any, Optional, Tuple, Union

from lib.data_utils.interaction_hashable import CellInteraction


class VertexInfo:

    def __init__(self, cell_id: int, cell_type: str):
        self.cell_id = cell_id
        self.cell_type = cell_type


class InteractionGraphEdge:

    def __init__(self,
                 source_cell_id: int,
                 dest_cell_id: int,
                 interaction_type: CellInteraction,
                 **kwargs):
        self.source_cell_id = source_cell_id
        self.dest_cell_id = dest_cell_id

        self.interaction_type = interaction_type
        self.additional_attributes = {key: val for key, val in kwargs.items()}

    @classmethod
    def construct_from_vertices(cls, source_vertex_info: VertexInfo, dest_vertex_info: VertexInfo, **kwargs) \
            -> 'InteractionGraphEdge':
        return InteractionGraphEdge(
            source_vertex_info.cell_id,
            dest_vertex_info.cell_id,
            CellInteraction(source_vertex_info.cell_type, dest_vertex_info.cell_type),
            **kwargs
        )

    def clone(self) -> 'InteractionGraphEdge':
        return InteractionGraphEdge(self.source_cell_id, self.dest_cell_id, self.interaction_type,
                                    **self.additional_attributes)

    def __repr__(self):
        return 'InteractionGraphEdge(source_cellid={0}, dest_cell_id={1}, additional_attributes={2})'.format(
            self.source_cell_id,
            self.dest_cell_id,
            self.additional_attributes.__repr__()
        )

    def __hash__(self):
        return (self.source_cell_id, self.dest_cell_id).__hash__()

    def __eq__(self, other):
        return isinstance(other, InteractionGraphEdge) \
               and self.source_cell_id == other.source_cell_id and self.dest_cell_id == other.dest_cell_id

    def assign_attributes(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.additional_attributes[key] = val

    def query_attribute(self, attribute_key: str) -> Any:
        return self.additional_attributes[attribute_key]

    def clone_attributes(self) -> Dict[str, Any]:
        return {key: val for key, val in self.additional_attributes.items()}


class InteractionGraph:
    '''
    Unified representation of interaction graph; should be used for everything
        that involves pairwise cell-to-cell interactions

    Though the interaction graph is undirected (by definition cell A interacting with cell B
        means that cell B must interact with cell A), we represent the interaction graph
        as a directed graph, with forward and backward edges corresponding to each interaction

    Intended functionality:
    (1) Keep track of existence of every pairwise cell-to-cell interaction [IMPLEMENTED]
    (2) Track correlation and other quantities that are associated with edges
        in the interaction graph [IMPLEMENTED]
    (3) Be able to dynamically update attributes associated with edges (i.e. we can
        set the correlation value corresponding to a given edge after the data structure
        has been created) [IMPLEMENTED]
    (4) Graph traversals (DFS, BFS, that sort of stuff)
    (5) Be able to query complicated stuff about interactions, for example,
        "get every ON parasol/ON parasol interaction, without double-counting", or
        "count the number of interactions cell_id cell has with ON parasols", or
        "get the selection indices of every interaction between ON parasols and OFF parasols",
        and so on

    Internal representation is a modified adjacency list, with some additional Dicts to keep track
        of cell types and index positions. Everything internal to the data structure will be keyed by
        the white noise cell id, since those are unique to a given cell (and its useful to be able to
        back out what the interaction is w.r.t. the original white noise dataset)

    '''

    def __init__(self):

        ### The following three data structures
        # * self.cell_id_to_type
        # * self.cell_type_to_id_list
        # * self.adjacency_list
        # are assumed to be consistent with each other at all times
        # Adding vertices requires updating all three simultaneously
        # Adding edges only requires updating self.adjacency_list

        # data structures to keep association between cell id and type handy
        self.cell_id_to_type = {}  # type: Dict[int, str]
        self.cell_type_to_id_list = {}  # type: Dict[str, List[int]]

        # main adjacency list data structure
        self.adjacency_list = {}  # type: Dict[int, List[InteractionGraphEdge]]

        # also a separate data structure to keep track of all of the possible
        # type interactions that this graph contains
        self.possible_interactions = []  # type: List[CellInteraction]

    def get_possible_interactions(self) -> List[CellInteraction]:
        return [x.clone() for x in self.possible_interactions]

    def _add_vertex(self, vertex_info: VertexInfo) -> None:
        '''
        Adds a graph vertex (corresponding to a single cell) to the graph data structuure
            if that vertex hasn't previously been added

        Does nothing if the vertex has previously been added

        :param vertex_info:
        :return:
        '''
        target_id = vertex_info.cell_id
        target_type = vertex_info.cell_type
        if target_id not in self.adjacency_list:
            self.cell_id_to_type[target_id] = target_type
            self.adjacency_list[target_id] = []

            if target_type not in self.cell_type_to_id_list:
                self.cell_type_to_id_list[target_type] = []
            self.cell_type_to_id_list[target_type].append(target_id)

    def add_update_interaction(self, vertex_a: VertexInfo, vertex_b: VertexInfo, **kwargs):
        '''
        Adds or updates an edge in the graph. Edge does not have to exist already

        :param vertex_a: information struct for vertex A
        :param vertex_b: information struct for vertex B
        :param kwargs: attributes associated with this edge
        :return:
        '''

        self._add_vertex(vertex_a)
        self._add_vertex(vertex_b)

        interaction_type = CellInteraction(vertex_a.cell_type, vertex_b.cell_type)

        forward_edge = InteractionGraphEdge(vertex_a.cell_id, vertex_b.cell_id, interaction_type, **kwargs)
        forward_exist_already = False
        for existing_forward_edge in self.adjacency_list[vertex_a.cell_id]:
            if existing_forward_edge == forward_edge:  # edge exists already, just update values
                existing_forward_edge.assign_attributes(**kwargs)
                forward_exist_already = True
        if not forward_exist_already:
            self.adjacency_list[vertex_a.cell_id].append(forward_edge)

        backward_edge = InteractionGraphEdge(vertex_b.cell_id, vertex_a.cell_id, interaction_type, **kwargs)
        backward_exist_already = False
        for existing_backward_edge in self.adjacency_list[vertex_b.cell_id]:
            if existing_backward_edge == backward_edge:
                existing_backward_edge.assign_attributes(**kwargs)
                backward_exist_already = True
        if not backward_exist_already:
            self.adjacency_list[vertex_b.cell_id].append(backward_edge)

        f_interaction = CellInteraction(vertex_a.cell_type, vertex_b.cell_type)
        if f_interaction not in self.possible_interactions:
            self.possible_interactions.append(f_interaction)

    def query_cell_interaction_edges(self,
                                     source_cell_id: int,
                                     dest_cell_type: Optional[str] = None) \
            -> List[InteractionGraphEdge]:
        '''
        Gets all edges associated with a specific cell, and some other optional criteria about destination
            cell type
        :param source_cell_id:
        :param dest_cell_type:
        :return:
        '''

        dest_edge_list = []  # type: List[InteractionGraphEdge]

        for edge in self.adjacency_list[source_cell_id]:

            if dest_cell_type is None or self.cell_id_to_type[edge.dest_cell_id] == dest_cell_type:
                dest_edge_list.append(edge.clone())
        return dest_edge_list

    def query_type_interaction_edges(self,
                                     source_cell_type: str,
                                     dest_cell_type: str,
                                     return_attrs: bool = False) \
            -> Union[Tuple[List[int], List[int]], Tuple[List[int], List[int], List[Dict[str, Any]]]]:

        '''
        Gets all edges associated with a specific cell-type interaction

        There are two cases that need to be handled separately:
            1. source_cell_type != dest_cell_type, in which case we don't have to
                worry about double counting
            2. source_cell_type == dest_cell_type, in which case we have to make
                sure that we don't double count the edges

        :param source_cell_id:
        :param source_cell_type:
        :param dest_cell_type:
        :param attributes:
        :return:
        '''

        exchangeable = (source_cell_type == dest_cell_type)
        if not exchangeable:
            type_a_ids = self.cell_type_to_id_list[source_cell_type]

            a_origin_list = []  # type: List[int]
            b_dest_list = []  # type: List[int]
            attr_list = []  # type: List[Dict[str, Any]]

            for type_a_id in type_a_ids:
                for edge in self.adjacency_list[type_a_id]:
                    if self.cell_id_to_type[edge.dest_cell_id] == dest_cell_type:
                        a_origin_list.append(type_a_id)
                        b_dest_list.append(edge.dest_cell_id)
                        attr_list.append(edge.clone_attributes())

            if return_attrs:
                return a_origin_list, b_dest_list, attr_list
            return a_origin_list, b_dest_list

        else:
            # in this case, we have to worry returning the same edge twice
            type_a_ids = self.cell_type_to_id_list[source_cell_type]

            origin_list = []  # type: List[int]
            dest_list = []  # type: List[int]
            attr_list = []  # type: List[Dict[str, Any]]

            edge_set = set()
            for type_a_id in type_a_ids:
                for edge in self.adjacency_list[type_a_id]:
                    if self.cell_id_to_type[edge.dest_cell_id] == source_cell_type \
                            and (type_a_id, edge.dest_cell_id) not in edge_set:
                        # this is a unique edge
                        origin_list.append(type_a_id)
                        dest_list.append(edge.dest_cell_id)
                        attr_list.append(edge.clone_attributes())

                        edge_set.add((type_a_id, edge.dest_cell_id))
                        edge_set.add((edge.dest_cell_id, type_a_id))

            if return_attrs:
                return origin_list, dest_list, attr_list
            return origin_list, dest_list
