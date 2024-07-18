import numpy as np
import visionloader as vl
import argparse
import pickle

import lib.data_utils.matched_cells_struct as mcs
from lib.data_utils.interaction_graph import InteractionGraph, VertexInfo

import lib.data_utils.cell_curation as curate
from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection, \
    FeaturizedInteractionSection

from typing import List, Dict, Tuple

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find interaction pairs from a known good matching')
    parser.add_argument('cfg_path', type=str, help='path to config file')

    args = parser.parse_args()

    config_settings_dict = read_config_file(args.cfg_path)

    cell_types = config_settings_dict['CellTypes']  # type: List[str]

    ref_dataset_info = config_settings_dict['ReferenceDataset']

    ref_dataset = vl.load_vision_data(ref_dataset_info.path,
                                      ref_dataset_info.name,
                                      include_neurons=True,
                                      include_params=True,
                                      include_ei=True,
                                      include_sta=True)

    if ref_dataset_info.classification_file is not None:
        ref_dataset.update_cell_type_classifications_from_text_file(ref_dataset_info.classification_file)

    with open(config_settings_dict['responses_ordered'], 'rb') as picklefile:
        ordered_matched_cells = pickle.load(picklefile)  # type: mcs.OrderedMatchedCellsStruct

    print('Constructing pairwise interaction graph')
    interaction_graph = InteractionGraph()

    # first add all the vertices, since we do need to represent all of the vertices
    # even if they aren't connected to anything
    for cell_type in ordered_matched_cells.get_cell_types():
        for cell_id in ordered_matched_cells.get_reference_cell_order(cell_type):
            interaction_graph._add_vertex(VertexInfo(cell_id, cell_type))

    # now based on the matched cells, figure out what interaction terms we want to include
    # Instead of using NND distance for interaction criteria, we just use outright RF distance
    # in Vision STA units
    for feature_interaction in config_settings_dict[FeaturizedInteractionSection.FEATURIZED_INTERACTION_LIST]:
        cell_type_a, cell_type_b = feature_interaction.interaction_types()
        interaction_distance = feature_interaction.dist

        # get the matched cell ids of each type, and calculate which pairs are relevant
        of_type_a = ordered_matched_cells.get_reference_cell_order(cell_type_a)
        of_type_b = ordered_matched_cells.get_reference_cell_order(cell_type_b)

        interaction_list = curate.get_neighbors_between_lists(ref_dataset,
                                                              of_type_a,
                                                              of_type_b,
                                                              interaction_distance)

        # compute additional numbers to quantify the distance of the interaction
        # may be useful for parameterizing the strength of interaction correlations later on
        interaction_distance_params = curate.calculate_mosaic_interaction_info_from_neighbor_pairs(
            ref_dataset,
            cell_type_a,
            cell_type_b,
            interaction_list
        )

        for (aa, bb), distance_kwargs in zip(interaction_list, interaction_distance_params):
            vertex_aa, vertex_bb = VertexInfo(aa, cell_type_a), VertexInfo(bb, cell_type_b)
            interaction_graph.add_update_interaction(vertex_aa, vertex_bb, **distance_kwargs)

    # now write the output out to pickle
    with open(config_settings_dict['featurized_interactions_ordered'], 'wb') as picklefile:
        pickle.dump(interaction_graph, picklefile)

    print('Done')
