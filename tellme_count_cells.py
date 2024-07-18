import pickle
import argparse

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Match cells between WN and nscenes with cosine similarity')
    parser.add_argument('cfg_path', type=str, help='path to config file')
    args = parser.parse_args()

    config_settings = dcp.read_config_file(args.cfg_path)

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    total_cells = 0
    for cell_type in ct_order:

        n_cells_of_type = len(cells_ordered.get_reference_cell_order(cell_type))
        print(f'{n_cells_of_type} {cell_type}')
        total_cells += n_cells_of_type

    print(f'{total_cells} total cells')