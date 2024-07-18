import argparse
import pickle

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Print out the available WN cell ids available for use in reconstruction (i.e. matched properly)')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    args = parser.parse_args()

    cell_type = args.cell_type
    config_settings = dcp.read_config_file(args.cfg_file)

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    print(','.join([str(x) for x in cells_ordered.get_reference_cell_order(cell_type)]))
