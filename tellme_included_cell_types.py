import argparse
import lib.dataset_config_parser.dataset_config_parser as dcp

import sys
from typing import List

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parse the results of the GLM fit grid search')
    parser.add_argument('cfg_path', type=str, help='path to config')

    args = parser.parse_args()

    config_settings_dict = dcp.read_config_file(args.cfg_path)

    cell_types = config_settings_dict['CellTypes']  # type: List[str]

    sys.stdout.write(';'.join(cell_types))
