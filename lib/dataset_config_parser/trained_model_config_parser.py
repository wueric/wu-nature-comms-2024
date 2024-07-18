import yaml
import os
from typing import Dict, Tuple


def parse_prefit_glm_paths(yaml_path: str) -> Dict[str, str]:
    with open(yaml_path, 'r') as yaml_file:
        unparsed_dict = yaml.safe_load(yaml_file)

    basepath = unparsed_dict['basepath']
    full_path_dict = {}
    for key, val in unparsed_dict.items():
        if key != 'basepath':
            full_path_dict[key] = os.path.join(basepath, val)

    return full_path_dict


def parse_mixnmatch_path_yaml(yaml_path: str) \
        -> Tuple[Dict[str, str], Dict[str, int]]:

    with open(yaml_path, 'r') as yaml_file:
        unparsed_dict = yaml.safe_load(yaml_file)

    full_path_dict = {}
    jitter_time_dict = {}
    for key, val in unparsed_dict.items():

        file_ext, jitter_samples = val
        jitter_samples = int(jitter_samples)
        full_path_dict[key] = file_ext
        jitter_time_dict[key] = jitter_samples

    return full_path_dict, jitter_time_dict


def write_glm_yaml_file_paths2(yaml_path: str,
                               basepath: str,
                               cell_type_names: Dict[str, str]):

    towrite_dict = cell_type_names.copy()
    towrite_dict['basepath'] = basepath

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml.dump(towrite_dict))


def write_glm_yaml_file_paths(yaml_path: str,
                              base_cell_type_ext_dict: Dict[str, str]) -> None:
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml.dump(base_cell_type_ext_dict))
