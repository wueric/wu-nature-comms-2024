import argparse
from typing import List

from lib.dataset_config_parser.trained_model_config_parser import write_glm_yaml_file_paths2


def parse_fnames_to_dict(fnames_arglist: List[str]):
    print(fnames_arglist)
    ret_dict = {}
    for ix in range(0, len(fnames_arglist), 2):
        ct, ct_key = fnames_arglist[ix], fnames_arglist[ix + 1]
        ret_dict[ct] = ct_key
    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Write model .yaml file in an automated way')
    parser.add_argument('yaml_path', type=str, help='YAML path')
    parser.add_argument('basepath', type=str, help='path to root folder')
    parser.add_argument('fnames', type=str, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    ct_fnames_dict = parse_fnames_to_dict(args.fnames)

    write_glm_yaml_file_paths2(args.yaml_path,
                               args.basepath,
                               ct_fnames_dict)
