import numpy as np
import pickle

import sys
from collections import namedtuple

import argparse

GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight'])
GridSearchReconstructions = namedtuple('GridSearchReconstructions',
                                       ['ground_truth', 'reconstructions', 'mse', 'ssim', 'ms_ssim'])
EyeMovementGridSearchParams = namedtuple('EyeMovementGridSearchParams', ['eye_movement_weight'])
GridSearchParams1F = namedtuple('GridSearchParams1F', ['prior_weight'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parse the results of the grid search')
    parser.add_argument('grid_file', type=str, help='path to grid search file file')
    parser.add_argument('-g', '--gaussian_prior',
                        action='store_true', default=False, help='Gaussian prior grid search')
    parser.add_argument('-e', '--eye_movement_weight',
                        action='store_true', default=False, help='eye movement weight; default lambda and prior')
    parser.add_argument('-v', '--value',
                        action='store_true', default=False, help='return the value rather than the params; only for inspection')

    args = parser.parse_args()

    with open(args.grid_file, 'rb') as pfile:
        grid_results = pickle.load(pfile)

    fixed_ordering_keys = []
    fixed_ordering_values = []
    for key, val in grid_results.items():
        fixed_ordering_keys.append(key)
        fixed_ordering_values.append(val)

    sort_order = np.argsort([x.ms_ssim for x in fixed_ordering_values])

    best_parameters = fixed_ordering_keys[sort_order[-1]]

    if not args.value:
        if args.gaussian_prior:
            sys.stdout.write(str(best_parameters.prior_weight))
        elif args.eye_movement_weight:
            sys.stdout.write(str(best_parameters.eye_movement_weight))
        else:
            sys.stdout.write(';'.join(map(str, [best_parameters.lambda_start, best_parameters.lambda_end, best_parameters.prior_weight])))
    else:
        sys.stdout.write(str(fixed_ordering_values[sort_order[-1]].ms_ssim))




