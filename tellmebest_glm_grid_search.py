import numpy as np

import argparse
import pickle
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parse the results of the GLM fit grid search')
    parser.add_argument('grid_file', type=str, help='path to grid search file file')
    parser.add_argument('-l21', '--fix_l21', type=float, default=None,
                        help='get the best L1 regularizer for a fixed L21 value',)
    parser.add_argument('-fb', '--fbonly', action='store_true', default=False,
                        help='set flag for best L1 parameter for FB only GLM')
    parser.add_argument('-c', '--combined', action='store_true', default=False,
                        help='choose based on combined loss')

    args = parser.parse_args()

    with open(args.grid_file, 'rb') as pfile:
        contents = pickle.load(pfile)

    if args.combined:
        mean_test_loss = np.mean(contents['jitter_test_loss'], axis=0) + \
                         np.mean(contents['flashed_test_loss'], axis=0)
    else:
        mean_test_loss = np.mean(contents['test_loss'], axis=0)

    if args.fbonly:
        l1_keys = contents['L1']
        l1_sel = np.argmin(mean_test_loss)
        opt_l1_value = l1_keys[l1_sel]
        sys.stdout.write(str(opt_l1_value))

    else:

        if args.fix_l21 is None:
            l1_keys, l21_keys = contents['L1'], contents['L21']
            (l1_sel, l21_sel) = np.unravel_index(np.argmin(mean_test_loss),
                                                 mean_test_loss.shape)
            opt_l1_value = l1_keys[l1_sel]
            opt_l21_value = l21_keys[l21_sel]
            sys.stdout.write(';'.join(map(str, [opt_l1_value, opt_l21_value])))

        else:
            l1_keys, l21_keys = contents['L1'], contents['L21']
            l21_ix, = np.where(l21_keys == args.fix_l21)
            sliced_test_loss = mean_test_loss[:, l21_ix]

            min_l1_ix = np.argmin(sliced_test_loss)

            opt_l1_value = l1_keys[min_l1_ix]
            sys.stdout.write(str(opt_l1_value))
