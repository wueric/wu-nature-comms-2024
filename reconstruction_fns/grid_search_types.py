from collections import namedtuple

GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight', 'max_iter'])
GridSearchReconstructions = namedtuple('GridSearchReconstructions',
                                       ['ground_truth', 'reconstructions', 'mse', 'ssim', 'ms_ssim'])
