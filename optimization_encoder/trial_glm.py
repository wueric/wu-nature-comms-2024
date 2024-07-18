from dataclasses import dataclass
import pickle

import numpy as np

import torch

from typing import Tuple, Dict, Union, Callable, Any


def batch_bernoulli_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                        spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (batch, n_bins)
    :param spike_vector: shape (batch, n_bins)
    :return:
    '''

    prod = generator_sig * spike_vector
    log_sum_exp_term = torch.log(1.0 + torch.exp(generator_sig))
    return torch.mean(torch.sum(log_sum_exp_term - prod, dim=1), dim=0)


def mean_bin_batch_bernoulli_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                                 spike_vector: torch.Tensor) -> torch.Tensor:
    prod = generator_sig * spike_vector
    log_sum_exp_term = torch.log(1.0 + torch.exp(generator_sig))
    return torch.mean(log_sum_exp_term - prod)


def batch_poisson_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                      spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (batch, n_bins)
    :param spike_vector: shape (batch, n_bins)
    :return:
    '''

    prod = generator_sig * spike_vector
    return torch.mean(torch.exp(generator_sig) - prod)


def make_batch_binomial_spiking_neg_ll_loss(binom_max: int) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def batch_binomial_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                           spike_vector: torch.Tensor) -> torch.Tensor:
        '''

        :param generator_sig: shape (batch, n_bins)
        :param spike_vector: shape (batch, n_bins)
        :return:
        '''

        prod = generator_sig * spike_vector

        log_term = binom_max * torch.log(1.0 + torch.exp(generator_sig))

        return torch.mean(torch.sum(log_term - prod, dim=1), dim=0)

    return batch_binomial_spiking_neg_ll_loss


def make_batch_mean_binomial_spiking_neg_ll_loss(binom_max: int) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def batch_binomial_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                           spike_vector: torch.Tensor) -> torch.Tensor:
        '''

        :param generator_sig: shape (batch, n_bins)
        :param spike_vector: shape (batch, n_bins)
        :return:
        '''

        prod = generator_sig * spike_vector

        log_term = binom_max * torch.log(1.0 + torch.exp(generator_sig))

        return torch.mean(log_term - prod)

    return batch_binomial_spiking_neg_ll_loss


@dataclass
class FittedLNP:
    cell_id: int

    spatial_weights: np.ndarray
    spatial_bias: np.ndarray

    timecourse_weights: np.ndarray

    fitting_params: Dict[str, Any]

    train_loss: float
    test_loss: float


@dataclass
class FittedLNPFamily:
    fitted_models: Dict[int, FittedLNP]

    spatial_basis: Union[np.ndarray, None]
    timecourse_basis: np.ndarray


def load_fitted_lnp_families(type_to_path_dict: Dict[str, str]) \
        -> Dict[str, FittedLNPFamily]:
    output_dict = {}  # type: Dict[str, FittedLNPFamily]
    for key, path in type_to_path_dict.items():
        with open(path, 'rb') as pfile:
            output_dict[key] = pickle.load(pfile)

    return output_dict


@dataclass
class FittedFBOnlyGLM:
    cell_id: int
    spatial_weights: np.ndarray
    spatial_bias: np.ndarray

    timecourse_weights: np.ndarray

    feedback_weights: np.ndarray

    fitting_params: Dict[str, Any]

    train_loss: float
    test_loss: float


@dataclass
class FittedFBOnlyGLMFamily:
    fitted_models: Dict[int, FittedFBOnlyGLM]

    spatial_basis: Union[np.ndarray, None]
    timecourse_basis: np.ndarray
    feedback_basis: np.ndarray


@dataclass
class FittedGLM:
    cell_id: int
    spatial_weights: np.ndarray
    spatial_bias: np.ndarray

    timecourse_weights: np.ndarray

    feedback_weights: np.ndarray

    coupling_cells_weights: Tuple[np.ndarray, np.ndarray]

    fitting_params: Dict[str, Any]

    train_loss: float
    test_loss: float


@dataclass
class FittedGLMFamily:
    fitted_models: Dict[int, FittedGLM]

    spatial_basis: Union[np.ndarray, None]
    timecourse_basis: np.ndarray
    feedback_basis: np.ndarray
    coupling_basis: np.ndarray


def load_fitted_glm_families(type_to_path_dict: Dict[str, str]) \
        -> Union[Dict[str, FittedGLMFamily], Dict[str, FittedFBOnlyGLMFamily]]:
    output_dict = {}  # type: Union[Dict[str, FittedGLMFamily], Dict[str, FittedFBOnlyGLMFamily]]
    for key, path in type_to_path_dict.items():
        with open(path, 'rb') as pfile:
            output_dict[key] = pickle.load(pfile)
    return output_dict
