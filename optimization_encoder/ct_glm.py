import torch


def bernoulli_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                  spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (n_bins, )
    :param spike_vector: shape (n_bins, )
    :return:
    '''

    prod = generator_sig * spike_vector
    log_sum_exp_term = torch.log(1.0 + torch.exp(generator_sig))
    return torch.mean(log_sum_exp_term - prod, dim=0)


# we can't pickle a JIT function
# not like the JIT helps us anyway
# @torch.jit.script
def fused_bernoulli_neg_ll_loss(generator_sig: torch.Tensor,
                                spike_vector: torch.Tensor) -> torch.Tensor:
    prod = generator_sig * spike_vector
    log_sum_exp_term = torch.log(1.0 + torch.exp(generator_sig))
    return torch.mean(log_sum_exp_term - prod)


def poisson_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (n_bins, )
    :param spike_vector: shape (n_bins, )
    :return:
    '''

    prod = generator_sig * spike_vector
    return torch.mean(torch.exp(generator_sig) - prod)

# we can't pickle a JIT function
# not like the JIT helps us anyway
# @torch.jit.script
def fused_poisson_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                      spike_vector: torch.Tensor) -> torch.Tensor:
    prod = generator_sig * spike_vector
    return torch.mean(torch.exp(generator_sig) - prod)
