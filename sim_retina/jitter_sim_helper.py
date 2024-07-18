from typing import List, Optional, Callable, Dict, Tuple, Union

import numpy as np
import torch
import tqdm

from lib.data_utils.dynamic_data_util import _JitterReconstructionBrownianMovieBlock
from lib.data_utils import dynamic_data_util as ddu
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_specific_ttl_corrections.nsbrownian_ttl_structure_corrections import SynchronizedNSBrownianSection
from sim_retina.single_cell_sim import CTSimGLM, CTSimFBOnlyGLM
from sim_retina.load_model_for_sim import SingleCellGLMForSim, SingleCellUncoupledGLMForSim

from movie_upsampling import batch_compute_interval_overlaps, batch_flat_sparse_upsample_transpose_cuda


class SingleCellRepeatsJitteredMovieDataloader:
    '''
    Dataloader for getting stuff from the dataset for simulating
        repeats for a single center cell

    Fetches the cropped stimulus for that cell, the spike train for
        that cell, and the spike trains for the coupled cell

    '''

    def __init__(self,
                 loaded_brownian_movies: List[ddu.LoadedBrownianMovies],
                 cell_matching: OrderedMatchedCellsStruct,
                 bin_width: int,
                 image_rescale_lambda: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 n_stimuli_to_get: int = 1):

        self.cell_matching = cell_matching

        self.bin_width = bin_width

        self.image_rescale_lambda = image_rescale_lambda

        self.n_stimuli_to_get = n_stimuli_to_get

        self.data_blocks = []  # type: List[_JitterReconstructionBrownianMovieBlock]
        self.cell_ids_to_bin = {}  # type: Dict[str, List[List[int]]]
        self.n_stimuli = 0

        for loaded_brownian_movie in loaded_brownian_movies:

            # this is guaranteed to be the repeat partition
            synchro_section_list = loaded_brownian_movie.repeat_blocks  # type: List[SynchronizedNSBrownianSection]
            for synchro_block in synchro_section_list:
                self.data_blocks.append(_JitterReconstructionBrownianMovieBlock(
                    loaded_brownian_movie.name,
                    loaded_brownian_movie.dataset,
                    synchro_block
                ))

                self.n_stimuli = synchro_block.n_stimuli - self.n_stimuli_to_get

        self.n_repeats = len(self.data_blocks)

    def fetch_data_repeat_stimulus_only(self,
                                        stim_ix: int) \
            -> np.ndarray:
        '''
        (This hits disk, so it might be expensive)
        (Call only once for each stimulus, caller will figure out the cropping situation)
        :param stim_ix:
        :param stimulus_crop_box:
        :return:
        '''

        relev_data_block = self.data_blocks[0]  # doesn't matter which repeat it is
        # they all have the same stimulus
        synchro_block = relev_data_block.timing_synchro

        snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
            stim_ix, stim_ix + self.n_stimuli_to_get)

        if self.image_rescale_lambda is not None:
            return self.image_rescale_lambda(snippet_frames)
        return snippet_frames

    def fetch_data_repeat_spikes_and_bin_times(self,
                                               stim_ix: int,
                                               repeat_ix: int,
                                               center_cell_id: int,
                                               coupled_cell_ids: Optional[List[int]] = None) \
            -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, np.ndarray]:
        '''

        :param stim_ix:
        :param repeat_ix:
        :param center_cell_id:
        :param coupled_cell_ids:
        :return:
        '''

        has_coupled_cells = (coupled_cell_ids is not None) and (len(coupled_cell_ids) > 0)

        relev_data_block = self.data_blocks[repeat_ix]
        synchro_block = relev_data_block.timing_synchro

        snippet_transitions = synchro_block.get_snippet_transition_times(stim_ix, stim_ix + self.n_stimuli_to_get)
        start_sample, end_sample = synchro_block.get_snippet_sample_times(stim_ix, stim_ix + self.n_stimuli_to_get)

        bin_start = max(start_sample, int(np.ceil(snippet_transitions[0])))
        bin_end = min(end_sample, int(np.floor(snippet_transitions[-1])))

        spike_bins = np.r_[bin_start:bin_end:self.bin_width]

        cells_to_bin = [self.cell_matching.get_match_ids_for_ds(center_cell_id, relev_data_block.vision_name), ]
        if has_coupled_cells:
            for coupled_cell_id in coupled_cell_ids:
                cells_to_bin.append(
                    self.cell_matching.get_match_ids_for_ds(coupled_cell_id, relev_data_block.vision_name))

        # bin spikes for this
        binned_spikes = ddu.movie_bin_spikes_multiple_cells2(
            relev_data_block.vision_dataset,
            cells_to_bin,
            spike_bins
        )

        center_cell_spikes = binned_spikes[0, :]

        coupled_cell_spikes = None
        if has_coupled_cells:
            coupled_cell_spikes = binned_spikes[1:, :]

        shift_amount = spike_bins[0]
        shifted_spike_bins = spike_bins - shift_amount
        shifted_snippet_transitions = (snippet_transitions - shift_amount).astype(np.float32)

        # also modify shifted_snippet_transitions so that it ends after the last spike
        # this won't affect the final output, since we are going to throw away anything
        # that results from the padding anyway

        return center_cell_spikes, coupled_cell_spikes, shifted_spike_bins, shifted_snippet_transitions


def batch_combine_max_with_zero_pad(
        repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
        center_cell_id: int,
        coupled_cell_ids: Union[List[int], None],
        stim_ix: int,
        repeat_low_high: Tuple[int, int]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low, high = repeat_low_high

    accum_center, accum_coupled = [], []
    accum_bin_times = []
    accum_frame_times = []
    max_bin_times = -1

    n_coupled_cells = 0
    if coupled_cell_ids is not None:
        n_coupled_cells = len(coupled_cell_ids)

    for bb in range(low, high):
        center_spikes, coupled_spikes, bin_times, frame_transitions = repeats_dataloader.fetch_data_repeat_spikes_and_bin_times(
            stim_ix, bb,
            center_cell_id, coupled_cell_ids
        )

        max_bin_times = max(bin_times.shape[0], max_bin_times)

        accum_center.append(center_spikes)
        accum_coupled.append(coupled_spikes)
        accum_bin_times.append(bin_times)
        accum_frame_times.append(frame_transitions)

    # now construct a batch
    # zero pad by adding samples onto the back for the shorter spike trains
    # add additional bin cutoff times accordingly (take the median bin width or something
    # so we don't have any surprise blow-ups)
    # mark which bins are real and which bins are padding so we can deconstruct the batch later

    # the number of frames is fixed no matter what, so padding is not needed there
    is_real_data = np.zeros((high - low, max_bin_times - 1), dtype=bool)
    padded_center_spikes = np.zeros((high - low, max_bin_times - 1), dtype=np.float32)

    padded_coupled_spikes = None
    if coupled_cell_ids is not None:
        padded_coupled_spikes = np.zeros((high - low, n_coupled_cells, max_bin_times - 1), dtype=np.float32)

    padded_bin_times = np.zeros((high - low, max_bin_times), dtype=np.float32)
    for ix, (center, coupled, bin_times, frame_transitions) in \
            enumerate(zip(accum_center, accum_coupled, accum_bin_times, accum_frame_times)):
        n_bins_trial = center.shape[0]

        is_real_data[ix, :n_bins_trial] = True
        padded_center_spikes[ix, :n_bins_trial] = center[:]

        if coupled_cell_ids is not None:
            padded_coupled_spikes[ix, :, :n_bins_trial] = coupled[:, :]

        median_bin_diff = np.median(bin_times[1:] - bin_times[:-1])
        n_needs_padding = max_bin_times - bin_times.shape[0]
        last_bin_time = bin_times[-1]
        generated_padded = last_bin_time + (np.r_[1:1 + n_needs_padding] * median_bin_diff)

        padded_bin_times[ix, :bin_times.shape[0]] = bin_times[:]
        padded_bin_times[ix, bin_times.shape[0]:] = generated_padded

    # shape (batch, n_frames + 1)
    all_frame_times = np.stack(accum_frame_times, axis=0)

    # modify all_frame_times so that it ends after the last bin time
    # this won't affect the final result since all of this would occur
    # in the padding region that gets thrown away later

    # this is so that we can upsample the movie, which makes the assumption
    # that the spike bins are entirely contained within the movie bins
    largest_padded_bin_times = padded_bin_times[:, -1]
    all_frame_times[:, -1] = largest_padded_bin_times[:]

    return is_real_data, padded_center_spikes, padded_coupled_spikes, padded_bin_times, all_frame_times


def batch_separate_unpad(batch_simulated_spike_trains: np.ndarray,
                         is_real_data: np.ndarray) -> List[np.ndarray]:
    '''

    :param batch_simulated_spike_trains: shape (batch, n_bins)
    :param is_real_data: shape (batch, n_bins), boolean-valued
    :return:
    '''

    batch = batch_simulated_spike_trains.shape[0]
    acc = []
    for ix in range(batch):
        acc.append(batch_simulated_spike_trains[ix, is_real_data[ix, :]])
    return acc


def ct_full_glm_simulate_single_cell_single_stimulus(repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
                                                     sim_model: CTSimGLM,
                                                     stim_ix: int,
                                                     center_cell_id: int,
                                                     coupled_cell_ids: List[int],
                                                     h_w_slice: Tuple[Tuple[int, int], Tuple[int, int]],
                                                     device: torch.device,
                                                     max_batch_size: int = 8) -> List[np.ndarray]:
    n_repeats = repeats_dataloader.n_repeats
    crop_bounds_h, crop_bounds_w = h_w_slice
    crop_h_low, crop_h_high = crop_bounds_h
    crop_w_low, crop_w_high = crop_bounds_w

    stimulus = repeats_dataloader.fetch_data_repeat_stimulus_only(stim_ix)

    # shape (n_frames, crop_height, crop_width)
    cropped_stimulus = stimulus[:, crop_h_low:crop_h_high, crop_w_low:crop_w_high]

    # move stimulus to GPU
    # shape (n_frames, crop_height, crop_width)
    cropped_stimulus_torch = torch.tensor(cropped_stimulus, dtype=torch.float32, device=device)

    # shape (n_frames, n_pix)
    cropped_stimulus_flat_torch = cropped_stimulus_torch.reshape(cropped_stimulus_torch.shape[0], -1)

    repeat_acc = []
    for low in range(0, n_repeats, max_batch_size):
        high = min(n_repeats, low + max_batch_size)
        batch_size = high - low

        is_real, batched_center, batched_coupled, batched_bin_times, batched_frame_transitions = batch_combine_max_with_zero_pad(
            repeats_dataloader,
            center_cell_id,
            coupled_cell_ids,
            stim_ix,
            (low, high)
        )
        # compute the interval overlap and selection indices
        batched_frame_sel, batched_interval_weights, _, _ = batch_compute_interval_overlaps(
            batched_frame_transitions, batched_bin_times)

        with torch.inference_mode():
            batched_center_torch = torch.tensor(batched_center, dtype=torch.float32, device=device)
            # shape (batch, n_bins_filter)
            batched_spikes_init_torch = batched_center_torch[:, :sim_model.n_bins_filter]

            batched_coupled_torch = torch.tensor(batched_coupled, dtype=torch.float32, device=device)

            batched_frame_sel_torch = torch.tensor(batched_frame_sel, dtype=torch.long, device=device)
            batched_frame_weights_torch = torch.tensor(batched_interval_weights, dtype=torch.float32, device=device)

            batch_upsampled_stimulus_torch = batch_flat_sparse_upsample_transpose_cuda(
                cropped_stimulus_flat_torch[None, :, :].expand(batch_size, -1, -1).contiguous(),
                batched_frame_sel_torch,
                batched_frame_weights_torch,
            )

            # shape (batch, n_bins)
            batched_simulated_spike_train = sim_model.simulate_cell(
                batch_upsampled_stimulus_torch,
                batched_spikes_init_torch,
                batched_coupled_torch,
                n_repeats=1
            ).squeeze(1).detach().cpu().numpy()

        unbatched_simulated_spike_trains = batch_separate_unpad(
            batched_simulated_spike_train,
            is_real
        )

        repeat_acc.extend(unbatched_simulated_spike_trains)

        del batched_center_torch, batched_spikes_init_torch, batched_coupled_torch
        del batched_frame_sel_torch, batched_frame_weights_torch, batch_upsampled_stimulus_torch

    del cropped_stimulus_flat_torch, cropped_stimulus_torch

    return repeat_acc


def ct_full_glm_simulate_single_cell(full_glm_sim_parameters: SingleCellGLMForSim,
                                     sim_spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                                     h_w_slice: Tuple[Tuple[int, int], Tuple[int, int]],
                                     repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
                                     device: torch.device,
                                     max_batch_size: int = 8) -> List[List[np.ndarray]]:
    '''
    Subtleties here in the implementation:
        (1) Batching multiple stimuli or different repeats of the same stimuli
            doesn't make much sense, since each stimulus presentation may take
            a different amount of time

            This also means that the length of the simulated spike train for each
            repeat or different stimulus presentation may be different

        (2) It makes sense to generate all repeats for the same stimulus/same cell
            in rapid succession, since we can use the same model and can reuse fetching
            the stimulus frame

    :param full_glm_sim_parameters:
    :param repeats_dataloader
    :param device:
    :return: outer list over each different stimulus
             inner list over each repeat simulation

             each np.ndarray has shape (n_bins, ) corresponding to the simulated
             spike train for the center cell. n_bins may be different for each element
    '''

    # let the garbage collector worry about GPU resources
    n_stimuli = repeats_dataloader.n_stimuli

    # create simulation model on GPU
    ct_sim_glm = CTSimGLM(
        full_glm_sim_parameters.cropped_spatial_filter.reshape(-1),
        full_glm_sim_parameters.stimulus_timecourse,
        full_glm_sim_parameters.feedback_filter,
        full_glm_sim_parameters.coupling_params[0],
        full_glm_sim_parameters.bias,
        sim_spike_generation_callable,
        dtype=torch.float32
    ).to(device)

    simulated = []
    pbar = tqdm.tqdm(total=n_stimuli)
    for stim_ix in range(n_stimuli):
        simulated.append(ct_full_glm_simulate_single_cell_single_stimulus(
            repeats_dataloader,
            ct_sim_glm,
            stim_ix,
            full_glm_sim_parameters.cell_id,
            list(full_glm_sim_parameters.coupling_params[1]),
            h_w_slice,
            device,
            max_batch_size=max_batch_size))

        pbar.update(1)

    pbar.close()

    del ct_sim_glm

    return simulated


def ct_uncoupled_glm_simulate_single_cell_single_stimulus(repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
                                                          sim_model: CTSimFBOnlyGLM,
                                                          stim_ix: int,
                                                          center_cell_id: int,
                                                          h_w_slice: Tuple[Tuple[int, int], Tuple[int, int]],
                                                          device: torch.device,
                                                          max_batch_size: int = 8) -> List[np.ndarray]:
    n_repeats = repeats_dataloader.n_repeats
    crop_bounds_h, crop_bounds_w = h_w_slice
    crop_h_low, crop_h_high = crop_bounds_h
    crop_w_low, crop_w_high = crop_bounds_w

    stimulus = repeats_dataloader.fetch_data_repeat_stimulus_only(stim_ix)

    # shape (n_frames, crop_height, crop_width)
    cropped_stimulus = stimulus[:, crop_h_low:crop_h_high, crop_w_low:crop_w_high]

    # move stimulus to GPU
    # shape (n_frames, crop_height, crop_width)
    cropped_stimulus_torch = torch.tensor(cropped_stimulus, dtype=torch.float32, device=device)

    # shape (n_frames, n_pix)
    cropped_stimulus_flat_torch = cropped_stimulus_torch.reshape(cropped_stimulus_torch.shape[0], -1)

    repeat_acc = []
    for low in range(0, n_repeats, max_batch_size):
        high = min(n_repeats, low + max_batch_size)
        batch_size = high - low

        is_real, batched_center, _, batched_bin_times, batched_frame_transitions = batch_combine_max_with_zero_pad(
            repeats_dataloader,
            center_cell_id,
            None,
            stim_ix,
            (low, high)
        )

        # compute the interval overlap and selection indices
        batched_frame_sel, batched_interval_weights, _, _ = batch_compute_interval_overlaps(
            batched_frame_transitions, batched_bin_times)

        with torch.inference_mode():
            batched_center_torch = torch.tensor(batched_center, dtype=torch.float32, device=device)
            # shape (batch, n_bins_filter)
            batched_spikes_init_torch = batched_center_torch[:, :sim_model.n_bins_filter]

            batched_frame_sel_torch = torch.tensor(batched_frame_sel, dtype=torch.long, device=device)
            batched_frame_weights_torch = torch.tensor(batched_interval_weights, dtype=torch.float32, device=device)

            batch_upsampled_stimulus_torch = batch_flat_sparse_upsample_transpose_cuda(
                cropped_stimulus_flat_torch[None, :, :].expand(batch_size, -1, -1).contiguous(),
                batched_frame_sel_torch,
                batched_frame_weights_torch,
            )

            # shape (batch, n_bins)
            batched_simulated_spike_train = sim_model.simulate_cell(
                batch_upsampled_stimulus_torch,
                batched_spikes_init_torch,
                n_repeats=1
            ).squeeze(1).detach().cpu().numpy()

        unbatched_simulated_spike_trains = batch_separate_unpad(
            batched_simulated_spike_train,
            is_real
        )

        repeat_acc.extend(unbatched_simulated_spike_trains)

        del batched_center_torch, batched_spikes_init_torch
        del batched_frame_sel_torch, batched_frame_weights_torch, batch_upsampled_stimulus_torch

    del cropped_stimulus_flat_torch, cropped_stimulus_torch

    return repeat_acc


def ct_uncoupled_glm_simulate_single_cell(
        uncoupled_glm_sim_parameters: SingleCellUncoupledGLMForSim,
        sim_spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
        h_w_slice: Tuple[Tuple[int, int], Tuple[int, int]],
        repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
        device: torch.device,
        max_batch_size: int = 8) -> List[List[np.ndarray]]:
    n_stimuli = repeats_dataloader.n_stimuli

    uncoupled_sim_glm = CTSimFBOnlyGLM(
        uncoupled_glm_sim_parameters.cropped_spatial_filter.reshape(-1),
        uncoupled_glm_sim_parameters.stimulus_timecourse,
        uncoupled_glm_sim_parameters.feedback_filter,
        uncoupled_glm_sim_parameters.bias,
        sim_spike_generation_callable,
        dtype=torch.float32
    ).to(device)

    simulated = []
    pbar = tqdm.tqdm(total=n_stimuli)
    for stim_ix in range(n_stimuli):
        simulated.append(ct_uncoupled_glm_simulate_single_cell_single_stimulus(
            repeats_dataloader,
            uncoupled_sim_glm,
            stim_ix,
            uncoupled_glm_sim_parameters.cell_id,
            h_w_slice,
            device,
            max_batch_size=max_batch_size
        ))

        pbar.update(1)

    pbar.close()

    del uncoupled_sim_glm

    return simulated
