from typing import List, Tuple, Dict, Callable, Union

import numpy as np
import torch
import tqdm
import visionloader as vl
from movie_upsampling import movie_sparse_upsample_transpose_cuda

import lib.data_utils.dynamic_data_util as ddu
from glm_precompute.ct_glm_precompute import multidata_precompute_spat_basis_mul_only, multidata_bin_center_spikes, \
    multidata_bin_coupling_spikes
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
from lib.dataset_specific_ttl_corrections.nsbrownian_ttl_structure_corrections import SynchronizedNSBrownianSection
from sim_retina.single_cell_sim import CTSimGLM
from optimization_encoder.multimovie_ct_glm import MM_SingleCellEncodingLoss


def preload_cropped_frames(block_sequence: List[SynchronizedNSBrownianSection],
                           bounding_box: CroppedSTABoundingBox,
                           crop_hlow: int = 0,
                           crop_hhigh: int = 0,
                           crop_wlow: int = 0,
                           crop_whigh: int = 0,
                           spat_downsample_factor: int = 1) -> List[np.ndarray]:
    h_low_high, w_low_high = bounding_box.make_cropping_sliceobj(crop_hlow=crop_hlow,
                                                                 crop_wlow=crop_wlow,
                                                                 crop_whigh=crop_whigh,
                                                                 crop_hhigh=crop_hhigh,
                                                                 downsample_factor=spat_downsample_factor,
                                                                 return_bounds=True)
    ret_list = []
    pbar = tqdm.tqdm(total=len(block_sequence))
    for i, block in enumerate(block_sequence):
        cropped_section = block.fetch_frames_bw(h_low_high=h_low_high, w_low_high=w_low_high)
        ret_list.append(cropped_section)
        pbar.update(1)
    pbar.close()

    return ret_list


def preload_test_frame_section(block: SynchronizedNSBrownianSection,
                               low_ix: int,
                               high_ix: int,
                               bounding_box: CroppedSTABoundingBox,
                               crop_hlow: int = 0,
                               crop_hhigh: int = 0,
                               crop_wlow: int = 0,
                               crop_whigh: int = 0,
                               spat_downsample_factor: int = 1) -> np.ndarray:
    h_low_high, w_low_high = bounding_box.make_cropping_sliceobj(crop_hlow=crop_hlow,
                                                                 crop_wlow=crop_wlow,
                                                                 crop_whigh=crop_whigh,
                                                                 crop_hhigh=crop_hhigh,
                                                                 downsample_factor=spat_downsample_factor,
                                                                 return_bounds=True)

    movie_section = block.fetch_frames_bw(start_frame_ix=low_ix,
                                          stop_frame_ix=high_ix,
                                          h_low_high=h_low_high,
                                          w_low_high=w_low_high)
    return movie_section


def torch_time_upsample_test_frame_section(movie_section: np.ndarray,
                                           sel_idx_overlap_w: Tuple[np.ndarray, np.ndarray],
                                           image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                           device: torch.device) -> torch.Tensor:
    sel_a, weight_a = sel_idx_overlap_w

    movie_section_torch = image_transform_lambda(torch.tensor(movie_section, dtype=torch.float32, device=device))
    sel_a_torch = torch.tensor(sel_a, dtype=torch.long, device=device)
    weight_a_torch = torch.tensor(weight_a, dtype=torch.float32, device=device)

    # shape (height, width, n_bins)
    movie_upsampled_torch = movie_sparse_upsample_transpose_cuda(movie_section_torch,
                                                                 sel_a_torch,
                                                                 weight_a_torch)

    # shape (n_pixels, n_bins)
    movie_upsampled_torch = movie_upsampled_torch.reshape(-1, movie_upsampled_torch.shape[2])

    del movie_section, sel_a_torch, weight_a_torch

    return movie_upsampled_torch


class JitteredMovieTrainingData:

    def __init__(self,
                 movie_dataset: ddu.LoadedBrownianMovies,
                 include_training_blocks: List[int]):
        self.movie_dataset = movie_dataset
        self.include_training_blocks = include_training_blocks

    def get_included_blocks(self) -> List[SynchronizedNSBrownianSection]:
        blocks = []
        for i in self.include_training_blocks:
            blocks.extend(self.movie_dataset.train_blocks[i])
        return blocks


class JitteredMovieTestData:
    def __init__(self,
                 movie_dataset: ddu.LoadedBrownianMovies,
                 include_blocks: List[int]):
        self.movie_dataset = movie_dataset
        self.include_blocks = include_blocks

    def get_included_blocks(self) -> List[SynchronizedNSBrownianSection]:
        return [self.movie_dataset.test_blocks[i] for i in self.include_blocks]


class JitteredMovieHeldoutData:
    def __init__(self,
                 movie_dataset: ddu.LoadedBrownianMovies,
                 include_blocks: List[int]):
        self.movie_dataset = movie_dataset
        self.include_blocks = include_blocks

    def get_included_blocks(self) -> List[SynchronizedNSBrownianSection]:
        return [self.movie_dataset.heldout_blocks[i] for i in self.include_blocks]


JitteredMovieTestHeldoutData = Union[JitteredMovieTestData, JitteredMovieHeldoutData]


def precompute_spat_basis_eval_glm_model(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        stim_spat_basis: np.ndarray,
        stim_spat_w: np.ndarray,
        timecourse_filter: np.ndarray,
        feedback_filter: np.ndarray,
        coupling_filter: np.ndarray,
        bias: np.ndarray,
        center_cell_wn: Tuple[str, int],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        bin_width_samples: int,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device) -> float:
    '''
    Computes either the test or heldout loss for GLM fit

    Uses the sparse representation of the spatial stimulus as an intermediate, to conserve
        GPU space

    :param movie_test_heldout_datasets:
    :param moive_patches_by_dataset:
    :param stim_spat_basis: shape (n_basis_stim_spat, n_pixels)
    :param stim_spat_w: (1, n_basis_stim_spat)
    :param timecourse_filter: shape (n_bins_filter, )
    :param feedback_filter: shape (n_bins_filter, )
    :param coupling_filter: shape (n_coupled_cells, n_bins_filter)
    :param cells_ordered:
    :param bin_width_samples:
    :param image_transform_lambda:
    :param loss_callable:
    :param device:
    :return:
    '''

    # Step 0: unpack everything that was a Dict into a List
    # Also bin spikes, and compute frame overlaps in the same loop
    # to make sure that the ordering of the lists doesn't get messed up
    timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                    bin_width_samples)

    frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                          timebins_all)

    center_spikes_all_torch = multidata_bin_center_spikes(
        jittered_movie_blocks, timebins_all,
        center_cell_wn, cells_ordered,
        device,
    )

    coupling_spikes_all_torch = multidata_bin_coupling_spikes(
        jittered_movie_blocks, timebins_all,
        coupled_cells, cells_ordered,
        device,
    )

    # pre-compute the application of the spatial basis
    # with the fixed timecourse on the upsampled movie
    # and store the result on GPU
    precomputed_spatial_convolutions = multidata_precompute_spat_basis_mul_only(
        jittered_movie_blocks,
        frame_sel_and_overlaps_all,
        stim_spat_basis,
        image_transform_lambda,
        device
    )

    # now construct the evaluation model
    evaluation_model = MM_SingleCellEncodingLoss(
        stim_spat_w,
        timecourse_filter,
        feedback_filter,
        coupling_filter,
        bias,
        loss_callable,
        dtype=torch.float32
    ).to(device)

    total_loss_torch = evaluation_model(precomputed_spatial_convolutions,
                                        center_spikes_all_torch,
                                        coupling_spikes_all_torch).item()

    # clean up GPU explicitly
    del precomputed_spatial_convolutions, center_spikes_all_torch
    del coupling_spikes_all_torch

    return total_loss_torch


def compute_lnp_filters(spat_basis_repr: np.ndarray,
                        spat_basis: np.ndarray,
                        timecourse_basis_repr: np.ndarray,
                        timecourse_basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Converts LNP filters in their low-rank basis representations to their full representations

    :param spat_basis_repr: (1, n_basis_spat)
    :param spat_basis: (n_basis_spat, n_pixels)
    :param timecourse_basis_repr: (1, n_basis_timecourse)
    :param timecourse_basis: (n_basis_timecourse, n_bins_filt)
    '''

    # shape (1, n_basis_spat) @ (n_basis_spat, n_pixels)
    # -> (1, n_pixels) -> (n_pixels, )
    spat_filt_full = (spat_basis_repr @ spat_basis).squeeze(0)

    # shape (1, n_basis_timecourse) @ (n_basis_timecourse, n_bins_filt)
    # -> (1, n_bins_filt) -> (n_bins_filt, )
    timecourse_filt_full = timecourse_basis_repr @ timecourse_basis

    return spat_filt_full, timecourse_filt_full


def compute_spat_timecourse_feedback_filters(
        spat_basis_repr: np.ndarray,
        spat_basis: np.ndarray,
        timecourse_basis_repr: np.ndarray,
        timecourse_basis: np.ndarray,
        feedback_filt_w: np.ndarray,
        feedback_basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Converts GLM filters in their low-rank basis representations to their full representations

    Does not include the coupling filters

    :param spat_basis_repr: (1, n_basis_spat)
    :param spat_basis: (n_basis_spat, n_pixels)
    :param timecourse_basis_repr: (1, n_basis_timecourse)
    :param timecourse_basis: (n_basis_timecourse, n_bins_filt)
    :param feedback_filt_w: (1, n_basis_feedback)
    :param feedback_basis: (n_basis_feedback, n_bins_filt)
    :return: In order, spatial, timecourse, feedback, and coupling filters
        Spatial filter: shape (n_pixels, )
        Timecourse filter: shape (n_bins_filt, )
        Feedback filter: shape (n_bins_filt, )
    '''

    # shape (1, n_basis_spat) @ (n_basis_spat, n_pixels)
    # -> (1, n_pixels) -> (n_pixels, )
    spat_filt_full = (spat_basis_repr @ spat_basis).squeeze(0)

    # shape (1, n_basis_timecourse) @ (n_basis_timecourse, n_bins_filt)
    # -> (1, n_bins_filt) -> (n_bins_filt, )
    timecourse_filt_full = timecourse_basis_repr @ timecourse_basis

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filt)
    # -> (1, n_bins_filt) -> (n_bins_filt, )
    feedback_filt_full = (feedback_filt_w @ feedback_basis).squeeze(0)

    return spat_filt_full, timecourse_filt_full, feedback_filt_full


def compute_full_filters(spat_basis_repr: np.ndarray,
                         spat_basis: np.ndarray,
                         timecourse_basis_repr: np.ndarray,
                         timecourse_basis: np.ndarray,
                         feedback_filt_w: np.ndarray,
                         feedback_basis: np.ndarray,
                         coupling_filt_w: np.ndarray,
                         coupling_basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Converts GLM filters in their low-rank basis representations to their full representations

    :param spat_basis_repr: (1, n_basis_spat)
    :param spat_basis: (n_basis_spat, n_pixels)
    :param timecourse_basis_repr: (1, n_basis_timecourse)
    :param timecourse_basis: (n_basis_timecourse, n_bins_filt)
    :param feedback_filt_w: (1, n_basis_feedback)
    :param feedback_basis: (n_basis_feedback, n_bins_filt)
    :param coupling_filt_w: (n_coupled_cells, n_basis_coupling)
    :param coupling_basis: (n_basis_coupling, n_bins_filt)
    :return: In order, spatial, timecourse, feedback, and coupling filters
        Spatial filter: shape (n_pixels, )
        Timecourse filter: shape (n_bins_filt, )
        Feedback filter: shape (n_bins_filt, )
        Coupling filter: shape (n_coupled_cells, n_bins_filt)
    '''

    # shape (1, n_basis_spat) @ (n_basis_spat, n_pixels)
    # -> (1, n_pixels) -> (n_pixels, )
    spat_filt_full = (spat_basis_repr @ spat_basis).squeeze(0)

    # shape (1, n_basis_timecourse) @ (n_basis_timecourse, n_bins_filt)
    # -> (1, n_bins_filt) -> (n_bins_filt, )
    timecourse_filt_full = timecourse_basis_repr @ timecourse_basis

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filt)
    # -> (1, n_bins_filt) -> (n_bins_filt, )
    feedback_filt_full = (feedback_filt_w @ feedback_basis).squeeze(0)

    # shape (n_coupled_cells, n_basis_coupling) @ (n_basis_coupling, n_bins_filt)
    # -> (n_coupled_cells, n_bins_filt)
    coupling_filt_full = coupling_filt_w @ coupling_basis

    return spat_filt_full, timecourse_filt_full, feedback_filt_full, coupling_filt_full


def sim_ct_glm_repeats(spat_filter: np.ndarray,
                       timecourse_filter: np.ndarray,
                       feedback_filt: np.ndarray,
                       coupling_filt: np.ndarray,
                       bias: np.ndarray,
                       spike_gen_fn: Callable[[torch.Tensor], torch.Tensor],
                       center_cell_iden: Tuple[str, int],
                       coupled_cells: Dict[str, List[int]],
                       cell_matching: OrderedMatchedCellsStruct,
                       repeats_blocks: List[Tuple[vl.VisionCellDataTable, str, SynchronizedNSBrownianSection]],
                       bounding_box: CroppedSTABoundingBox,
                       image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                       stimulus_window: Tuple[int, int],
                       n_sim_repeats_per_repeat: int,
                       device: torch.device,
                       crop_hlow: int = 0,
                       crop_hhigh: int = 0,
                       crop_wlow: int = 0,
                       crop_whigh: int = 0,
                       nscenes_downsample_factor: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Assumes that the frame rate for each of the repeats is consistent enough
        that we don't have to adjust the timing between each repeat to make them
        comparable to each other. This assumption is probably good when we simulate
        responses to a small number of stimuli, since the accumulated drift in frame
        timing will be small....

    :param repeats_blocks:
    :param trial_start:
    :param trial_end:
    :param n_sim_repeats_per_repeat:
    :return:
    '''

    ct_simulation_model = CTSimGLM(
        spat_filter,
        timecourse_filter,
        feedback_filt,
        coupling_filt,
        bias,
        spike_gen_fn
    ).to(device)

    ret_list = []
    for nscenes_dataset, nscenes_name, repeat_block in repeats_blocks:
        # bin spikes, compute overlaps, etc.
        repeat_frame_ix, repeat_overlaps, repeat_spikes, repeat_couple_spikes = ddu.extract_singledata_movie_and_spikes(
            nscenes_dataset,
            nscenes_name,
            repeat_block,
            20,
            stimulus_window,
            center_cell_iden,
            coupled_cells,
            cell_matching
        )

        repeat_frames_no_upsample = preload_test_frame_section(
            repeat_block,
            repeat_frame_ix[0],
            repeat_frame_ix[1],
            bounding_box,
            crop_hlow=crop_hlow,
            crop_hhigh=crop_hhigh,
            crop_wlow=crop_wlow,
            crop_whigh=crop_whigh,
            spat_downsample_factor=nscenes_downsample_factor
        )

        # upsample the frames, precompute basis convolutions
        repeat_frames_upsampled_torch = torch_time_upsample_test_frame_section(
            repeat_frames_no_upsample, repeat_overlaps, image_transform_lambda, device)

        coupled_cell_spikes_torch_for_sim = torch.tensor(repeat_couple_spikes, dtype=torch.float32, device=device)
        initial_spike_section = torch.zeros((timecourse_filter.shape[0],), dtype=torch.float32, device=device)

        sim_spikes_torch = ct_simulation_model.sim_ar_sim_repeats(
            repeat_frames_upsampled_torch,
            initial_spike_section,
            coupled_cell_spikes_torch_for_sim,
            n_repeats=n_sim_repeats_per_repeat
        )

        # move stuff back to numpy
        sim_spikes_np = sim_spikes_torch.detach().cpu().numpy()
        ret_list.append((repeat_spikes, sim_spikes_np))

        # clean up single-use stuff on GPU
        del repeat_frames_upsampled_torch, coupled_cell_spikes_torch_for_sim, initial_spike_section
        del sim_spikes_torch

    del ct_simulation_model

    return ret_list
