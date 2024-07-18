import argparse
import pickle
from typing import Dict, List, Tuple, Callable, Optional, Union
import torch.multiprocessing as mp
from collections import namedtuple
from enum import Enum

import numpy as np
import torch

import tqdm

import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from lib.data_utils.interaction_graph import InteractionGraph
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
from lib.dataset_specific_hyperparams.jitter_glm_hyperparameters import \
    CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from new_style_fit_multimovie_ct_glm import new_style_wn_regularized_precompute_convs_and_fit_glm
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator
from optimization_encoder.trial_glm import FittedGLM, FittedGLMFamily
from optimization_encoder.ct_glm import fused_bernoulli_neg_ll_loss
from fit_multimovie_ct_glm import compute_full_filters
from lib.data_utils.dynamic_data_util import preload_bind_jittered_movie_patches_to_synchro, \
    preload_bind_wn_movie_patches

# FIXME eventually don't hardcode this
INCLUDED_BLOCKS_BY_NAME = {
    # these were for 2018-08-07-5
    '2018-08-07-5': {
        'data009': [1, 2, 3, 4, 6, 7, 8, 9],
        'data010': [1, 2, 3, 4, 6, 7, 8, 9],
    },

    # these are for 2019-11-07-0
    '2019-11-07-0': {
        'data005': [1, 2, 3, 4, 6, 7, 8, 9],
        'data006': [1, 2, 3, 4, 6, 7, 8, 9],
    },

    # these are for 2018-11-12-5
    '2018-11-12-5': {
        'data004': [1, 2, 3, 4, 6, 7, 8, 9],
        'data005': [1, 2, 3, 4, 6, 7, 8, 9],
    }
}


CTMovieData = namedtuple('CTMovieData',
                         ['cell_id', 'training_movie_blocks', 'training_wn_movie_blocks', 'spat_spline_basis_imshape',
                          'coupled_cells_subset', 'coupled_cell_ids_ordered'])

DONE_TOKEN = 'WEASEL'


class SolverMode(Enum):
    FULL = 1
    STUPID = 2
    DEBUG = 3


def get_ct_movie_from_disk(out_queue: mp.Queue,
                           loaded_synchronized_datasets,
                           brownian_blocks_to_include: Dict[str, List[int]],
                           loaded_wn_movie: ddu.LoadedWNMovies,
                           seconds_wn: float,
                           cell_type: str,
                           cells_ordered: OrderedMatchedCellsStruct,
                           pairwise_interactions: InteractionGraph,
                           cell_distances_by_type: Dict[str, float],
                           bounding_boxes_by_type,
                           basis_params: Tuple[Tuple[int, int], str],
                           crop_height_low: int,
                           crop_height_high: int,
                           crop_width_low: int,
                           crop_width_high: int,
                           nscenes_downsample_factor: int,
                           wn_rel_downsample: int,
                           wn_interval: int) -> None:
    relevant_cell_ids = cells_ordered.get_reference_cell_order(cell_type)
    ct_order = cells_ordered.get_cell_types()
    (n_spatbasis_h, n_spatbasis_w), spat_basis_type = basis_params

    for idx, cell_id in enumerate(relevant_cell_ids):

        print("!!!!!!!!!!!!!!! FETCHING DATA !!!!!!!!!!!!!!!!!!!!", flush=True)

        coupled_cells_subset = {}  # type: Dict[str, List[int]]
        all_coupled_cell_ids_ordered = []  # type: List[int]
        for coupled_cell_type in ct_order:
            max_coupled_distance_typed = cell_distances_by_type[coupled_cell_type]
            interaction_edges = pairwise_interactions.query_cell_interaction_edges(cell_id, coupled_cell_type)
            coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                x.additional_attributes['distance'] < max_coupled_distance_typed]
            coupled_cells_subset[coupled_cell_type] = coupled_cell_ids
            all_coupled_cell_ids_ordered.extend(coupled_cell_ids)

        all_coupled_cell_ids_ordered_np = np.array(all_coupled_cell_ids_ordered, np.int64)

        ### Get cropping objects ########################################
        cell_ix = relevant_cell_ids.index(cell_id)
        crop_bbox = bounding_boxes_by_type[cell_type][cell_ix]  # type: CroppedSTABoundingBox
        crop_bounds_h, crop_bounds_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor,
            return_bounds=True
        )

        sta_cropping_slice_h, sta_cropping_slice_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=wn_rel_downsample * nscenes_downsample_factor
        )

        hhh = crop_bounds_h[1] - crop_bounds_h[0]
        www = crop_bounds_w[1] - crop_bounds_w[0]

        spat_spline_basis = make_spatial_basis_from_hyperparameters(crop_bounds_h, crop_bounds_w,
                                                                    n_spatbasis_h, n_spatbasis_w,
                                                                    spat_basis_type)

        spat_spline_basis_imshape = spat_spline_basis.T.reshape(-1, hhh, www)

        ##########################################################################
        # Load the frames
        training_movie_blocks = preload_bind_jittered_movie_patches_to_synchro(
            loaded_synchronized_datasets,
            brownian_blocks_to_include,
            (crop_bounds_h, crop_bounds_w),
            ddu.PartitionType.TRAIN_PARTITION,
            verbose=False
        )

        training_wn_movie_blocks = preload_bind_wn_movie_patches(
            loaded_wn_movie,
            [(0, int(WN_FRAME_RATE * seconds_wn // wn_interval)), ],
            (sta_cropping_slice_h, sta_cropping_slice_w)
        )

        out_queue.put(CTMovieData(cell_id, training_movie_blocks, training_wn_movie_blocks,
                                  spat_spline_basis_imshape, coupled_cells_subset, all_coupled_cell_ids_ordered_np))

    out_queue.put(DONE_TOKEN)

    return


def fit_model_from_queue(input_data_queue: mp.Queue,
                         output_fit_model_queue: mp.Queue,
                         cells_ordered: OrderedMatchedCellsStruct,
                         timecourse_basis: np.ndarray,
                         feedback_basis: np.ndarray,
                         coupling_basis: np.ndarray,
                         cell_type: str,
                         samples_per_bin,
                         rescale_low_high: Tuple[float, float],  # we can't pickle lambda functions
                         loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         wn_weight: float,
                         l21_param: float,
                         l1_param: float,
                         n_outer_iter: int,
                         eps_outer: float,
                         device: torch.device,
                         initial_guess_timecourse: Union[torch.Tensor, np.ndarray],
                         trim_spikes_seq: int,
                         solver_mode: SolverMode,
                         jitter_spike_times: float = 0.0) -> None:

    image_rescale_low, image_rescale_high = rescale_low_high
    image_rescale_lambda_torch = du.make_torch_image_transform_lambda(image_rescale_low, image_rescale_high)

    while True:

        if not input_data_queue.empty():
            print(f'Data queue approx length {input_data_queue.qsize()}', flush=True)
            next_data_chunk = input_data_queue.get(block=True)  # type: Union[str, CTMovieData]
            if next_data_chunk == DONE_TOKEN:
                output_fit_model_queue.put(DONE_TOKEN)
                break

            if solver_mode == SolverMode.DEBUG:
                solver_schedule = (ProxSolverParameterGenerator(500, 250), ProxSolverParameterGenerator(500, 250))
            elif solver_mode == SolverMode.STUPID:
                solver_schedule = (ProxSolverParameterGenerator(25, 25), ProxSolverParameterGenerator(25, 25))
            else:
                solver_schedule = (ProxSolverParameterGenerator(1000, 250), ProxSolverParameterGenerator(1000, 250))

            loss, glm_fit_params = new_style_wn_regularized_precompute_convs_and_fit_glm(
                next_data_chunk.training_movie_blocks,
                next_data_chunk.training_wn_movie_blocks,
                wn_weight,
                next_data_chunk.spat_spline_basis_imshape,
                timecourse_basis,
                feedback_basis,
                coupling_basis,
                (cell_type, next_data_chunk.cell_id),
                next_data_chunk.coupled_cells_subset,
                cells_ordered,
                samples_per_bin,
                image_rescale_lambda_torch,
                loss_fn,
                l21_param,  # group sparsity
                l1_param,  # spatial sparsity
                n_outer_iter,
                eps_outer,
                solver_schedule,
                device,
                initial_guess_timecourse=initial_guess_timecourse[None, :],
                trim_spikes_seq=trim_spikes_seq,
                log_verbose_ascent=True,
                movie_spike_dtype=torch.float16,
                jitter_spike_times=jitter_spike_times
            )

            coupling_w, feedback_w, spat_w, timecourse_w, bias = glm_fit_params

            n_cells, hhh, www = next_data_chunk.spat_spline_basis_imshape.shape
            spat_spline_basis_flat = next_data_chunk.spat_spline_basis_imshape.reshape(n_cells, -1)

            spat_filters, time_filters, feedback_td, coupling_td = compute_full_filters(
                spat_w,
                spat_spline_basis_flat,
                timecourse_w,
                timecourse_basis,
                feedback_w,
                feedback_basis,
                coupling_w,
                coupling_basis,
            )

            spat_filt_as_image_shape = spat_filters.reshape(hhh, www)

            fitting_params_to_store = {
                'l1_spat_sparse': l1_param,
                'l21_group_sparse': l21_param,
                'wn_weight': wn_weight
            }

            model_summary_parameters = FittedGLM(next_data_chunk.cell_id, spat_filt_as_image_shape, bias,
                                                 timecourse_w, feedback_w,
                                                 (coupling_w, next_data_chunk.coupled_cell_ids_ordered),
                                                 fitting_params_to_store, loss, None)

            output_fit_model_queue.put(model_summary_parameters)

            del next_data_chunk


def main(args,
         device):
    config_settings = read_config_file(args.cfg_file)

    cell_type = args.cell_type
    if args.debug:
        solver_mode = SolverMode.DEBUG
    elif args.stupid:
        solver_mode = SolverMode.STUPID
    else:
        solver_mode = SolverMode.FULL

    ref_dataset_info = config_settings[dcp.ReferenceDatasetSection.OUTPUT_KEY] # type: dcp.DatasetInfo

    piece_name, ref_datarun_name = dcp.awsify_piece_name_and_datarun_lookup_key(
        ref_dataset_info.path,
        ref_dataset_info.name)

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = samples_per_bin // 20

    ################################################################
    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    # calculate timing for the nscenes dataset
    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    ###################################################################
    # Load the white noise reference dataset since we are joint fitting
    reference_dataset = vl.load_vision_data(ref_dataset_info.path,
                                            ref_dataset_info.name,
                                            include_sta=True,
                                            include_neurons=True,
                                            include_params=True)
    ttl_times_whitenoise = reference_dataset.get_ttl_times()

    wn_frame_generator = RandomNoiseFrameGenerator.construct_from_xml(ref_dataset_info.wn_xml)
    wn_interval = wn_frame_generator.refresh_interval
    wn_rel_downsample = wn_frame_generator.stixel_height // (2 * nscenes_downsample_factor)

    wn_synchro_block = WhiteNoiseSynchroSection(
        wn_frame_generator,
        ttl_times_whitenoise,
        WN_N_FRAMES_PER_TRIGGER // wn_interval,
        SAMPLE_RATE)

    loaded_wn_movie = ddu.LoadedWNMovies(ref_dataset_info.path,
                                         ref_dataset_info.name,
                                         reference_dataset,
                                         wn_synchro_block)

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    ###### Get model hyperparameters ###########################################
    # FIXME we need a new hyperparameter dict
    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = \
        CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE[ref_ds_lookup_key][bin_width_time_ms]()[cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    feedback_basis = model_fit_hyperparameters.feedback_basis
    coupling_basis = model_fit_hyperparameters.coupling_basis

    l1_spat_sparse = args.l1_sparse
    l21_group_sparse = args.l21_couple
    wn_model_weight = args.wn_weight
    n_iter_outer = args.outer_iter

    ###### Compute the STA timecourse, and use that as the initial guess for ###
    ###### the timecourse ######################################################
    # Load the timecourse initial guess
    with open(config_settings[dcp.OutputSection.INITIAL_GUESS_TIMECOURSE], 'rb') as pfile:
        timecourses_by_type = pickle.load(pfile)
        avg_timecourse = timecourses_by_type[cell_type]

    initial_timevector_guess = np.linalg.solve(timecourse_basis @ timecourse_basis.T,
                                               timecourse_basis @ avg_timecourse)

    ##########################################################################
    # Begin model fitting for each cell

    relevant_cell_ids = cells_ordered.get_reference_cell_order(cell_type)

    data_queue = mp.Queue(maxsize=3)
    output_model_queue = mp.Queue()

    load_data_P = mp.Process(
        target=get_ct_movie_from_disk,
        args=(
            data_queue,
            loaded_synchronized_datasets,
            INCLUDED_BLOCKS_BY_NAME[piece_name],
            loaded_wn_movie,
            args.seconds_wn,
            cell_type,
            cells_ordered,
            pairwise_interactions,
            cell_distances_by_type,
            bounding_boxes_by_type,
            model_fit_hyperparameters.spatial_basis,
            crop_height_low,
            crop_height_high,
            crop_width_low,
            crop_width_high,
            nscenes_downsample_factor,
            wn_rel_downsample,
            wn_interval
        )
    )

    fit_model_P = mp.Process(
        target=fit_model_from_queue,
        args=(
            data_queue,
            output_model_queue,
            cells_ordered,
            timecourse_basis,
            feedback_basis,
            coupling_basis,
            cell_type,
            samples_per_bin,
            (image_rescale_low, image_rescale_high),
            fused_bernoulli_neg_ll_loss,
            wn_model_weight,
            l21_group_sparse,
            l1_spat_sparse,
            n_iter_outer,
            1e-5,
            device,
            initial_timevector_guess,
            feedback_basis.shape[1] - 1,
            solver_mode
        ),
        kwargs={
            'jitter_spike_times': args.jitter
        }
    )

    load_data_P.start()
    fit_model_P.start()

    model_fit_pbar = tqdm.tqdm(total=len(relevant_cell_ids))
    fitted_models_dict = {}  # type: Dict[int, FittedGLM]

    while True:
        if not output_model_queue.empty():
            fitted_model = output_model_queue.get()
            if fitted_model == DONE_TOKEN:
                model_fit_pbar.close()
                break
            else:
                fitted_models_dict[fitted_model.cell_id] = fitted_model
                model_fit_pbar.update(1)

    fitted_model_family = FittedGLMFamily(
        fitted_models_dict,
        model_fit_hyperparameters.spatial_basis,
        timecourse_basis,
        feedback_basis,
        coupling_basis
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(fitted_model_family, pfile)

    load_data_P.join()
    fit_model_P.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit LNBRC (coupled) encoding models jointly to eye movements stimulus and white noise. Multiprocessing implementation (faster).')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=2,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-s', '--stupid', action='store_true', default=False)
    parser.add_argument('-l', '--l1_sparse', type=float, default=1e-6,
                        help='hyperparameter for L1 spatial filter sparsity')
    parser.add_argument('-g', '--l21_couple', type=float, default=1e-5,
                        help='hyperparameter for L21 group sparsity for coupling filters')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    parser.add_argument('-j', '--jitter', type=float, default=0.0,
                        help='Amount to jitter the spike times by, in samples; Default 0')
    parser.add_argument('-m', '--seconds_wn', type=int, default=300,
                        help='number of seconds of white noise data to use')
    args = parser.parse_args()

    device = torch.device('cuda')

    mp.set_start_method('spawn')

    main(args, device)
