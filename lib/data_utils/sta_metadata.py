import visionloader as vl

import numpy as np

from typing import List, Dict, Tuple, Sequence, Union, Optional, Any
from collections import namedtuple
import itertools

import statsmodels.robust as robust
from scipy import interpolate
from scipy import ndimage

import shapely.affinity as affinity
from shapely.geometry import MultiPoint, Point

from skimage.measure import label

RGB_CONVERSION = np.array([[0.2989, 0.5870, 0.1140], ])


def compute_sig_stixel_mask(blurred_stas_by_type: Dict[str, List[np.ndarray]],
                            crop_hlow: int = 0,
                            crop_hhigh: int = 0,
                            crop_wlow: int = 0,
                            crop_whigh: int = 0,
                            downsample_factor: int = 1,
                            min_coverage: int = 2,
                            threshold: float = 1e-2,
                            use_cc_alg: bool = False) \
        -> np.ndarray:
    '''
    Assumes the blurred sig-stixel mask already has had the downsample applied, but
        no crop yet

    :param blurred_stas_by_type:
    :param crop_hlow:
    :param crop_hhigh:
    :param crop_wlow:
    :param crop_whigh:
    :param downsample_factor:
    :param threshold:
    :return:
    '''

    crop_hlow_div, crop_hhigh_div = crop_hlow // downsample_factor, crop_hhigh // downsample_factor
    crop_wlow_div, crop_whigh_div = crop_wlow // downsample_factor, crop_whigh // downsample_factor

    # need to compute the sig stixel mask; break this out into a function
    sig_stixel_pile = {}
    for cell_type, sig_stixel_list in blurred_stas_by_type.items():
        total = np.sum(np.abs(np.array(sig_stixel_list)), axis=0)
        sig_stixel_pile[cell_type] = total

    exceeds_threshold = np.stack([
        sig_stixel_pile[cell_type] > threshold for cell_type in sig_stixel_pile.keys()
    ], axis=0).astype(np.int64)

    union_mask = np.sum(exceeds_threshold, axis=0) > min_coverage

    orig_height, orig_width = union_mask.shape
    cropped_union_mask = union_mask[crop_hlow_div:orig_height - crop_hhigh_div,
                         crop_wlow_div:orig_width - crop_whigh_div]

    if use_cc_alg:
        return getLargestCC(cropped_union_mask)
    return cropped_union_mask


def getLargestCC(segmentation):
    ''' From stackoverflow '''
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


def compute_convex_hull_of_mask(original_mask_boolean: np.ndarray,
                                shrinkage_factor: float = 1.0) -> np.ndarray:
    '''

    :param original_mask:
    :param shrinkage_factor:
    :return:
    '''

    xx, yy = np.meshgrid(np.r_[0:original_mask_boolean.shape[1]],
                         np.r_[0:original_mask_boolean.shape[0]])

    xx_flat, yy_flat = xx.reshape(-1), yy.reshape(-1)

    within_xx, within_yy = xx[original_mask_boolean], yy[original_mask_boolean]
    point_tuples = [(x, y) for x, y in zip(within_xx, within_yy)]

    mp_boundary = MultiPoint(point_tuples)
    mp_hull = mp_boundary.convex_hull

    shrunk_hull = affinity.scale(mp_hull, xfact=shrinkage_factor,
                                 yfact=shrinkage_factor, origin='centroid')

    include_point = [shrunk_hull.contains(Point(x, y)) for x, y in zip(xx_flat, yy_flat)]
    new_mask = np.array(include_point).reshape(xx.shape)

    return new_mask


def greg_field_simpler_significant_stixels(sta_container: vl.STAContainer,
                                           std_threshold: float):
    # sum rgb together to get an intensity
    # use median and mad to get robust mean and std estimation
    # significant stixels are std_threshold * std away from the mean

    combined = sta_container.red + sta_container.green + sta_container.blue

    median = np.median(combined)
    std_robust = robust.scale.mad(combined.flatten())

    normalized = (combined - median) / std_robust

    # smash in time by taking the largest/smallest values
    significant_high = (np.max(normalized, axis=2) > std_threshold)
    significant_low = (np.min(normalized, axis=2) < (-std_threshold))

    significant_stixels = np.logical_or(significant_high, significant_low)

    return significant_stixels


def greg_field_simpler_significant_stixels_matrix(sta_matrix: np.ndarray,
                                                  std_threshold: float) -> np.ndarray:
    '''

    :param sta_matrix: STA matrix in np.ndarary form,
        shape (height, width, n_timepoints, n_channels = 3)
    :param std_threshold:
    :return:
    '''

    combined = np.sum(sta_matrix, axis=3)
    median = np.median(combined)
    std_robust = robust.scale.mad(combined.flatten())

    normalized = (combined - median) / std_robust

    # smash in time by taking the largest/smallest values
    significant_high = (np.max(normalized, axis=2) > std_threshold)
    significant_low = (np.min(normalized, axis=2) < (-std_threshold))

    significant_stixels = np.logical_or(significant_high, significant_low)

    return significant_stixels


SigTimecourseContainer = namedtuple('SigTimecourseContainer', ['red', 'green', 'blue'])


def find_spatial_sta_fit_regression_green_only(sta_container: vl.STAContainer,
                                               timecourse_container: SigTimecourseContainer) -> np.ndarray:
    # first normalize the absolute maximum of the green timecourse to 1 (so either 1 or -1, depending on ON or OFF)

    biggest_abs_val = np.max(np.abs(timecourse_container.green))
    normalized_g_timecourse = timecourse_container.green / biggest_abs_val
    # normalized_g_timecourse = timecourse_container.green

    # now do regression for every stixel in the green STA
    green_sta_matrix = sta_container.green

    self_magnitude = normalized_g_timecourse.dot(normalized_g_timecourse)

    cross_proj = np.inner(green_sta_matrix, normalized_g_timecourse)

    contour_matrix = cross_proj / self_magnitude

    return contour_matrix


def greg_field_calculate_weighted_centers_of_mass(spatial_weights_positive: np.ndarray,
                                                  spatial_mask: np.ndarray):
    include_in_weighting = spatial_mask

    nonzero_indices = np.nonzero(include_in_weighting)
    inc_row, inc_col = nonzero_indices

    weights = spatial_weights_positive[nonzero_indices]

    centroid_y = inc_row.dot(weights) / np.sum(weights)
    centroid_x = inc_col.dot(weights) / np.sum(weights)

    return np.array([centroid_x, centroid_y])


def calculate_center_from_sta(vision_dataset: vl.VisionCellDataTable,
                              cell_types_by_cell_id: Dict[str, List[int]],
                              sig_stixel_threshold: float = 4.0) -> Dict[str, Dict[int, np.ndarray]]:
    '''
    Calculates the centers of RFs from STA matrices

    Uses the simple significant stixels code

    :param vision_dataset:
    :param cell_types_by_cell_id:
    :return:
    '''
    ret_dict = {}  # type: Dict[str, Dict[int, np.ndarray]]
    for cell_type, cell_id_list in cell_types_by_cell_id.items():
        typed_centers_dict = {}  # type: Dict[int, np.ndarray]

        for cell_id in cell_id_list:
            sta_container = vision_dataset.get_sta_for_cell(cell_id)
            sig_stixels = greg_field_simpler_significant_stixels(sta_container, sig_stixel_threshold)

            if np.any(sig_stixels):
                sig_stixels_mask = np.nonzero(sig_stixels)

                sig_stixel_r_timecourses = sta_container.red[sig_stixels_mask[0], sig_stixels_mask[1], :]

                r_mean = np.mean(sig_stixel_r_timecourses.T, axis=1)

                sig_stixel_g_timecourses = sta_container.green[sig_stixels_mask[0], sig_stixels_mask[1], :]
                g_mean = np.mean(sig_stixel_g_timecourses.T, axis=1)

                sig_stixel_b_timecourses = sta_container.blue[sig_stixels_mask[0], sig_stixels_mask[1], :]
                b_mean = np.mean(sig_stixel_b_timecourses.T, axis=1)

                sig_timecourse_container = SigTimecourseContainer(r_mean, g_mean, b_mean)

                z = find_spatial_sta_fit_regression_sum_channels(sta_container,
                                                                 sig_timecourse_container)

                center_of_mass = greg_field_calculate_weighted_centers_of_mass(z, sig_stixels)
                typed_centers_dict[cell_id] = center_of_mass.squeeze()

        ret_dict[cell_type] = typed_centers_dict

    return ret_dict


def calculate_average_timecourse_highres(highres_sta_dict: Dict[int, np.ndarray],
                                         cell_id_list: List[int],
                                         sig_stixels_cutoff: float = 5.0,
                                         return_rgb=False) -> np.ndarray:
    '''
    Computes average timecourses for the high-resolution STA

    :param highres_sta_dict: Dict[int, np.ndarray], key is integer cell id,
        value is high-res STA, with shape (n_timepoints, height, width, 3)
    :param cell_id_list: List of cell ids that we want to include in the average,
        each entry in this list should correspond to a key in high_res_sta_dict
    :param sig_stixels_cutoff: cutoff for significant stixels, should be higher
        than for the standard STA because higher prob. of erroneous significance
        with more timepoints
    :param return_rgb:
    :return: np.ndarray, shape either (n_timepoints, ) or (3, n_timepoints) depending
        on whether or not we return RGB or BW
    '''
    # get the first STA so we can figure out what the dimensions are
    first_cell_id = cell_id_list[0]
    first_sta_array = highres_sta_dict[first_cell_id]
    n_timepoints = first_sta_array.shape[0]

    accumulation_multichannel = np.zeros((3, n_timepoints), dtype=np.float64)
    mean_denom = 0

    for cell_id in cell_id_list:

        sta_array = highres_sta_dict[cell_id]
        sig_stixels = greg_field_simpler_significant_stixels_matrix(sta_array.transpose(1, 2, 0, 3),
                                                                    sig_stixels_cutoff)

        if np.any(sig_stixels):
            h_sel, w_sel = np.where(sig_stixels)
            sig_stixel_timecourses = sta_array[:, h_sel, w_sel, :]

            accumulation_multichannel += np.sum(sig_stixel_timecourses,
                                                axis=1).T
            mean_denom += sig_stixel_timecourses.shape[1]

    mean_rgb = accumulation_multichannel / mean_denom

    if return_rgb:
        return mean_rgb
    else:
        return (RGB_CONVERSION @ mean_rgb).squeeze(0)


def calculate_average_timecourse(dataset: vl.VisionCellDataTable,
                                 cell_id_list: List[int],
                                 sig_stixels_cutoff: float = 4.0) -> np.ndarray:
    '''
    Calculate the average timecourse, weighted by total number of stixels
        NOTE: we are not averaging over cells, but over individual
        significant stixels

    :param dataset: vision dataset
    :param cell_id_list: List of cell id
    :return: np.ndarray of np.float64, with shape (3, n_timepoints)
    '''

    # get the first STA so we can figure out what the dimensions are
    first_cell_id = cell_id_list[0]
    sta_container_first = dataset.get_sta_for_cell(first_cell_id)
    n_timepoints_sta = sta_container_first.red.shape[2]

    # sum timecourses here, 3 rows because of three colors, in order RGB
    summed_timecourses_for_average = np.zeros((3, n_timepoints_sta), dtype=np.float64)
    n_cells_with_timecourse_estimate = 0

    # get the STAs for each cell
    # find the significant stixels
    # then use the significant stixels to calculate a time course for each cell
    for cell_id in cell_id_list:

        sta_container = dataset.get_sta_for_cell(cell_id)
        x = greg_field_simpler_significant_stixels(sta_container, sig_stixels_cutoff)

        if np.any(x):
            sig_stixels = np.nonzero(x)

            sig_stixel_r_timecourses = sta_container.red[sig_stixels[0], sig_stixels[1], :]
            r_sum = np.sum(sig_stixel_r_timecourses.T, axis=1)
            summed_timecourses_for_average[0, :] += r_sum

            sig_stixel_g_timecourses = sta_container.green[sig_stixels[0], sig_stixels[1], :]
            g_sum = np.sum(sig_stixel_g_timecourses.T, axis=1)
            summed_timecourses_for_average[1, :] += g_sum

            sig_stixel_b_timecourses = sta_container.blue[sig_stixels[0], sig_stixels[1], :]
            b_sum = np.sum(sig_stixel_b_timecourses.T, axis=1)
            summed_timecourses_for_average[2, :] += b_sum

            n_cells_with_timecourse_estimate += sig_stixel_r_timecourses.shape[1]  # number of
            # significant stixels that we added

    average_timecourse_rgb = summed_timecourses_for_average / n_cells_with_timecourse_estimate  # type: np.ndarray
    return average_timecourse_rgb


def find_spatial_sta_fit_regression_sum_channels(sta_container: vl.STAContainer,
                                                 timecourse_container: SigTimecourseContainer) -> np.ndarray:
    rgb_timecourse = np.array([timecourse_container.red, timecourse_container.green, timecourse_container.blue])
    bw_timecourse = RGB_CONVERSION @ rgb_timecourse  # shape (1, n_timepoints)

    biggest_abs_val = np.max(np.abs(bw_timecourse))
    normalized_summed_timecourse = bw_timecourse / biggest_abs_val  # shape (1, n_timepoints)
    # normalized_summed_timecourse = summed_timecourse

    # now do regression for every stixel in the sum
    summed_sta_matrix = RGB_CONVERSION[0, 0] * sta_container.red + RGB_CONVERSION[0, 1] * sta_container.green + \
                        RGB_CONVERSION[0, 2] * sta_container.blue

    # also normalize the summed sta matrix by its max pixel intensity (suggested by EJ)
    summed_sta_matrix = summed_sta_matrix / np.max(np.abs(summed_sta_matrix))

    self_magnitude = normalized_summed_timecourse @ normalized_summed_timecourse.T

    cross_proj = np.inner(summed_sta_matrix, normalized_summed_timecourse)

    contour_matrix = cross_proj / self_magnitude

    return contour_matrix


def fit_interpolate_sta_image(sta_container: vl.STAContainer,
                              rgb_timecourse: np.ndarray,
                              upsample_factor: int = 2) -> np.ndarray:
    contour_matrix = find_spatial_sta_fit_regression_sum_channels(sta_container, rgb_timecourse)
    orig_rows, orig_cols, _ = contour_matrix.shape

    target_rows, target_cols = upsample_factor * orig_rows, upsample_factor * orig_cols

    orig_grid_rows, orig_grid_cols = np.r_[0:target_rows:upsample_factor], np.r_[0:target_cols:upsample_factor]
    resampled_grid_rows, resampled_grid_cols = np.r_[0:target_rows], np.r_[0:target_cols]

    interpolated_function = interpolate.interp2d(orig_grid_cols, orig_grid_rows, contour_matrix)
    interpolated_image = interpolated_function(resampled_grid_cols, resampled_grid_rows)
    return interpolated_image


def determine_minimum_overlap_cutoff(sta_image_list: List[np.ndarray],
                                     threshold_set: Sequence[float]) -> float:
    image_size = sta_image_list[0].shape

    opt_single_coverage, opt_thresh = -np.inf, -np.inf
    for thresh in threshold_set:
        coverage = np.zeros(image_size)
        for image in sta_image_list:
            exceeds_threshold = (image > thresh).astype(np.int32)
            coverage += exceeds_threshold

        single_coverage = np.sum(coverage == 1)
        if single_coverage > opt_single_coverage:
            opt_single_coverage, opt_thresh = single_coverage, thresh

    return opt_thresh


def calculate_optimal_sta_masks_with_sta_intensity(
        wn_dataset: vl.VisionCellDataTable,
        cell_id_list: List[int],
        sta_upsample_factor: int = 1,
        sig_stixel_thresh: float = 4.0) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    '''
    Method:

    (1) Find significant stixels and use significant stixels pooled over all cells in cell_id_list
        to determine average timecourse
    (2) Regress out the timecourse from the original STA to get a surface map of similarity of each stixel's timecourse
        to the average timecourse
    (3) Interpolation on that surface map if necessary to upsample to match a target resolution
    (4) Determine the optimum cutoff threshold to maximize the amount of area covered by a single cell
    (5) Use that cutoff threshold to produce masks in upsampled space for each of the cells in cell_id_list
    :param wn_dataset:
    :param cell_id_list:
    :param sta_upsample_factor:
    :return:
    '''

    average_timecourse = calculate_average_timecourse(wn_dataset,
                                                      cell_id_list,
                                                      sig_stixels_cutoff=sig_stixel_thresh)

    interp_images_by_cell_id = {}  # type: Dict[int, np.ndarray]

    for cell_id in cell_id_list:
        raw_sta_container = wn_dataset.get_sta_for_cell(cell_id)
        upsampled_image = fit_interpolate_sta_image(raw_sta_container,
                                                    average_timecourse,
                                                    upsample_factor=sta_upsample_factor)
        interp_images_by_cell_id[cell_id] = upsampled_image

    pooled_images = list(interp_images_by_cell_id.values())

    optimum_cutoff = determine_minimum_overlap_cutoff(pooled_images,
                                                      np.r_[0:1.0:0.1])

    masks_by_cell_id = {cell_id: (interp_image > optimum_cutoff) for cell_id, interp_image in
                        interp_images_by_cell_id.items()}

    return masks_by_cell_id, interp_images_by_cell_id


class DownsampledCroppedFullBox:

    def __init__(self, coord_def_stixel_size: int, dims: Tuple[int, int], full_res_stixel_size: int = 2):
        self.coord_def_stixel_size = coord_def_stixel_size
        self.full_res_stixel_size = full_res_stixel_size
        scaleup = coord_def_stixel_size // full_res_stixel_size
        self.fullres_height, self.fullres_width = dims[0] * scaleup, dims[1] * scaleup

    def make_cropping_sliceobj(self, crop_hlow: int = 0, crop_wlow: int = 0, crop_hhigh: int = 0,
                               crop_whigh: int = 0, downsample_factor: int = 1):
        cropping_hlow = crop_hlow
        cropping_hhigh = self.fullres_height - crop_hhigh

        cropping_wlow = crop_wlow
        cropping_whigh = self.fullres_width - crop_whigh

        if downsample_factor == 1:
            return np.s_[cropping_hlow:cropping_hhigh, cropping_wlow:cropping_whigh]
        return np.s_[(cropping_hlow // downsample_factor):(cropping_hhigh // downsample_factor),
               (cropping_wlow // downsample_factor):(cropping_whigh // downsample_factor)]


class CroppedSTABoundingBox:
    '''
    Generate appropriate slice objects for the stimulus
        given a bounding box around the significant stixels in the STA

    Because the bounding box is calculated from the STA, it has the
        coarsest possible resolution that we will use in any computation

    Therefore, we will be able to use any reasonable downsample possible,
        up to and including the resolution of the STA

    IMPORTANT: there are a few different types of bounding box slice objects
        that we might need to create. In particular:
        (1) A slice object, that, given an image that has already been cropped and
            downsampled, grabs the corresponding patch from the cropped and downsampled
            image

        (2) A slice object, that, given an image that has been downsampled but not cropped
            grabs the same patch as the above (taking into account the crop, even though
            the source image itself hasn't been cropped)

        (3) A placement slice object,
    '''

    def __init__(self, sta_height_coords: Tuple[int, int], sta_width_coords: Tuple[int, int],
                 coord_def_stixel_size: int, dims: Tuple[int, int], full_res_stixel_size: int = 2):
        '''
        Constructs a bounding box using coordinates estimated from the STA

        The coordinates used in this constructor are defined relative to the STA. The STA is the lowest
            resolution image that we can use, with its resolution defined by coord_def_stixel_size, and
            full_res_stixel_size defines the highest resolution image we can use. Everything must be a power
            of two so that integer division downsampling works cleanly

        The coordinates provided to the constructor are assumed to be uncropped (i.e. directly from an STA
            calculated using the full stimulus). Possible cropping can be taken care of when
            generating the slice objects from an instance of CroppedSTABoundingBox

        :param sta_height_coords: low and high height coordinates for the bounding box,
            defined relative to an uncropped image with stixel size coord_def_stixel_size
        :param sta_width_coords: low and high width coordinates for the bounding box,
            defined relative to an uncropped image with stixel size coord_def_stixel_size
        :param coord_def_stixel_size: stixel size that the above coordinates are defined
            in. Needs to be a power of two but doesn't have to be the smallest stixel size
        :param full_res_dims: (height, width) of the uncropped image in stixel size
            coord_def_stixel_size. Not necessarily the shape of the full resolution image
        :param full_res_stixel_size: stixel size of the full resolution image; i.e. the stixel
            size of the highest-resolution image that these bounding boxes will ever be used to
            deal with
        '''

        self.height_low, self.height_high = sta_height_coords
        self.width_low, self.width_high = sta_width_coords

        self.coord_def_stixel_size = coord_def_stixel_size

        self.full_res_stixel_size = full_res_stixel_size

        scaleup = coord_def_stixel_size // full_res_stixel_size

        self.fullres_uncrop_hlow, self.fullres_uncrop_hhigh = self.height_low * scaleup, self.height_high * scaleup
        self.fullres_uncrop_wlow, self.fullres_uncrop_whigh = self.width_low * scaleup, self.width_high * scaleup

        self.fullres_height, self.fullres_width = dims[0] * scaleup, dims[1] * scaleup

    def compute_post_crop_downsample_shape(self,
                                           crop_hlow: int = 0,
                                           crop_wlow: int = 0,
                                           crop_hhigh: int = 0,
                                           crop_whigh: int = 0,
                                           downsample_factor: int = 1) -> Tuple[int, int]:
        '''
        Computes the dimensions of the image post-crop and downsample

        Order of operations is
            (1) crop the full-resolution image
            (2) downsample

        :param crop_hlow: int, number of pixels to crop, expressed in units of full resolution image pixels
        :param crop_wlow: int, number of pixels to crop, expressed in units of full resolution image pixel
        :param crop_hhigh: int, number of pixels to crop, expressed in units of full resolution image pixel
        :param crop_whigh: int, number of pixels to crop, expressed in units of full resolution image pixel
        :param downsample_factor: int, downsample factor
        :return:
        '''

        output_height_pre_ds = self.fullres_height - (crop_hlow + crop_hhigh)
        output_width_pre_ds = self.fullres_width - (crop_wlow + crop_whigh)

        output_height = output_height_pre_ds // downsample_factor
        output_width = output_width_pre_ds // downsample_factor

        return output_height, output_width

    def make_precropped_sliceobj(self,
                                 crop_hlow: int = 0,
                                 crop_wlow: int = 0,
                                 crop_hhigh: int = 0,
                                 crop_whigh: int = 0,
                                 downsample_factor: int = 1,
                                 return_bounds: bool = False):
        '''
        Produces a slice that grabs the relevant patch from a pre-cropped image (where the crop
            was defined according to crop_hlow, etc. but already done)

        Semantics:
            * Resolution is defined relative to the full resolution stimulus, which had stixel
                size determined by full_res_stixel_size in the constructor
            * The crop is defined in units of full_res_stixel_size relative to the uncropped image

        :param crop_hlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_wlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_hhigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_whigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param downsample_factor: how much to downsample by, relative to the full resolution stimulus
            must be a power of 2
        :param return_bounds: bool, whether to return Tuple bound rather than slice object. Default False
        :return:
        '''

        hstart_raw = max(self.fullres_uncrop_hlow - crop_hlow, 0)
        hend_raw = min(self.fullres_uncrop_hhigh - crop_hlow, self.fullres_height - crop_hhigh - crop_hlow)
        wstart_raw = max(self.fullres_uncrop_wlow - crop_wlow, 0)
        wend_raw = min(self.fullres_uncrop_whigh - crop_wlow, self.fullres_width - crop_whigh - crop_wlow)

        if downsample_factor == 1:
            if return_bounds:
                return ((hstart_raw, hend_raw), (wstart_raw, wend_raw))
            return np.s_[hstart_raw:hend_raw, wstart_raw:wend_raw]
        else:
            hstart_ds, hend_ds = hstart_raw // downsample_factor, hend_raw // downsample_factor
            wstart_ds, wend_ds = wstart_raw // downsample_factor, wend_raw // downsample_factor
            if return_bounds:
                return ((hstart_ds, hend_ds), (wstart_ds, wend_ds))
            return np.s_[hstart_ds:hend_ds, wstart_ds:wend_ds]

    def make_selection_sliceobj(self, crop_hlow: int = 0, crop_wlow: int = 0, crop_hhigh: int = 0,
                                crop_whigh: int = 0, downsample_factor: int = 1):
        '''
        Makes a slice object for placing the patch into the original full size bounding
            box. Useful for batch processing on a large number of cells.

        Semantics:
            * Resolution is defined relative to the full resolution stimulus, which had stixel
                size determined by full_res_stixel_size in the constructor
            * The crop is defined in units of full_res_stixel_size relative to the uncropped image

        Examples: Assume that the full size bounding box is (16, 16)

            This will typically output slices np.r_[0:16,0:16] corresponding to no crop
                when the image patch is not clipped by any of the boundaries

            When the image patch is clipped by the boundary, it will return a smaller
                slice object, where the edges are determined based on how the image is cropped

        :param crop_hlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_wlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_hhigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_whigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param downsample_factor: how much to downsample by, relative to the full resolution stimulus
            must be a power of 2
        :return:
        '''

        bbox_height_nobound = self.fullres_uncrop_hhigh - self.fullres_uncrop_hlow
        bbox_width_nobound = self.fullres_uncrop_whigh - self.fullres_uncrop_wlow

        clip_hlow = max(0, crop_hlow - self.fullres_uncrop_hlow)
        clip_hhigh = min(bbox_height_nobound, self.fullres_height - crop_hhigh - self.fullres_uncrop_hlow)

        clip_wlow = max(0, crop_wlow - self.fullres_uncrop_wlow)
        clip_whigh = min(bbox_width_nobound, self.fullres_width - crop_whigh - self.fullres_uncrop_wlow)

        if downsample_factor == 1:
            return np.s_[clip_hlow:clip_hhigh, clip_wlow:clip_whigh]
        return np.s_[(clip_hlow // downsample_factor):(clip_hhigh // downsample_factor),
               (clip_wlow // downsample_factor):(clip_whigh // downsample_factor)]

    def make_cropping_sliceobj(self,
                               crop_hlow: int = 0,
                               crop_wlow: int = 0,
                               crop_hhigh: int = 0,
                               crop_whigh: int = 0,
                               downsample_factor: int = 1,
                               return_bounds: bool = False):
        '''
        Makes a slice object, that, given an image that has been downsampled but not cropped
            grabs the same patch as in the cropped case (taking into account the crop margins,
            even though the source image itself hasn't been cropped)

        :param crop_hlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_wlow: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_hhigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param crop_whigh: crop in stixels, defined w.r.t. full resolution stimulus
        :param downsample_factor: how much to downsample by, relative to the full resolution stimulus
            must be a power of 2
        :param return_bounds: bool, whether to return Tuple bound rather than slice object. Default False
        :return:
        '''
        cropping_hlow = max(crop_hlow, self.fullres_uncrop_hlow)
        cropping_hhigh = min(self.fullres_uncrop_hhigh, self.fullres_height - crop_hhigh)

        cropping_wlow = max(crop_wlow, self.fullres_uncrop_wlow)
        cropping_whigh = min(self.fullres_uncrop_whigh, self.fullres_width - crop_whigh)

        if downsample_factor == 1:
            if return_bounds:
                return (cropping_hlow, cropping_hhigh), (cropping_wlow, cropping_whigh)
            return np.s_[cropping_hlow:cropping_hhigh, cropping_wlow:cropping_whigh]
        else:
            if return_bounds:
                return (cropping_hlow // downsample_factor, cropping_hhigh // downsample_factor), \
                       (cropping_wlow // downsample_factor, cropping_whigh // downsample_factor)
            return np.s_[(cropping_hlow // downsample_factor):(cropping_hhigh // downsample_factor),
                   (cropping_wlow // downsample_factor):(cropping_whigh // downsample_factor)]


def make_bounding_box(sig_stixels_matrix: np.ndarray,
                      target_mindim: int,
                      orig_stixel_size: int) -> CroppedSTABoundingBox:
    '''
    Makes a bounding box around the significant stixels
        with a minimum number of stixels per side for the bounding box

    :param sig_stixels_matrix: Sig stixels matrix, computed from white noise STAs
        This has shape corresponding to the raw white noise stixels
    :param target_mindim: Minimum dimension of the bounding box, in units of white noise stixels
    :param orig_stixel_size: Stixel size of the white noise run
    :return: CroppedSTABoundingBox, corresponding to the definition provided
    '''
    height, width = sig_stixels_matrix.shape
    half_side = target_mindim // 2

    com_h, com_w = ndimage.measurements.center_of_mass(np.abs(sig_stixels_matrix))
    com_h_integer, com_w_integer = int(np.rint(com_h)), int(np.rint(com_w))

    hlow, hhigh = com_h_integer - half_side, com_h_integer + half_side
    wlow, whigh = com_w_integer - half_side, com_w_integer + half_side
    return CroppedSTABoundingBox((hlow, hhigh),
                                 (wlow, whigh),
                                 orig_stixel_size,
                                 (height, width))


def make_fixed_size_bounding_box(sig_stixels_matrix: np.ndarray,
                                 fixed_size: int,
                                 wn_stixel_size: int,
                                 crop_width_low: int = 0,
                                 crop_height_low: int = 0,
                                 crop_width_high: int = 0,
                                 crop_height_high: int = 0) -> CroppedSTABoundingBox:
    '''
    Produces a bounding box that, after applying the crops specified by crop_width_low, etc.
        to the natural scenes stimulus, is always guaranteed to have size (fixed_size, fixed_size)

    Explicitly makes the assumption that the (possibly-cropped) stimulus is bigger than
        any bounding box that you can try to make. This guarantees that there exists some
        shift of the bounding box that produces a box that doesn't cut off any of the required
        parts of the stimulus AND has the minimum square size determined by fixed_size

    :param sig_stixels_matrix:
    :param fixed_size: int, defined in terms of WN stixels
    :param wn_stixel_size:
    :param crop_width_low: int, defined in terms of stixels of the natural scenes stimulus,
        which in our case is hard-coded to 2 pixels
    :param crop_height_low: int, defined in terms of stixels of the natural scenes stimulus,
        which in our case is hard-coded to 2 pixels
    :param crop_width_high: int, defined in terms of stixels of the natural scenes stimulus,
        which in our case is hard-coded to 2 pixels
    :param crop_height_high: int, defined in terms of stixels of the natural scenes stimulus,
        which in our case is hard-coded to 2 pixels
    :return:
    '''

    ns_stixels_per_wn_stixel = wn_stixel_size // 2

    height, width = sig_stixels_matrix.shape

    wn_scale_cropped_min_w = crop_width_low // ns_stixels_per_wn_stixel
    wn_scale_cropped_max_w = width - (crop_width_high // ns_stixels_per_wn_stixel)
    wn_scale_cropped_min_h = crop_height_low // ns_stixels_per_wn_stixel
    wn_scale_cropped_max_h = height - (crop_height_high // ns_stixels_per_wn_stixel)

    half_side = fixed_size // 2
    actual_window_side = half_side * 2

    com_h, com_w = ndimage.measurements.center_of_mass(np.abs(sig_stixels_matrix))
    com_h_integer, com_w_integer = int(np.rint(com_h)), int(np.rint(com_w))

    hlow, hhigh = com_h_integer - half_side, com_h_integer + half_side
    wlow, whigh = com_w_integer - half_side, com_w_integer + half_side

    # figure out what the h coordinates need to be
    if hlow < wn_scale_cropped_min_h:
        hlow, hhigh = wn_scale_cropped_min_h, wn_scale_cropped_min_h + actual_window_side
        assert hhigh < wn_scale_cropped_max_h, 'Required bounding box too big'
    elif hhigh > wn_scale_cropped_max_h:
        hlow, hhigh = wn_scale_cropped_max_h - actual_window_side, wn_scale_cropped_max_h
        assert hlow >= wn_scale_cropped_min_h, 'Required bounding box too big'

    # figure out what the w coordinates need to be
    if wlow < wn_scale_cropped_min_w:
        wlow, whigh = wn_scale_cropped_min_w, wn_scale_cropped_min_w + actual_window_side
        assert whigh < wn_scale_cropped_max_w, 'Required bounding box too big'
    elif whigh > wn_scale_cropped_max_w:
        wlow, whigh = wn_scale_cropped_max_w - actual_window_side, wn_scale_cropped_max_w
        assert wlow >= wn_scale_cropped_min_w, 'Required bounding box too big'

    return CroppedSTABoundingBox((hlow, hhigh),
                                 (wlow, whigh),
                                 wn_stixel_size,
                                 (height, width))


def make_flat_index_bbox_placement(bbox: CroppedSTABoundingBox,
                                   downsample_factor: int = 1,
                                   crop_width_low: int = 0,
                                   crop_height_low: int = 0,
                                   crop_height_high: int = 0,
                                   crop_width_high: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    '''
    Computes indices for placing the crop specified by the bounding box into
        the flattened, already cropped image

    :param bbox:
    :param downsample_factor:
    :param crop_width_low:
    :param crop_height_low:
    :param crop_height_high:
    :param crop_width_high:
    :return:
    '''

    uncropped_full_height, uncropped_full_width = bbox.fullres_height, bbox.fullres_width
    full_height = uncropped_full_height - (crop_height_low + crop_height_high)
    full_width = uncropped_full_width - (crop_width_low + crop_width_high)

    (hstart, hend), (wstart, wend) = bbox.make_precropped_sliceobj(
        crop_hlow=crop_height_low,
        crop_hhigh=crop_height_high,
        crop_wlow=crop_width_low,
        crop_whigh=crop_width_high,
        downsample_factor=downsample_factor,
        return_bounds=True
    )

    # height is the first axis,
    # width is the second axis
    bbox_height, bbox_width = hend - hstart, wend - wstart

    npix_sel = bbox_height * bbox_width

    pix_sel = np.zeros((npix_sel, ), dtype=np.int64)

    if bbox_height != 64 or bbox_width != 64:
        print(bbox_height, bbox_width, pix_sel.shape)

    for write_ix, (h, w) in enumerate(itertools.product(range(hstart, hend), range(wstart, wend))):
        pix_sel[write_ix] = h * full_width + w

    return pix_sel, (bbox_height, bbox_width)


def get_max_bounding_box_dimension(bbox_list: List[CroppedSTABoundingBox],
                                   downsample_factor: int = 1,
                                   crop_width_low: int = 0,
                                   crop_height_low: int = 0,
                                   crop_width_high: int = 0,
                                   crop_height_high: int = 0) -> Tuple[int, int]:
    '''
    Determines the maximum size of bounding box used among the list

    :param bbox_list:
    :param downsample_factor:
    :param crop_width_low:
    :param crop_height_low:
    :param crop_width_high:
    :param crop_height_high:
    :return:
    '''

    max_height, max_width = 0, 0
    for bbox in bbox_list:
        (h_start, h_end), (w_start, w_end) = bbox.make_precropped_sliceobj(crop_hlow=crop_height_low,
                                                                           crop_hhigh=crop_height_high,
                                                                           crop_wlow=crop_width_low,
                                                                           crop_whigh=crop_width_high,
                                                                           downsample_factor=downsample_factor,
                                                                           return_bounds=True)

        max_height = max(max_height, h_end - h_start)
        max_width = max(max_width, w_end - w_start)

    return max_height, max_width


def load_sigstixels_spatial_only_stas(vision_dataset: vl.VisionCellDataTable,
                                      cell_id_list_ordered: List[int],
                                      sig_stixels_threshold: float = 4.0) -> np.ndarray:
    '''
    Method for flattening spatial STAs

    The above method blows up the noise floor for stixels that are not part of the STA
        which affects reconstruction performance

    This is a hacky way of denoising the calculated STA filters, which sets all insignificant stixels
        to zero

    :param vision_dataset:
    :param cell_id_list_ordered : list of cell id
    :param sig_stixels_threshold : cutoff for significant stixels
    :return: np.ndarray, shape (ncells, width, height)
    '''

    filter_stack = []  # type: List[np.ndarray]
    for cell_id in cell_id_list_ordered:
        sta_container = vision_dataset.get_sta_for_cell(cell_id)

        sig_stixels = greg_field_simpler_significant_stixels(sta_container,
                                                             sig_stixels_threshold)
        not_significant_stixels = np.logical_not(sig_stixels)

        bw_matrix = 0.2989 * sta_container.red + 0.5870 * sta_container.green + 0.1140 * sta_container.blue
        # shape (width, height, nframes)

        _, _, peak_time = np.unravel_index(np.argmax(np.abs(bw_matrix)), bw_matrix.shape)

        sta_spatial_filter = bw_matrix[:, :, peak_time]
        sta_spatial_filter[not_significant_stixels] = 0
        filter_stack.append(sta_spatial_filter)

    return np.array(filter_stack)  # shape (n_cells, width, height)
