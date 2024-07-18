import h5py
import numpy as np
import torch.utils.data as torch_data

from typing import List


class ImagenetMaskedDataloader(torch_data.Dataset):

    FRAMES_KEY = 'data'

    def __init__(self,
                 hdf5_path: str,
                 noise_levels: List[float],
                 masks: List[np.ndarray],
                 augment_masks: bool = True):

        self.hdf5_path = hdf5_path
        self.dataset_frames = None

        self.noise_levels = noise_levels # units of STD

        with h5py.File(self.hdf5_path, 'r') as file:
            self.dataset_len, self.height, self.width = file[self.FRAMES_KEY].shape

        if augment_masks:
            self.masks = self.generate_mask_augmentations(masks)
        else:
            self.masks = np.array(masks, dtype=np.float32)
        self.mask_sel_ix = np.r_[0:self.masks.shape[0]]

    def generate_mask_augmentations(self,
                                    masks: List[np.ndarray]) -> np.ndarray:
        '''
        Generates a set of augmented valid region masks
            for the image given real valid region masks

        Augmentations: flips only, since those are easy

        Also add a no-mask, since our goal is to train the network
            to use the mask to determine which parts of the image are useful
        :param masks:
        :return:
        '''

        augmented_masks_list = []
        for mask in masks:
            augmented_masks_list.append(mask)
            augmented_masks_list.append(mask[::-1, :])
            augmented_masks_list.append(mask[:, ::-1])
            augmented_masks_list.append(mask[::-1, ::-1])

        augmented_masks_list.append(np.ones((self.height, self.width), dtype=np.float32))

        return np.array(augmented_masks_list, dtype=np.float32)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        if self.dataset_frames is None:
            self.dataset_frames = h5py.File(self.hdf5_path, 'r')[self.FRAMES_KEY]

        # either shape (batch, height, width) or
        # shape (height, width)
        selected_frames = np.array(self.dataset_frames[index, ...],
                                   dtype=np.float32)

        if selected_frames.ndim == 2:

            # shape (1, )
            noise_levels = np.random.choice(self.noise_levels, size=1, replace=True).astype(np.float32)

            # shape (height, width)
            noise_level_array = np.ones_like(selected_frames) * noise_levels

            ##### randomly pick a mask #############
            random_mask_ix = np.random.choice(self.mask_sel_ix, size=1, replace=True)

            # shape (height, width)
            selected_masks = self.masks[random_mask_ix, ...].squeeze(0)

            masked_selected_frames = selected_frames * selected_masks + (1.0 - selected_masks) * (255 / 2.0)

            ##### corrupt the image ###############
            # shape (height, width)
            scaled_noise = noise_levels * np.random.randn(*selected_frames.shape).astype(np.float32)

            # shape (height, width)
            corrupted_image = masked_selected_frames + scaled_noise

            ##### Build the training input #########
            # shape (3, height, width)
            nn_input = np.stack([corrupted_image, noise_level_array, 255 * selected_masks], axis=0)

            return nn_input, selected_frames, selected_masks

        else:
            # ndim is 3
            batch = selected_frames.shape[0]

            ##### randomly pick noise level #######
            noise_levels = np.random.choice(self.noise_levels,
                                            size=batch, replace=True).astype(np.float32)

            # shape (batch, height, width)
            noise_level_array = np.ones_like(selected_frames) * noise_levels[:, None, None]

            ##### randomly pick a mask #############
            random_mask_ix = np.random.choice(self.mask_sel_ix,
                                              size=batch, replace=True)

            # shape (batch, height, width)
            selected_masks = self.masks[random_mask_ix, ...]

            ##### corrupt the image ###############
            # shape (batch, height, width)
            additive_noise = np.random.randn(*selected_frames.shape).astype(np.float32)
            scaled_noise = noise_levels[None, :, :] * additive_noise

            masked_selected_frames = selected_frames * selected_masks (1.0 - selected_masks) * (255 / 2.0)

            # shape (batch, height, width)
            corrupted_image = masked_selected_frames + scaled_noise

            ##### Build the training input #########
            # shape (batch, 3, height, width)
            nn_input = np.stack([corrupted_image, noise_level_array, 255 * selected_masks], axis=1)

            return nn_input, selected_frames, selected_masks
