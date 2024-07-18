import numpy as np
from matplotlib.pyplot import imread
import h5py

import os
import argparse
import multiprocessing as mp
from typing import List
import functools


from lib.data_utils.sta_metadata import RGB_CONVERSION

CROP_HEIGHT = 160
CROP_WIDTH = 256

MAX_IM_PER_CLASS = 500

HALF_CROP_HEIGHT = CROP_HEIGHT // 2
HALF_CROP_WIDTH = CROP_WIDTH // 2


def process_folder_thread(dump_queue: mp.Queue,
                          folder_path: str) -> None:
    folder_content_images = os.listdir(folder_path)

    for im_name in folder_content_images[:MAX_IM_PER_CLASS]:
        im = imread(os.path.join(folder_path, im_name))

        height, width = im.shape[0], im.shape[1]

        if height >= CROP_HEIGHT and width >= CROP_WIDTH:

            if im.ndim == 3:
                # shape (height, width, 3) @ (1, 3, 1)
                # -> (height, width, 1) -> (height, width)
                im_bw = np.clip(np.round((im @ RGB_CONVERSION[:, :, None]).squeeze(2)),
                                a_min=0, a_max=255).astype(np.float32)
            else:
                im_bw = im

            # now need to apply crop
            center_h, center_w = height // 2, width // 2
            low_h, high_h = center_h - HALF_CROP_HEIGHT, center_h + HALF_CROP_HEIGHT
            low_w, high_w = center_w - HALF_CROP_WIDTH, center_w + HALF_CROP_WIDTH

            cropped_im_bw = im_bw[low_h:high_h, low_w:high_w]

            dump_queue.put(cropped_im_bw)


def write_hdf5_thread(dump_queue: mp.Queue,
                      fname: str):

    dset_len = 0
    with h5py.File(fname, mode='w') as h5_file:

        while True:
            if not dump_queue.empty():
                im_to_write = dump_queue.get()
                if im_to_write is not None:
                    if dset_len == 0:
                        h5_file.create_dataset('data', data=im_to_write[None, :, :],
                                               maxshape=(None, CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)
                    else:
                        h5_file["data"].resize(h5_file["data"].shape[0] + 1, axis=0)
                        h5_file["data"][-1, :, :] = im_to_write
                    dset_len += 1
                else:
                    break


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Create hdf5 dataset for training denoiser model')
    parser.add_argument('imagenet_path', type=str, help='path to imagenet directory')
    parser.add_argument('hdf5_path', type=str, help='path to hdf5_folder')

    args = parser.parse_args()

    prefix = args.imagenet_path
    directory_list = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

    m = mp.Manager()
    dump_queue = m.Queue()

    p = mp.Process(target=write_hdf5_thread, args=(dump_queue, args.hdf5_path))
    p.start()

    pool = mp.Pool(processes=min(8, len(directory_list)))
    pool.map(functools.partial(process_folder_thread, dump_queue), directory_list)

    dump_queue.put(None)

    print('should be done')
    p.join()