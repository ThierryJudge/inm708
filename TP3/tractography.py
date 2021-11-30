import argparse
import random

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_trk, save_tractogram
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from dipy.tracking import utils

def unit_vector(vector):
    """https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


if __name__ == '__main__':

    # from dipy.data import read_stanford_labels
    #
    # hardi_img, gtab, labels_img = read_stanford_labels()
    # data = hardi_img.get_data()
    # labels = labels_img.get_data()
    # affine = hardi_img.affine
    #
    # print(data.shape)
    # print(labels.shape)
    #
    # exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_path", type=str, default='Data/t1.nii.gz')
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz')
    parser.add_argument("--wm_mask_path", type=str, default='Data/t1_skullstrip_pve_2.nii.gz')

    parser.add_argument("--direction_path", type=str, default='principal_dir.nii.gz')
    args = parser.parse_args()

    DEBUG = False

    t1_data = nib.load(args.t1_path)
    t1 = t1_data.get_fdata().squeeze()

    wm_mask_data = nib.load(args.wm_mask_path)
    wm_mask = wm_mask_data.get_fdata().squeeze() > 0

    dmri_data = nib.load(args.dmri_path)

    direction_data = nib.load(args.direction_path)
    direction = direction_data.get_fdata().squeeze()

    print("T1 shape", t1.shape)
    print("wm_mask shape", wm_mask.shape)
    print("Direction shape", direction.shape)

    # plt.figure()
    # plt.imshow(wm_mask[:, :, 82])
    # plt.show()
    #
    # print(wm_mask.max())

    indices = np.transpose(wm_mask.nonzero())
    print("Possible seeds shape", indices.shape)
    print(np.count_nonzero(wm_mask))
    print(direction.shape[:3])

    streamlines = []

    # seeds = utils.seeds_from_mask(wm_mask, dmri_data.affine, density=1)
    # print("Seeds shape", seeds.shape)

    for i in tqdm(range(1000)):
        path = []

        seed = indices[random.randint(0, len(indices) - 1)]
        seed = (seed * direction.shape[:3] / t1.shape).astype(int)
        if DEBUG:
            print("Initial seed", seed)

        # seed = seeds[i]

        seed_round = seed.round().astype(int)
        last_direction = direction[seed_round[0], seed_round[1], seed_round[2]]
        seed = seed + last_direction

        for i in range(100000):
            seed_round = seed.round().astype(int)
            current_direction = direction[seed_round[0], seed_round[1], seed_round[2]]
            angle = angle_between(last_direction, current_direction)

            if (0 < angle < 45) or (315 < angle < 360):
                current_direction = -current_direction

            if (45 < angle < 90) or (270 < angle < 315):
                if DEBUG:
                    print("Illegal angle")
                break

            # If we leave white matter mask
            seed_in_mask = (seed * t1.shape / direction.shape[:3]).round().astype(int)
            if wm_mask[seed_in_mask[0], seed_in_mask[1], seed_in_mask[2]] == 0:
                if DEBUG:
                    print("Left white matter mask")
                break

            seed = seed + current_direction
            last_direction = current_direction
            path.append(seed)

        if len(path) > 10:
            path = np.array(path)
            if DEBUG:
                print("New streamline", path.shape)
            # path = (path * t1.shape / direction.shape[:3]).astype(float)
            streamlines.append(path)

    print(len(streamlines))
    # tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    # stream = nib.streamlines.TrkFile(tractogram, header=dmri_data.header)
    # nib.streamlines.save(stream, "streamline.trk") # with header
    # # nib.save(stream, "streamline.tck") # without header

    sft = StatefulTractogram(streamlines, dmri_data, Space.VOX)
    # save_trk(sft, "tractogram.trk", streamlines)
    save_tractogram(sft, "tractogram.trk", bbox_valid_check=False)

    plt.figure()
    plt.imshow(wm_mask[:, :, 82])
    for i in range(100):
        plt.scatter(streamlines[i][:, 0], streamlines[i][:, 1], c='r', s=5)
    plt.show()
