import argparse
import random

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_seeds(seeding_mask, nb_seed):
    """Return nb_seeds from seeding_mask"""
    possible_seeds = np.transpose(seeding_mask.nonzero())
    seed_indices = np.random.choice(len(possible_seeds), nb_seed)
    return possible_seeds[seed_indices]


def track_one_direction(initial_seed, principal_directions, wm_mask, step_size, voxelsize):
    path = []
    last_direction = principal_directions[tuple(initial_seed.round().astype(int))]
    last_direction = unit_vector(last_direction, voxel_size=voxelsize)
    len = np.linalg.norm(last_direction) * step_size
    seed = initial_seed + last_direction * step_size
    path.append(initial_seed)
    while True:
        current_direction = principal_directions[tuple(seed.round().astype(int))]
        current_direction = unit_vector(current_direction, voxelsize)
        len += np.linalg.norm(current_direction) * step_size

        angle = angle_between(last_direction, current_direction)
        current_direction = current_direction if angle < 90 else -current_direction

        if 45 < angle < 90:
            if DEBUG:
                print("Illegal angle")
            break

        # If we leave white matter mask
        seed_in_mask = (seed * wm_mask.shape / principal_directions.shape[:3]).round().astype(int)
        if wm_mask[seed_in_mask[0], seed_in_mask[1], seed_in_mask[2]] == 0:
            if DEBUG:
                print("Left white matter mask")
            break

        seed = seed + current_direction * step_size
        last_direction = current_direction
        path.append(seed)

    return np.array(path), len


def unit_vector(vector, voxel_size=None):
    """https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
    voxel_size = voxel_size or np.ones_like(vector)
    return vector / np.linalg.norm(vector * voxel_size)


def angle_between(v1, v2):
    """Returns angle between two vectors. Angle between 0 and 180.

    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz', help="Path to the DMRI nifti data")
    parser.add_argument("--wm_mask_path", type=str, default='Data/t1_skullstrip_pve_2.nii.gz',
                        help="Path to the white matter mask file (.nii)")
    parser.add_argument("--direction_path", type=str, default='principal_dir.nii.gz',
                        help="Path to the principal direction file (.nii)")
    parser.add_argument("--nb_seeds", type=int, default=100000, help='Number of seeds')
    parser.add_argument("--random_seed", type=int, default=1234, help='Number of seeds')
    parser.add_argument("--step_size", type=float, default=0.5, help='Step size in mm')
    parser.add_argument("--min_length", type=float, default=20, help='Minimum length of streamline in mm')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    DEBUG = False

    wm_mask_data = nib.load(args.wm_mask_path)
    wm_mask = wm_mask_data.get_fdata().squeeze() > 0

    dmri_data = nib.load(args.dmri_path)

    direction_data = nib.load(args.direction_path)
    principal_directions = direction_data.get_fdata().squeeze()

    print("White matter mask shape", wm_mask.shape)
    print("White matter mask voxelsize", wm_mask_data.header.get_zooms())
    print("Principal direction shape", principal_directions.shape)
    print("Principal direction voxelsize", direction_data.header.get_zooms())

    seeds = get_seeds(wm_mask, args.nb_seeds)
    streamlines = []
    filtered_streamlines = []
    lengths, filtered_lengths = [], []

    for i in tqdm(range(len(seeds))):
        initial_seed = seeds[i]
        initial_seed = (initial_seed * principal_directions.shape[:3] / wm_mask.shape).astype(int)

        stream1, len1 = track_one_direction(initial_seed, principal_directions, wm_mask, args.step_size,
                                            direction_data.header.get_zooms()[:3])
        stream2, len2 = track_one_direction(initial_seed, -principal_directions, wm_mask, args.step_size,
                                            direction_data.header.get_zooms()[:3])

        stream = np.vstack((stream1[1:][::-1], stream2))  # ignore point and reverse stream1 and concat with stream2

        length = len1 + len2

        if DEBUG:
            print("Streamline length", length)
            s1 = stream1 * wm_mask.shape / principal_directions.shape[:3]
            s2 = stream2 * wm_mask.shape / principal_directions.shape[:3]
            plt.figure()
            plt.imshow(wm_mask[:, :, s1[0, 2].round().astype(int)])
            plt.plot(s1[:, 1], s1[:, 0], c='r')
            plt.plot(s2[:, 1], s2[:, 0], c='b')
            plt.show()

        streamlines.append(stream)
        lengths.append(length)
        if length > args.min_length:
            filtered_streamlines.append(stream)
            filtered_lengths.append(length)

    lengths = np.array(lengths)
    filtered_lengths = np.array(filtered_lengths)
    print("All streamlines: ", len(streamlines))
    print("Min length: ", lengths.min())
    print("Max length: ", lengths.max())
    print("Mean length: ", lengths.mean())
    print("Std length: ", lengths.std())

    print("Filtered streamlines: ", len(filtered_streamlines))
    print("Min length: ", filtered_lengths.min())
    print("Max length: ", filtered_lengths.max())
    print("Mean length: ", filtered_lengths.mean())
    print("Std length: ", filtered_lengths.std())

    sft = StatefulTractogram(streamlines, dmri_data, Space.VOX)
    save_tractogram(sft, "tractogram.trk", bbox_valid_check=False)

    sft = StatefulTractogram(filtered_streamlines, dmri_data, Space.VOX)
    save_tractogram(sft, "filtered_tractogram.trk", bbox_valid_check=False)
