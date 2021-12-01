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


def track_one_direction(initial_seed, principal_directions, wm_mask):
    path = []
    last_direction = principal_directions[tuple(initial_seed.round().astype(int))]
    seed = initial_seed + last_direction
    path.append(initial_seed)
    for i in range(100000):
        current_direction = principal_directions[tuple(seed.round().astype(int))]
        angle = angle_between(last_direction, current_direction)
        current_direction = current_direction if angle < 90 else -current_direction

        if (45 < angle < 90) or (270 < angle < 315):
            if DEBUG:
                print("Illegal angle")
            break

        # If we leave white matter mask
        seed_in_mask = (seed * wm_mask.shape / principal_directions.shape[:3]).round().astype(int)
        if wm_mask[seed_in_mask[0], seed_in_mask[1], seed_in_mask[2]] == 0:
            if DEBUG:
                print("Left white matter mask")
            break

        seed = seed + current_direction

        seed = seed + current_direction * 2
        last_direction = current_direction
        path.append(seed)

        ######
        # p = np.array(path) * wm_mask.shape / principal_directions.shape[:3]
        # print(p.shape)
        # plt.figure()
        # plt.imshow(wm_mask[:, :, p[0, 2].round().astype(int)])
        # plt.plot(p[:, 1], p[:, 0], c='r')
        # plt.show()

    return np.array(path)


def unit_vector(vector):
    """https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz')
    parser.add_argument("--wm_mask_path", type=str, default='Data/t1_skullstrip_pve_2.nii.gz')
    parser.add_argument("--direction_path", type=str, default='principal_dir.nii.gz')
    parser.add_argument("--nb_seeds", type=int, default=1000, help='Number of seeds')
    parser.add_argument("--random_seed", type=int, default=1234, help='Number of seeds')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    DEBUG = False

    wm_mask_data = nib.load(args.wm_mask_path)
    wm_mask = wm_mask_data.get_fdata().squeeze() > 0

    dmri_data = nib.load(args.dmri_path)

    direction_data = nib.load(args.direction_path)
    principal_directions = direction_data.get_fdata().squeeze()

    print("wm_mask shape", wm_mask.shape)
    print("Direction shape", principal_directions.shape)

    seeds = get_seeds(wm_mask, args.nb_seeds)
    streamlines = []

    for i in tqdm(range(len(seeds))):
        initial_seed = seeds[i]
        initial_seed = (initial_seed * principal_directions.shape[:3] / wm_mask.shape).astype(int)

        stream1 = track_one_direction(initial_seed, principal_directions, wm_mask)
        stream2 = track_one_direction(initial_seed, -principal_directions, wm_mask)

        stream = np.vstack((stream1[1:][::-1], stream2))  # ignore point and reverse stream1 and concat with stream2

        s1 = stream1 * wm_mask.shape / principal_directions.shape[:3]
        s2 = stream2 * wm_mask.shape / principal_directions.shape[:3]

        # plt.figure()
        # plt.imshow(wm_mask[:, :, s1[0, 2].round().astype(int)])
        # plt.plot(s1[:, 1], s1[:, 0], c='r')
        # plt.plot(s2[:, 1], s2[:, 0], c='b')
        # plt.show()

        if len(stream) > 15:
            # stream = np.array(stream) * wm_mask.shape / principal_directions.shape[:3]
            # print(path.shape)
            # plt.figure()
            # plt.imshow(wm_mask[:, :, path[0, 2].round().astype(int)])
            # plt.scatter(path[:, 1], path[:, 0], c='r', s=5)
            # plt.show()

            if DEBUG:
                print("New streamline", len(stream))
            streamlines.append(stream)

    print(len(streamlines))
    sft = StatefulTractogram(streamlines, dmri_data, Space.VOX)
    save_tractogram(sft, "tractogram.trk", bbox_valid_check=False)
