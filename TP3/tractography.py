import argparse
import random

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


def select_largest_blob(mask):
    lbl, num = ndimage.measurements.label(mask)

    # Count the number of elements per label
    count = np.bincount(lbl.flat)

    if not np.any(count[1:]):
        return mask

    # Select the largest blob
    maxi = np.argmax(count[1:]) + 1

    # Remove the other blobs
    lbl[lbl != maxi] = 0

    return lbl


def get_white_matter_maks(t1, debug=False):
    white_matter_mask_min = 200
    white_matter_mask_max = 400
    white_matter_mask = t1 > white_matter_mask_min
    # white_matter_mask = white_matter_mask > white_matter_mask_max

    img = nib.Nifti1Image(white_matter_mask, t1_data.affine, header=t1_data.header)
    nib.save(img, "white_matter_mask.nii.gz")

    # for i in range(white_matter_mask.shape[-1]):
    #     white_matter_mask[..., i] = select_largest_blob(white_matter_mask[..., i])
    #
    # print("White matter mask shape", white_matter_mask.shape)
    #
    # img = nib.Nifti1Image(white_matter_mask, t1_data.affine, header=t1_data.header)
    # nib.save(img, "white_matter_filtered_mask.nii.gz")

    if debug:
        plt.figure()
        plt.imshow(t1[:, :, 100])

        background_mask_value = 20
        plt.figure()
        plt.hist(t1[t1 > background_mask_value].flatten(), bins=50)

        plt.figure()
        plt.imshow(white_matter_mask[:, :, 100])

    return white_matter_mask.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_path", type=str, default='Data/t1.nii.gz')
    parser.add_argument("--direction_path", type=str, default='principal_dir.nii.gz')
    args = parser.parse_args()

    t1_data = nib.load(args.t1_path)
    t1 = t1_data.get_fdata().squeeze()

    direction_data = nib.load(args.direction_path)
    direction = direction_data.get_fdata().squeeze()

    print("T1 shape", t1.shape)
    print("Direction shape", direction.shape)

    white_matter_mask = get_white_matter_maks(t1)

    indices = np.transpose(white_matter_mask.nonzero())
    print("Possible seeds shape", indices.shape)

    seed = indices[random.randint(0, len(indices) - 1)]
    seed = (seed * direction.shape[:3] / t1.shape).astype(int)

    print(seed)
    print(direction[seed[0], seed[1], seed[2]])

    path = []
    for i in range(1000):
        seed = (seed + direction[seed[0], seed[1], seed[2]]).round().astype(int)
        path.append(seed)
        print(seed)
        print(direction[seed[0], seed[1], seed[2]])

    path = np.array(path)
    path = (path * t1.shape / direction.shape[:3]).astype(int)

    print(path.shape)

    plt.figure()
    plt.imshow(t1[:, :, path[0, 2]])
    plt.scatter(path[:, 0], path[:, 1], c='r', s=5)
    plt.show()
