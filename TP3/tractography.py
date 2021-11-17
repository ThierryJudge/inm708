import argparse
import random

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm


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
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz')

    parser.add_argument("--direction_path", type=str, default='principal_dir.nii.gz')
    args = parser.parse_args()

    t1_data = nib.load(args.t1_path)
    t1 = t1_data.get_fdata().squeeze()

    dmri_data = nib.load(args.dmri_path)


    # print(t1_data.header)
    # t1 = t1[..., None]
    # dtype=np.float32
    # img = nib.Nifti1Image(t1.astype(dtype), t1_data.affine, header=t1_data.header)
    # img.set_data_dtype(dtype)
    # nib.save(img, "new_t1.nii.gz")

    direction_data = nib.load(args.direction_path)
    direction = direction_data.get_fdata().squeeze()

    print("T1 shape", t1.shape)
    print("Direction shape", direction.shape)

    white_matter_mask = get_white_matter_maks(t1, True)

    indices = np.transpose(white_matter_mask.nonzero())
    print("Possible seeds shape", indices.shape)

    streamlines = []

    for i in tqdm(range(5000)):
        path = []
        seed = indices[random.randint(0, len(indices) - 1)]
        seed = (seed * direction.shape[:3] / t1.shape).astype(int)
        try:
            for i in range(10000):
                seed = (seed + direction[seed[0], seed[1], seed[2]]).round().astype(int)
                path.append(seed)
                # print(seed)
                # print(direction[seed[0], seed[1], seed[2]])

            path = np.array(path)
            path = (path * t1.shape / direction.shape[:3]).astype(int)
            streamlines.append(path)
        except Exception as e:
            pass

    print(len(streamlines))
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    stream = nib.streamlines.TrkFile(tractogram, header=dmri_data.header)
    nib.streamlines.save(stream, "streamline.trk") # with header
    # nib.save(img, "streamline.tck") # without header

    #
    # plt.figure()
    # plt.imshow(t1[:, :, path[0, 2]])
    # for i in range(100):
    #     plt.scatter(streamlines[i][:, 0], streamlines[i][:, 1], c=range(len(streamlines[i])), s=5)
    # plt.show()
