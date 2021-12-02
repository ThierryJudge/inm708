from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_seg", type=str, default='seg_cube_like.nii.gz')
    parser.add_argument("--streamlines", type=str, default='filtered_tractogram.trk')
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz')
    args = parser.parse_args()

    dmri_data = nib.load(args.dmri_path)
    dmri = dmri_data.get_fdata()
    print(dmri_data.shape)

    tractogram = load_tractogram(args.streamlines, dmri_data, Space.VOX)

    fmri_seg_data = nib.load(args.fmri_seg)
    fmri_seg = fmri_seg_data.get_fdata()

    indices = []

    for i in range(1, int(fmri_seg.max() + 1)):
        idx = []
        print(i)
        object_idx = np.argwhere(fmri_seg == i)
        min_x = object_idx[:, 0].min()
        max_x = object_idx[:, 0].max()
        min_y = object_idx[:, 1].min()
        max_y = object_idx[:, 1].max()
        min_z = object_idx[:, 2].min()
        max_z = object_idx[:, 2].max()

        print(f"X: {min_x}, {max_x}, Y: {min_y}, {max_y}, Z: {min_z}, {max_z}")

        for j in range(len(tractogram.streamlines)):

            for k in range(len(tractogram.streamlines[j])):
                point = tractogram.streamlines[j][k] * fmri_seg.shape / dmri_data.shape[:3]
                if min_x < point[0] < max_x and min_y < point[1] < max_y and min_z < point[2] < max_z:
                    idx.append(j)
                    break

            # start_point = tractogram.streamlines[j][0] * fmri_seg.shape / dmri_data.shape[:3]
            # end_point = tractogram.streamlines[j][-1] * fmri_seg.shape / dmri_data.shape[:3]
            # if min_x < start_point[0] < max_x and min_y < start_point[1] < max_y and min_z < start_point[2] < max_z:
            #         idx.append(j)
            # if min_x < end_point[0] < max_x and min_y < end_point[1] < max_y and min_z < end_point[2] < max_z:
            #     idx.append(j)

        # print(idx)
        indices.append(idx)

    print("INDICIES", len(indices))

    for i in range(len(indices)):
        print(i)
        print(len(tractogram.streamlines[indices[i]]))
        sft = StatefulTractogram(tractogram.streamlines[indices[i]], dmri_data, Space.VOX)
        save_tractogram(sft, f"tractogram_{i}.trk", bbox_valid_check=False)
        for j in range(len(indices)):
            if i != j:
                print(i, j)
                intersection = list(set(indices[i]) & set(indices[j]))
                print(intersection)
                if len(intersection) > 0:
                    sft = StatefulTractogram(tractogram.streamlines[intersection], dmri_data, Space.VOX)
                    save_tractogram(sft, f"tractogram_{i}_{j}.trk", bbox_valid_check=False)

    # plt.figure()
    # plt.imshow(fmri_seg[:, :, 10])
    # for i in range(500):
    #     stream = tractogram.streamlines[i] * fmri_seg.shape / dmri_data.shape[:3]
    #     plt.plot(stream[:, 1], stream[:, 0], c='r')
    # plt.show()
