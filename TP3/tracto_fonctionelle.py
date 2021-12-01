from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
import nibabel as nib
from matplotlib import pyplot as plt
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

    print(fmri_seg.shape)
    print(len(tractogram.streamlines))

    plt.figure()
    plt.imshow(fmri_seg[:, :, 10])
    for i in range(500):
        stream = tractogram.streamlines[i] * fmri_seg.shape / dmri_data.shape[:3]
        plt.plot(stream[:, 1], stream[:, 0], c='r')
    plt.show()



