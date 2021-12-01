import yaml
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt

from viewer import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_path", type=str)
    parser.add_argument("--ss", type=str, default="histogram")
    parser.add_argument("--t1_path", type=str)
    parser.add_argument("--ideal_path", type=str)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--viewer', dest='viewer', action='store_true')
    args = parser.parse_args()

    if args.fmri_path:
        fmri = nib.load(args.fmri_path)
    else:
        fmri = nib.load("Data/fmri.nii")

    if args.ideal_path:
        ideal = np.loadtxt(args.ideal_path, dtype=float)
    else:
        ideal = np.loadtxt("Data/ideal.txt", dtype=float)

    if args.t1_path:
        t1 = nib.load(args.ideal_path)
    else:
        t1 = nib.load("Data/t1.nii")

    img_fmri = fmri.get_fdata().squeeze()
    img_t1 = t1.get_fdata().squeeze()

    print(f"Image size: {img_fmri.shape}")
    print(f"Voxel size: {fmri.header.get_zooms()}")
    print(f"ideal size: {len(ideal)}")

    ideal /= np.linalg.norm(ideal, axis=0)


    ############################################
    # STEP 1: Visualization of data (slide 38) #
    ############################################

    # ------------------------- 1.1: Checking Data -------------------------- #
    if args.viewer:
        Viewer(img_fmri[:, :, :, 10])

    ###########################################
    #          STEP 2: Preprocessing          #
    ###########################################

    # --------------------------- Normalize data ---------------------------- #
    for i in range(img_fmri.shape[3]):
        s = np.max(img_fmri[..., i])
        img_fmri[..., i] /= s
    # -------------------------- 2.1 Removing 3 TR -------------------------- #
    img_fmri = img_fmri[:, :, :, 3:]
    ideal = ideal[3:]

    # ------------- 2.2 Skull stripping based on histogram of t1 ------------ #
    if args.ss == "histogram":
        img_fmri[img_fmri < 0.1] = 0

    # ----------------- 2.2.bis Skull stripping with Kmeans ----------------- #
    if args.ss == "kmeans":
        kmeans = KMeans(n_clusters=3, random_state=0).fit(
            img_fmri.reshape(-1, 1))

        c1 = img_fmri.reshape(-1, 1)[kmeans.labels_ == 0]
        c2 = img_fmri.reshape(-1, 1)[kmeans.labels_ == 1]
        c3 = img_fmri.reshape(-1, 1)[kmeans.labels_ == 2]
        # print(img_fmri.reshape(-1, 1).squeeze())

        BM = np.min([np.mean(c1), np.mean(c2), np.mean(c3)])

        if np.mean(c1) == BM:
            img_fmri[img_fmri < np.max(c1)] = 0
        elif np.mean(c2) == BM:
            img_fmri[img_fmri < np.max(c2)] = 0
        else:
            img_fmri[img_fmri < np.max(c3)] = 0

    # --------------------- 2.3 Check signal for a voxel --------------------- #
    # for i in [10, 20, 30, 40]:
    #     plt.plot(ideal)
    #     plt.plot([x for x in range(img_fmri.shape[3])], img_fmri[i, 25, 25, :])
    #     plt.show()

    # ------------------------ 2.4 Bluring (slide 50) ------------------------ #
    img_fmri = gaussian_filter(img_fmri, sigma=0.5)
    if args.plot:
        plt.imshow(img_fmri[20, :, :, 20], cmap='gray')
        plt.show()

    ###########################################
    #          STEP 3: Reconstruction         #
    ###########################################

    # --------- 3.1 Analyse correlation between data and ideal trace --------- #

    # ideal_matrix = np.tile(ideal, (img_fmri.shape[0],
    #                                img_fmri.shape[1],
    #                                img_fmri.shape[2],
    #                                1))

    corr_threshold = 3

    def corr(a):
        return np.correlate(a, ideal)

    corr_values = np.apply_along_axis(corr, 3, img_fmri)

    # --------------------------- 3.2 Segmentation --------------------------- #

    test = np.copy(corr_values)
    test[test < corr_threshold] = 0
    Viewer(test)

    a = np.where(corr_values > corr_threshold)
    indexes_ = np.zeros((len(a[0]), 3))
    for i in range(len(a)-1):
        indexes_[:, i] = np.array(a[i])

    print(indexes_)
