import os
import yaml
import matplotlib.colors as colors
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.ndimage.interpolation import shift

from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt

from viewer import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_path", type=str, help='Path to frmi file')
    parser.add_argument("--ss", type=str, default="histogram",
                        help='skull stripping type')
    parser.add_argument("--t1_path", type=str, help='Path to t1 file')
    parser.add_argument("--ideal_path", type=str, help='Path to ideal file')
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Plot some element (to debug)')
    parser.add_argument('--viewer', dest='viewer', action='store_true',
                        help='Plot some element (to debug)')
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

    # -------------------- 2.1 Removing 3 TR (slides 43) -------------------- #
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
            img_fmri[img_fmri < np.mean(c1)] = 0
        elif np.mean(c2) == BM:
            img_fmri[img_fmri < np.mean(c2)] = 0
        else:
            img_fmri[img_fmri < np.mean(c3)] = 0


    # --------------------- 2.3 Check signal for a voxel --------------------- #
    # for i in [10, 20, 30, 40]:
    #     plt.plot(ideal)
    #     plt.plot([x for x in range(img_fmri.shape[3])], img_fmri[i, 25, 25, :])
    #     plt.show()

    # ------------------------ 2.4 Bluring (slide 50) ------------------------ #
    img_fmri = gaussian_filter(img_fmri, sigma=0.15)
    if args.plot:
        plt.imshow(img_fmri[20, :, :, 20], cmap='gray')
        plt.show()

    ###########################################
    #          STEP 3: Reconstruction         #
    ###########################################

    # --------- 3.1 Analyse correlation between data and ideal trace --------- #

    corr_threshold = 0.5

    def corr(a):
        if np.sum(a) != 0:
            return float(round(np.corrcoef(a, ideal)[0][1], 5))
        else:
            return 0

    corr_values = np.apply_along_axis(corr, 3, img_fmri)

    # --------------------------- 3.2 Segmentation --------------------------- #
    print(corr_values)
    corr_values[corr_values < corr_threshold] = 0
    print(np.sum(corr_values))
    print(np.mean(corr_values))

    new_image = nib.Nifti1Image(corr_values, affine=np.eye(4))
    nib.save(new_image, os.path.join('produced_im', 'corr.nii.gz'))

    plt.imsave('produced_im/proj_sagittal.png', np.sum(corr_values, axis=0), cmap='magma')
    plt.imsave('produced_im/proj_coronal.png', np.sum(corr_values, axis=1), cmap='magma')
    plt.imsave('produced_im/proj_axial.png', np.sum(corr_values, axis=2), cmap='magma')

    a = np.where(corr_values >= corr_threshold)
    indexes_ = np.zeros((len(a[0]), 4))
    for i in range(len(a)):
        indexes_[:, i] = np.array(a[i])
    indexes_ = indexes_[:, :3]
    print(indexes_)

    # -------------------------- 3.3 Post Processing ------------------------- #

    # Attributing each point to a cluster
    clustering = DBSCAN(eps=6, min_samples=12).fit(indexes_)
    print(clustering.labels_)
    clustering.labels_ += 1

    clust_ = np.full(corr_values.shape, 0, dtype=np.uint16)
    for i in range(indexes_.shape[0]):
        clust_[int(indexes_[i, 0]), int(indexes_[i, 1]), int(indexes_[i, 2])] =\
            int(clustering.labels_[i])

    new_image = nib.Nifti1Image(clust_, affine=fmri.affine, header=fmri.header)
    new_image.set_data_dtype(np.uint16)
    nib.save(new_image, os.path.join('produced_im', 'seg2.nii.gz'))

    # ---------------------- Making cube from clusters ---------------------- #
    f_ = np.zeros_like(clust_)
    print(np.unique(clustering.labels_))
    for cluster_i in (np.unique(clustering.labels_)):
        if cluster_i != 0:
            cluster_i_indexes = indexes_[np.where(clustering.labels_ == cluster_i)]
            max_x = int(max(cluster_i_indexes[:, 0]))
            min_x = int(min(cluster_i_indexes[:, 0]))
            max_y = int(max(cluster_i_indexes[:, 1]))
            min_y = int(min(cluster_i_indexes[:, 1]))
            max_z = int(max(cluster_i_indexes[:, 2]))
            min_z = int(min(cluster_i_indexes[:, 2]))

            print('\n\n')
            print(cluster_i)
            print(cluster_i_indexes)
            print(f"{min_x}, {max_x}")
            print(f"{min_y}, {max_y}")
            print(f"{min_z}, {max_z}")

            f_[min_x:max_x, min_y:max_y, min_z:max_z] = cluster_i

    new_image = nib.Nifti1Image(f_, affine=fmri.affine, header=fmri.header)
    new_image.set_data_dtype(np.uint16)
    nib.save(new_image, os.path.join('produced_im', 'seg_cube_like2.nii.gz'))

    plt.imsave('produced_im/seg_proj_sagittal.png', np.sum(clust_, axis=0), cmap='gray')
    plt.imsave('produced_im/seg_proj_coronal.png', np.sum(clust_, axis=1), cmap='gray')
    plt.imsave('produced_im/seg_proj_axial.png', np.sum(clust_, axis=2), cmap='gray')