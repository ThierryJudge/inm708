import argparse

import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from numpy import linalg as LA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz', help="Path to the input dmri file (.nii)")
    parser.add_argument("--grad_path", type=str, default='Data/gradient_directions_b-values.txt',
                        help="Path to the input dmri gradient information file (.txt)")
    args = parser.parse_args()

    file_data = nib.load(args.dmri_path)
    img = file_data.get_fdata().squeeze()

    with open(args.grad_path) as f:
        lines = f.readlines()
        values = []
        for line in lines:
            nums = line.split()  # split the line into a list of strings by whitespace
            nums = list(map(float, nums))  # turn each string into a float
            values.append(nums)
        values = np.array(values)

    q, b_values = values[1:, 0:3], values[1:, 3]

    B = np.array([q[:, 0] ** 2,
                  2 * q[:, 0] * q[:, 1],
                  2 * q[:, 0] * q[:, 2],
                  q[:, 1] ** 2,
                  2 * q[:, 1] * q[:, 2],
                  q[:, 2] ** 2,
                  ]).T

    S = img[..., 1:]
    S0 = img[..., 0][..., None]  # Add last dimension to broadcast

    _, mask = median_otsu(img[..., 0], median_radius=3, numpass=2)

    print("Image shape", img.shape)
    print("Mask shape", mask.shape)
    print("b-values shape", b_values.shape)
    print("q shape", q.shape)
    print("B matrix shape", B.shape)
    print("S shape", S.shape)
    print("S0 shape", S0.shape)

    X = -1 / b_values * np.log(S / (S0 + 1e-8) + 1e-8)
    print("X shape", X.shape)

    B1 = np.linalg.inv((B.T @ B)) @ B.T
    D = np.dot(B1, X.transpose(0, 1, 3, 2)).transpose(1, 2, 3, 0)

    D[mask == 0, :] = 0

    print("D shape", D.shape)

    tensor = np.array([
        [D[..., 0], D[..., 1], D[..., 2]],
        [D[..., 1], D[..., 3], D[..., 4]],
        [D[..., 2], D[..., 4], D[..., 5]]
    ]).transpose(2, 3, 4, 0, 1)

    print("Tensor shape", tensor.shape)

    dtype = np.float32
    img = nib.Nifti1Image(D.astype(dtype), file_data.affine, header=file_data.header)
    img.set_data_dtype(dtype)
    nib.save(img, "tensor.nii.gz")

    w, v = LA.eigh(tensor)  # Assumes symmetric matrix and returns sorted eigen vectors and values.
    print("Eigenvalues shape: ", w.shape)
    print("Eigenvector shape: ", v.shape)

    # From dipy.dti
    vals = np.rollaxis(w, -1)
    fa = np.sqrt(0.5 * ((vals[0] - vals[1]) ** 2 +
                        (vals[1] - vals[2]) ** 2 +
                        (vals[2] - vals[0]) ** 2) / ((vals * vals).sum(0) + 1e-8))
    img = nib.Nifti1Image(fa, file_data.affine, header=file_data.header)
    nib.save(img, "fa.nii.gz")

    adc = np.mean(D, axis=-1)
    adc[mask == 0] = 0
    img = nib.Nifti1Image(adc, file_data.affine, header=file_data.header)
    nib.save(img, "adc.nii.gz")


    principal = v[:, :, :, :, -1]
    print("Principal direction shape", principal.shape)
    img = nib.Nifti1Image(principal, file_data.affine, header=file_data.header)
    nib.save(img, "principal_dir.nii.gz")
