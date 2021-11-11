import argparse

import numpy as np
import nibabel as nib
from numpy import linalg as LA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmri_path", type=str, default='Data/dmri.nii.gz')
    parser.add_argument("--grad_path", type=str, default='Data/gradient_directions_b-values.txt')
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

    print("Image shape", img.shape)
    print("b-values shape", b_values.shape)
    print("q shape", q.shape)
    print("B matrix shape", B.shape)
    print("S shape", S.shape)
    print("S0 shape", S0.shape)

    X = -1 / b_values * np.log(S / (S0 + 1e-8) + 1e-8)
    print("X shape", X.shape)

    B1 = np.linalg.inv((B.T @ B)) @ B.T
    D = np.dot(B1, X.transpose(0, 1, 3, 2)).transpose(1, 2, 3, 0)

    print("D shape", D.shape)

    tensor = np.array([
        [D[..., 0], D[..., 1], D[..., 2]],
        [D[..., 1], D[..., 3], D[..., 4]],
        [D[..., 2], D[..., 4], D[..., 5]]
    ]).transpose(2, 3, 4, 0, 1)

    print("Tensor shape", tensor.shape)

    img = nib.Nifti1Image(D, file_data.affine, header=file_data.header)
    nib.save(img, "tensor.nii.gz")

    w, v = LA.eig(tensor)
    max_idx = w.max(axis=-1)
    w2 = np.equal(w, max_idx[..., None])
    principal = v[w2].reshape((128, 128, 60, 3))  # Get larges eigen vector for each voxel.

    print("Principal direction shape", principal.shape)

    img = nib.Nifti1Image(principal, file_data.affine, header=file_data.header)
    nib.save(img, "principal_dir.nii.gz")
