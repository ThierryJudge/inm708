import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib

if __name__ == '__main__':

    file_data = nib.load('Data/fmri.nii.gz')
    img = file_data.get_fdata().squeeze()

    print(img.shape)

    plt.figure()
    plt.imshow(img[:, :, 25, 10])

    file_data = nib.load('Data/Tcorr1D.nii')
    corr = file_data.get_fdata().squeeze()

    print(corr.shape)

    plt.figure()
    plt.hist(corr.flatten(), bins=50)

    plt.figure()
    plt.imshow(corr[:, :, 25])

    with open('Data/ideal.txt') as f:
        lines = f.readlines()
        values = []
        for line in lines:
            values.append(float(line))
        values = np.array(values)

    print(values.shape)

    plt.figure()
    plt.plot(values)
    plt.show()
