import yaml
import nibabel as nib
from utils import *
from matplotlib import pyplot as plt
from filters import *
import time
filters = {
    "gaussian": gaussian_filter,
    "bilateral": bilateral_filter,
    "nlmeans": nlmeans_filter,
    "anisotropic_diffusion": anisotropic_diffusion_filter
}

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

for name, file_config in config.items():
    print("====================================================================")
    print(f"{name} at {file_config['path']}")

    file_data = nib.load(file_config['path'])
    img = file_data.get_fdata().squeeze()

    print(bcolors.WARNING + "Question 2.a" + bcolors.ENDC)
    print(f"Image size: {img.shape}")
    print(f"Voxel size: {file_data.header.get_zooms()}")

    print(bcolors.WARNING + "Question 2.b" + bcolors.ENDC)
    print(bcolors.OKBLUE + "Michelson contrast" + bcolors.ENDC + ": ")
    print(michelson_contrast(img))
    print(bcolors.OKBLUE + "RMS contrast" + bcolors.ENDC + ": ")
    print(rms_contrast(img))

    if 'snr' in file_config.keys():
        print(bcolors.WARNING + "Question 2.e" + bcolors.ENDC)
        print(bcolors.OKBLUE + "SNR" + bcolors.ENDC + ": ")
        for snr_region, snr_config in file_config['snr'].items():
            snr = SNR(img, snr_config['fg'], snr_config['bg'], snr_config['window_size'])
            print(f"{snr_region} SNR {snr}")

    if 'filter' in file_config.keys():
        print(bcolors.WARNING + "Question 3" + bcolors.ENDC)
        nb_filters = len(file_config['filter'].keys())
        fig = plt.figure(figsize=(10, 10))

        slice = img.shape[2] // 2

        fig.add_subplot(nb_filters, 3, 1)
        plt.suptitle(name)
        plt.imshow(img[:, :, slice], cmap='gray')
        plt.title("Original")

        fig.add_subplot(nb_filters, 3, 4)
        plt.hist(img.flatten(), bins=50)
        plt.title("Histogram")

        if 'nlmeans' in file_config['filter'].keys():
            mask_value = file_config['filter']['nlmeans']['mask_value']
            fig.add_subplot(nb_filters, 3, 7)
            plt.hist(img[img > mask_value].flatten(), bins=50)
            plt.title(f"Masked Histogram with mask > {mask_value}")

        i = 2
        for filter_name, filter_config in file_config['filter'].items():
            start = time.time()
            filtered = filters[filter_name](img, **filter_config)
            end = time.time()
            print(f"{filter_name} {end-start}")
            fig.add_subplot(nb_filters, 3, i)
            plt.imshow(filtered[:, :, slice], cmap='gray')
            plt.title(f"{filter_name}")
            i += 1

            fig.add_subplot(nb_filters, 3, i)
            plt.imshow(img[:, :, slice] - filtered[:, :, slice], cmap='gray')
            plt.title(f"Original - {filter_name}")
            i += 2

            # Recompute SNR
            if 'snr' in file_config.keys():
                for snr_region, snr_config in file_config['snr'].items():
                    snr = SNR(filtered, snr_config['fg'], snr_config['bg'], snr_config['window_size'])
                    print(f"{filter_name} {snr_region} SNR {snr}")
            print('-----')

        plt.savefig(f"{name}_filters.png")

# plt.show()