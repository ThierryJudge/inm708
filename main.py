import yaml
import nibabel as nib
from utils import *

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

for name, file_config in config.items():
    print("====================================================================")
    print(f"{name} at {file_config['path']}")

    file_data = nib.load(file_config['path'])
    img = file_data.get_fdata()

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
        pass







