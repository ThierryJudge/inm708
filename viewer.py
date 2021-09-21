import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons, RangeSlider
from scipy import ndimage

from utils import *


class Viewer:
    PROJECTION_FUNCTIONS = {'MIP': np.max,
                            'mIP': np.min}

    COLORS = {'sagittal': 'r',
              'coronal': 'b',
              'axial': 'g'}

    def __init__(self, img, aspect=None, cmap='gray', rotate=True):

        if img.ndim == 2:
            plt.figure()
            plt.imshow(img)
            plt.show()
            return

        self.imshow_kwargs = {'cmap': cmap, 'aspect': aspect}

        axcolor = 'lightgoldenrodyellow'

        fig, ax = plt.subplots(1, 4)
        plt.subplots_adjust(bottom=0.3, left=0.25)

        img_sagittal = np.moveaxis(img, 0, 0)
        img_coronal = np.moveaxis(img, 1, 0)
        img_axial = np.moveaxis(img, 2, 0)

        if rotate:
            img_sagittal = img_sagittal.swapaxes(1, 2)
            img_sagittal = ndimage.rotate(img_sagittal, 180, reshape=False,axes=(1, 2))

            img_coronal = img_coronal.swapaxes(1, 2)
            img_coronal = ndimage.rotate(img_coronal, 180, reshape=False, axes=(1, 2))

        self.images = {'axial': img_axial,
                       'sagittal': img_sagittal,
                       'coronal': img_coronal}

        self.img_proj = img_axial
        proj_start = 0
        proj_end = self.img_proj.shape[0] // 2
        self.proj_fn = self.PROJECTION_FUNCTIONS['MIP']

        sag_start = img_sagittal.shape[0] // 2
        cor_start = img_coronal.shape[0] // 2
        ax_start = img_axial.shape[0] // 2

        im_sagittal = ax[0].imshow(img_sagittal[sag_start], **self.imshow_kwargs)
        im_coronal = ax[1].imshow(img_coronal[cor_start], **self.imshow_kwargs)
        im_axial = ax[2].imshow(img_axial[ax_start], **self.imshow_kwargs)

        self.im_proj = ax[3].imshow(self.proj_fn(self.img_proj[proj_start:proj_end], axis=0), **self.imshow_kwargs)

        ax[0].set_title("Sagittal", color=self.COLORS['sagittal'])
        ax[1].set_title("Coronal", color=self.COLORS['coronal'])
        ax[2].set_title("Axial", color=self.COLORS['axial'])
        ax[3].set_title("Projection")

        # Setup Sliders

        ax_sagittal = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        slider_sag = Slider(ax_sagittal, 'Sagittal', 0,
                            img_sagittal.shape[0] - 1, valinit=sag_start,
                            valstep=1)

        ax_coronal = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slider_cor = Slider(ax_coronal, 'Coronal', 0, img_coronal.shape[0] - 1,
                            valinit=cor_start, valstep=1)

        ax_axial = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        slider_ax = Slider(ax_axial, 'Axial', 0, img_axial.shape[0] - 1,
                           valinit=ax_start, valstep=1)

        # Setup lines
        sagittal_coronal_line = ax[0].axvline(slider_cor.val,
                                              color=self.COLORS['coronal'])
        sagittal_axial_line = ax[0].axhline(slider_ax.val,
                                            color=self.COLORS['axial'])

        coronal_sagittal_line = ax[1].axvline(slider_sag.val,
                                              color=self.COLORS['sagittal'])
        coronal_axial_line = ax[1].axhline(slider_ax.val,
                                           color=self.COLORS['axial'])

        axial_sagittal_line = ax[2].axhline(slider_sag.val,
                                            color=self.COLORS['sagittal'])
        axial_coronal_line = ax[2].axvline(slider_cor.val,
                                           color=self.COLORS['coronal'])

        def update(val=None):
            idx_sag = slider_sag.val
            idx_cor = slider_cor.val
            idx_ax = slider_ax.val

            # Draw lines
            im_sagittal.set_data(img_sagittal[idx_sag])
            im_coronal.set_data(img_coronal[idx_cor])
            im_axial.set_data(img_axial[idx_ax])

            sagittal_coronal_line.set_xdata(idx_cor)
            sagittal_axial_line.set_ydata(idx_ax)

            coronal_sagittal_line.set_xdata(idx_sag)
            coronal_axial_line.set_ydata(idx_ax)

            axial_sagittal_line.set_ydata(idx_sag)
            axial_coronal_line.set_xdata(idx_cor)

            fig.canvas.draw_idle()

        slider_sag.on_changed(update)
        slider_cor.on_changed(update)
        slider_ax.on_changed(update)

        # Projection
        proj_ax = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.proj_slider = RangeSlider(proj_ax, "Projection", 0,
                                       self.img_proj.shape[0] - 1,
                                       valinit=(proj_start, proj_end),
                                       valstep=1)

        def proj_update(val=None):
            val = self.proj_slider.val
            self.im_proj.set_data(
                self.proj_fn(self.img_proj[val[0]:val[1]], axis=0))
            fig.canvas.draw_idle()

        self.proj_slider.on_changed(proj_update)

        # Projection Type Button
        rax_type = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        radio_type = RadioButtons(rax_type, ('MIP', 'mIP'), active=0)

        def set_proj_type(label):
            self.proj_fn = self.PROJECTION_FUNCTIONS[label]
            proj_update(0)
            fig.canvas.draw_idle()

        radio_type.on_clicked(set_proj_type)

        # Projection img button
        rax_img = plt.axes([0.025, 0.2, 0.15, 0.15], facecolor=axcolor)
        radio_img = RadioButtons(rax_img, ('Sagittal', 'Coronal', 'Axial'),
                                 active=2)

        def set_proj_img(label):
            """Update function for projection selection button"""
            self.img_proj = self.images[label.lower()]

            proj_update(0)

            self.proj_slider.ax.set_xlim(0, self.img_proj.shape[0] - 1)
            self.proj_slider.valmax = self.img_proj.shape[0] - 1
            self.proj_slider.set_val((0, self.img_proj.shape[0] // 2))

            self.im_proj = ax[3].imshow(
                self.proj_fn(self.img_proj[proj_start:proj_end], axis=0),
                **self.imshow_kwargs)

            fig.canvas.draw_idle()

        radio_img.on_clicked(set_proj_img)

        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--aspect", type=str, choices=['auto'], default=None)
    parser.add_argument("--cmap", type=str, choices=['gray'], default='gray')
    parser.add_argument("--no_rotate", dest='rotate', action='store_false')
    args = parser.parse_args()

    print(args)

    file_data = nib.load(args.file_path)

    print(file_data.header)
    img = file_data.get_fdata()

    print(img.shape)

    Viewer(img, aspect=args.aspect, cmap=args.cmap, rotate=args.rotate)

    # if img.ndim == 2:
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.show()
    # else:
    #     Viewer(img)
    #
    # print(img.shape)

    # plt.figure()
    # plt.hist(img.flatten(), bins=50)
    #
    # plt.figure()
    # plt.hist(img[img > 30].flatten(), bins=50)

    # if img.ndim == 2:
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.show()
    # else:
    #     Viewer(img)

    # # 2.b
    # print(bcolors.WARNING + "Question 2.b" + bcolors.ENDC)
    # print(bcolors.OKBLUE + "Michelson contrast" + bcolors.ENDC + ": ")
    # print(michelson_contrast(img[:, :, 15]))
    # print(bcolors.OKBLUE + "RMS contrast" + bcolors.ENDC + ": ")
    # print(rms_contrast(img[:, :, 15]))
    #
    # # 2.e
    # print(bcolors.WARNING + "Question 2.e" + bcolors.ENDC)
    # print(bcolors.OKBLUE + "SNR" + bcolors.ENDC + ": ")
    # print(SNR(im=img[:, :, 15],
    #           S=(0, 0),
    #           fond=(0, 0),
    #           window_size=5))
    #
    # # 3
    #
    # image = img[:, :, 15]
    # gaussian_filtered = gaussian_filter(im=image, s=0.65)
    # median_filtered = median_filter(im=image)
    # bil_filtered = bilateral_filter(im=image)
    #
    # fig = plt.figure(figsize=(10, 10))
    #
    # fig.add_subplot(3, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title("Original")
    #
    # fig.add_subplot(3, 3, 2)
    # plt.imshow(gaussian_filtered, cmap='gray')
    # plt.title("Gaussian filter")
    #
    # fig.add_subplot(3, 3, 3)
    # plt.imshow(image - gaussian_filtered, cmap='gray')
    # plt.title("Original - Gaussian filter")
    #
    # fig.add_subplot(3, 3, 5)
    # plt.imshow(median_filtered, cmap='gray')
    # plt.title("Median filter")
    #
    # fig.add_subplot(3, 3, 6)
    # plt.imshow(image - median_filtered, cmap='gray')
    # plt.title("Original - Median filter")
    #
    # fig.add_subplot(3, 3, 8)
    # plt.imshow(bil_filtered, cmap='gray')
    # plt.title("Bilateral filter")
    #
    # fig.add_subplot(3, 3, 9)
    # plt.imshow(image - bil_filtered, cmap='gray')
    # plt.title("Original - Bilateral filter")
    #
    # plt.show()
