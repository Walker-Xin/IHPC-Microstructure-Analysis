import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple


def create_circular_mask(img: np.ndarray, r_range: Tuple[float, float]):
    rows, cols = img.shape
    center = [int(rows/2), int(cols/2)]
    mask = np.zeros((rows, cols, 2), np.int8)
    x, y = np.ogrid[:rows, :cols]

    r_in, r_out = r_range

    mask_area = np.logical_and(
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    return mask


def create_rectangular_masks(img: np.ndarray, r_masks: List[Tuple[float, float]]) -> np.ndarray:
    rows, cols = img.shape
    center = [int(rows/2), int(cols/2)]

    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]

    x0, y0 = center[0], center[1]
    for r_mask in r_masks:
        theta, delta_y = r_mask
        theta = theta*np.pi/180.0

        mask_area = np.logical_and(
            (y <= y0 + np.tan(theta)*(x - x0) + delta_y/2),
            (y >= y0 + np.tan(theta)*(x - x0) - delta_y/2)
        )
        mask[mask_area] = 1
    return mask


def fft(img: np.ndarray):
    '''Returns the magnitude spectrum of the fast Fourier transform of an image.
    '''
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def fft_filter(img: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * \
        np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    fshift_mask_mag = (fshift_mask_mag*255 /
                       fshift_mask_mag.max()).astype(np.uint8)
    img_back = (img_back*255 / img_back.max()).astype(np.uint8)
    return {
        'Input Image': img,
        'After FFT': magnitude_spectrum,
        'FFT + Mask': fshift_mask_mag,
        'After FFT Inverse': img_back
    }


def fft_circular(img: np.ndarray, r_range: Tuple[float, float]):
    image_fft = fft_filter(img, create_circular_mask(img, r_range))
    return image_fft['After FFT Inverse']


def fft_rectangular(img: np.ndarray, r_masks: List[Tuple[float, float]]):
    image_fft = fft_filter(img, create_rectangular_masks(img, r_masks))
    return image_fft['After FFT Inverse']
