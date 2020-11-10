import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple


def create_circular_mask(img: np.ndarray, r_range: Tuple[float, float]):
    '''Create a circular mask with the origin at the centre of the image. r_range indicates the inner and outer radius of the mask.
    '''
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
    '''Create a rectangular mask using the tangent function. 
    The first variable of r_masks dictates the angle of the rectangle, measured anti-clockwise from the positive y-axis.
    The second variable of r_masks dictates the width of the rectangle.
    '''
    rows, cols = img.shape
    center = [int(rows/2), int(cols/2)]

    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]

    x0, y0 = center[0], center[1]
    for r_mask in r_masks:
        theta, width = r_mask
        theta = theta*np.pi/180.0
        delta_y = width/np.cos(theta)

        mask_area = np.logical_and(
            (y <= y0 + np.tan(theta)*(x - x0) + delta_y/2),
            (y >= y0 + np.tan(theta)*(x - x0) - delta_y/2)
        )
        mask[mask_area] = 1
    return mask


def fft(img: np.ndarray):
    '''Return the magnitude spectrum of the fast Fourier transform of an image.
    '''
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def fft_filter(img: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    '''Generate the FFT transform of an image, apply a mask on it and generate the reverse FFT transform of the masked frequency image.
    The input image, frequency image, masked frequency image and the inverse transform are stored in a dictionary.
    Note that the low frequency parts are shifted to the centre.
    '''
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
        'input image': img,
        'after FFT': magnitude_spectrum,
        'FFT + mask': fshift_mask_mag,
        'after FFT inverse': img_back
    }


def fft_circular(img: np.ndarray, r_range: Tuple[float, float]):
    '''Conduct FFT transform and frequency domain operations on an image with a circular mask.
    Returns the FFT inverse of the masked frequency image.
    '''
    image_fft = fft_filter(img, create_circular_mask(img, r_range))
    return image_fft['after FFT inverse']


def fft_rectangular(img: np.ndarray, r_masks: List[Tuple[float, float]]):
    '''Conduct FFT transform and frequency domain operations on an image with a rectangular mask.
    Returns the FFT inverse of the masked frequency image.
    '''
    image_fft = fft_filter(img, create_rectangular_masks(img, r_masks))
    return image_fft['after FFT inverse']
