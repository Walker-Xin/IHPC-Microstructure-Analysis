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


def FFT_circular(img: np.ndarray, r_range: Tuple[float, float]):
    image_fft = fft_filter(img, create_circular_mask(img, r_range))
    return image_fft['After FFT Inverse']


def FFT_rectangular(img: np.ndarray, r_masks: List[Tuple[float, float]]):
    image_fft = fft_filter(img, create_rectangular_masks(img, r_masks))
    return image_fft['After FFT Inverse']


def display_image_2D(image_dict: Dict[str, np.ndarray], rows: int, cols: int, figsize: Tuple[int, int] = (10, 7), filename: Optional[str] = None):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    row, col = 0, 0
    for title, image in image_dict.items():
        axs[row][col].imshow(image, cmap='gray', interpolation='none')
        axs[row][col].set_xticks([])
        axs[row][col].set_yticks([])
        axs[row][col].set_title(title)
        col += 1
        if col == cols:
            row += 1
            col = 0
        if row*cols + col > len(image_dict):
            print('Not enough rows & columns')
            print(f'n = {len(image_dict)} > rows={rows} x cols={cols}')

    if filename:
        plt.savefig(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename)


def save_images(image_dict: Dict[str, np.ndarray]):
    filtered_dict = image_dict.copy()
    filtered_dict.pop('Original', None)
    for label, image in filtered_dict.items():
        filename = f'{label}.png'
        cv2.imwrite(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename, image)
        print(f"Saved: '{filename}'")

img = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\blur_gaussian_bilateral\Individual\Otsu\Otsu on Gaussian 05.png', 0)

fft = fft_filter(img, create_rectangular_masks(img, r_masks=[(75, 200), (-53, 40), (89, 1000)]))

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', fft['After FFT Inverse'])
cv2.waitKey()

#cv2.imwrite('slines_noise_FFT_inverse.png', fft['After FFT Inverse'])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(fft['After FFT'])
ax2.imshow(fft['FFT + Mask'])
plt.show()

plt.imshow(fft['FFT + Mask'])
plt.savefig('otsu_FFT_mask.png', dpi=300)
plt.show()

