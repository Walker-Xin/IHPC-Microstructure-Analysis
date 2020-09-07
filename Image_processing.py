import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple

os.chdir(r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics')

# Save images


def save_images(image_dict: Dict[str, np.ndarray]):
    filtered_dict = image_dict.copy()
    filtered_dict.pop('Original', None)
    for label, image in filtered_dict.items():
        filename = f'{label}.png'
        cv2.imwrite(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename, image)
        print(f"Saved: '{filename}'")

# Image display


def display_image(img: np.ndarray, figsize: Tuple[int, int] = (10, 7), interpolation: str = 'none', filename: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img, cmap='gray', interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename:
        plt.savefig(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename)


def display_image_1D(image_dict: Dict[str, np.ndarray], orientation: str, figsize: Tuple[int, int] = (10, 7), filename: Optional[str] = None):
    n = len(image_dict)
    if orientation == 'horizontal':
        fig, axs = plt.subplots(1, n, figsize=figsize)
    elif orientation == 'vertical':
        fig, axs = plt.subplots(n, 1, figsize=figsize)

    for i, (title, image) in enumerate(image_dict.items()):
        axs[i].imshow(image, cmap='gray', interpolation='none')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(title)

    if filename:
        plt.savefig(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename)


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


def display_image_single(img):
    assert type(img) == np.ndarray
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey()

# Algorithms


def denoise(img: np.ndarray, method: str = 'blur', **kwargs):
    if method == 'blur':
        img_denoised = cv2.blur(img, **kwargs)
    elif method == 'gaussian':
        img_denoised = cv2.GaussianBlur(img, **kwargs)
    elif method == 'median':
        img_denoised = cv2.medianBlur(img, **kwargs)
    elif method == 'fastNlMeans':
        img_denoised = cv2.fastNlMeansDenoising(img, **kwargs)
    elif method == 'bilateral':
        img_denoised = cv2.bilateralFilter(img, **kwargs)
    return img_denoised


def edge_detect(img: np.ndarray, method: str):
    if method == 'canny':
        edges = cv2.Canny(img, 40, 100, 3)
    elif method == 'sobel16':
        sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    elif method == 'sobel':
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    else:
        print(r'Invalid method = {method}')
        quit()

    if method in ['sobel', 'sobel16']:
        edges = np.abs(sobelx) + np.abs(sobely)
        print(f'New edges: range=({edges.min()}, {edges.max()})')
        edges_scaled = (edges*255/edges.max()).astype(np.uint8)
        return edges_scaled
    return edges


def edge_detect_multi(image_dict: Dict[str, np.ndarray], method: str):
    new_img_dict = {}
    method_str = method.title()
    for label, img in image_dict.items():
        new_label = f'{method_str} on {label}'
        new_img_dict[new_label] = edge_detect(image_dict[label], method=method)
    return new_img_dict


def threshold_image(img: np.ndarray, method: str = 'adaptive_gaussian'):
    if method == 'adaptive_gaussian':
        img_threshold = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        thresh, img_threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_threshold


def threshold_multi(image_dict: Dict[str, np.ndarray], method: str):
    new_img_dict = {}
    method_str = method.title()
    for label, img in image_dict.items():
        new_label = f'{method_str} on {label}'
        new_img_dict[new_label] = threshold_image(
            image_dict[label], method=method)
    return new_img_dict


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


def FFT_multi(image_dict: Dict[str, np.ndarray], method: str, r_range=(0, 30), r_masks=[(-52, 200), (75, 800)]):
    new_img_dict = {}
    method_str = method.title()
    if method == 'Circular mask':
        for label, img in image_dict.items():
            new_label = f'{method_str} on {label}'
            new_img_dict[new_label] = FFT_circular(image_dict[label], r_range)
        return new_img_dict
    elif method == 'Rectangular mask':
        for label, img in image_dict.items():
            new_label = f'{method_str} on {label}'
            new_img_dict[new_label] = FFT_rectangular(
                image_dict[label], r_masks)
        return new_img_dict
    else:
        raise Exception(
            'Invalid mask. Either [Circular mask] or [Rectangular mask] is allowed')


def watershed(img: np.ndarray, img_ori: np.ndarray, thresh_pre=30, dia_iter=1, thresh_dist=0.01, kernel=(7, 7)):
    ''' Excute watershed tansform on the original image (img_ori must be a 3-channel image).
    Need to provide a processed image to generate markers.
    Value of kernel and thresh must be adjusted according to the processed image.
    '''
    kernel = np.ones(kernel, np.uint8)

    _, img = cv2.threshold(img, thresh_pre, 255, 0)

    img = cv2.bitwise_not(img)

    sure_bg = cv2.dilate(img, kernel, iterations=dia_iter)

    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, thresh_dist*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_ori, markers)

    img_ori[markers == -1] = [0, 0, 255]

    return img_ori


def watershed_multi(image_dict: Dict[str, np.ndarray], img_ori: np.ndarray, thresh_pre=30, dia_iter=1, thresh_dist=0.01, kernel=(7, 7)):
    new_img_dict = {}
    for label, img in image_dict.items():
        new_label = f'Watershed on {label}'
        new_img_dict[new_label] = watershed(
            image_dict[label], img_ori, thresh_pre=thresh_pre, dia_iter=dia_iter, thresh_dist=thresh_dist, kernel=kernel)
    return new_img_dict


# Load image
img = cv2.imread('Ti6Al4V_cropped_2.png', 0)
img_ori = cv2.imread('Ti6Al4V_cropped_2.png')

assert type(img) == np.ndarray
assert len(img.shape) == 2

# Bluring dic for comparison across mutilple methods
'''img_blur_d = {
    'Blur 05': denoise(img, 'blur', ksize=(5, 5)),
    'Blur 10': denoise(img, 'blur', ksize=(10, 10)),
    'Blur 15': denoise(img, 'blur', ksize=(15, 15)),
    'Gaussian 05': denoise(img, 'gaussian', ksize=(5, 5), sigmaX=5),
    'Gaussian 09': denoise(img, 'gaussian', ksize=(9, 9), sigmaX=9),
    'Gaussian 15': denoise(img, 'gaussian', ksize=(15, 15), sigmaX=15),
    'Median 07': denoise(img, 'median', ksize=7),
    'Median 11': denoise(img, 'median', ksize=11),
    'Median 15': denoise(img, 'median', ksize=15),
    'NLMeans 05': denoise(img, 'fastNlMeans', h=5, templateWindowSize=7, searchWindowSize=21),
    'NLMeans 09': denoise(img, 'fastNlMeans', h=9, templateWindowSize=7, searchWindowSize=21),
    'NLMeans 15': denoise(img, 'fastNlMeans', h=15, templateWindowSize=7, searchWindowSize=21),
    'Bilateral 07': denoise(img, 'bilateral', d=7, sigmaColor=75, sigmaSpace=75),
    'Bilateral 11': denoise(img, 'bilateral', d=11, sigmaColor=75, sigmaSpace=75),
    'Bilateral 15': denoise(img, 'bilateral', d=15, sigmaColor=75, sigmaSpace=75)
    }'''

# Bluring dic
img_blur_d = {
    'Blur 05': denoise(img, 'blur', ksize=(5, 5)),
    'Blur 10': denoise(img, 'blur', ksize=(10, 10)),
    'Blur 15': denoise(img, 'blur', ksize=(15, 15)),
    'Blur 20': denoise(img, 'blur', ksize=(20, 20)),
    'Blur 25': denoise(img, 'blur', ksize=(25, 25)),
    'Gaussian 03': denoise(img, 'gaussian', ksize=(3, 3), sigmaX=3),
    'Gaussian 05': denoise(img, 'gaussian', ksize=(5, 5), sigmaX=5),
    'Gaussian 09': denoise(img, 'gaussian', ksize=(9, 9), sigmaX=9),
    'Gaussian 11': denoise(img, 'gaussian', ksize=(11, 11), sigmaX=11),
    'Gaussian 13': denoise(img, 'gaussian', ksize=(13, 13), sigmaX=13),
    'Bilateral 07': denoise(img, 'bilateral', d=7, sigmaColor=75, sigmaSpace=75),
    'Bilateral 09': denoise(img, 'bilateral', d=9, sigmaColor=75, sigmaSpace=75),
    'Bilateral 11': denoise(img, 'bilateral', d=11, sigmaColor=75, sigmaSpace=75),
    'Bilateral 13': denoise(img, 'bilateral', d=13, sigmaColor=75, sigmaSpace=75),
    'Bilateral 15': denoise(img, 'bilateral', d=15, sigmaColor=75, sigmaSpace=75),
}

# Edge detection
img_edge_sobel_d = edge_detect_multi(img_blur_d, 'sobel')
img_edge_sobel16_d = edge_detect_multi(img_blur_d, 'sobel16')
img_edge_canny_d = edge_detect_multi(img_blur_d, 'canny')

# Thresholding
img_threshold_otsu_d = threshold_multi(img_blur_d, method='otsu')
img_threshold_gaussian_d = threshold_multi(
    img_blur_d, method='adaptive_gaussian')

# Thresholding after edge detection
img_otsu_on_sobel_d = threshold_multi(img_edge_sobel_d, method='otsu')
img_otsu_on_canny_d = threshold_multi(img_edge_canny_d, method='otsu')

# Fast Fourier transform with circular mask
'''img_cir_low_otsu_d = FFT_multi(
    img_threshold_otsu_d, 'Circular mask', r_range=(0, 80))
img_cir_high_otsu_d = FFT_multi(
    img_threshold_otsu_d, 'Circular mask', r_range=(230, 10000))'''

# Fast Fourier transform with rectangular mask
img_rec_otsu_d = FFT_multi(
    img_threshold_otsu_d, 'Rectangular mask', r_masks=[(-30, 40), (60, 40), (80, 100)])
img_rec_sobel_d = FFT_multi(
    img_edge_sobel_d, 'Rectangular mask', r_masks=[(0, 40), (80, 40)])
img_rec_otsu_on_sobel_d = FFT_multi(
    img_otsu_on_sobel_d, 'Rectangular mask', r_masks=[(0, 40), (80, 40)])

# Saving images
save_images(img_rec_otsu_d)
