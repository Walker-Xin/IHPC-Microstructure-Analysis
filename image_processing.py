import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple


# Save images


def save_images(image_dict: Dict[str, np.ndarray]):
    '''Save all images in a dictionary.
    '''
    filtered_dict = image_dict.copy()
    filtered_dict.pop('Original', None)
    for label, image in filtered_dict.items():
        filename = f'{label}.png'
        cv2.imwrite(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename, image)
        print(f"Saved: '{filename}'")

# Image display


def display_image(img: np.ndarray, figsize: Tuple[int, int] = (10, 7), interpolation: str = 'none', filename: Optional[str] = None):
    '''Display an image. 
    If filename is provied, the image is saved with that filename.
    '''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img, cmap='gray', interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename:
        plt.savefig(
            r'C:/Users/Xin Wenkang/Documents/Scripts/IPHC/Pics/' + filename)


def display_image_1D(image_dict: Dict[str, np.ndarray], orientation: str, figsize: Tuple[int, int] = (10, 7), filename: Optional[str] = None):
    '''Display images in a dictionary in the horizontal or vertical direction. 
    If filename is provided, the images are saved in one single image file.
    '''
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
    '''Display images in a dictionary in a matrix. 
    If filename is provided, the images are saved in one single image file.
    '''
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
    '''Display an image with Matplotlib
    '''
    assert type(img) == np.ndarray
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey()

# Algorithms


def denoise(img: np.ndarray, method: str = 'blur', **kwargs):
    '''Conduct denoising on an image. 
    The method string dictates the denoising techinique used. 
    Keyword arguments are the ones used by that parcicular denoising technique.
    '''
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
    '''Conduct edge dectection on an image. 
    The method string dictates the edge detection techinique used.
    '''
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
    '''Conduct edge detection on a dictionary of images.
    '''
    new_img_dict = {}
    method_str = method.title()
    for label, img in image_dict.items():
        new_label = f'{method_str} on {label}'
        new_img_dict[new_label] = edge_detect(image_dict[label], method=method)
    return new_img_dict


def threshold(img: np.ndarray, method: str = 'adaptive_gaussian'):
    '''Conduct thresholding on an image. 
    The method string dictates the thresholding techinique used, either adaptive Gaussian or Otsu.
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'adaptive_gaussian':
        img_threshold = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        thresh, img_threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_threshold


def threshold_multi(image_dict: Dict[str, np.ndarray], method: str):
    '''Conduct thresholding on a dictionary of images.
    '''
    new_img_dict = {}
    method_str = method.title()
    for label, img in image_dict.items():
        new_label = f'{method_str} on {label}'
        new_img_dict[new_label] = threshold_image(
            image_dict[label], method=method)
    return new_img_dict
