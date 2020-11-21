import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple


# Save images


def save_images(image_dict: Dict[str, np.ndarray]):
    '''Save all images in a dictionary, which contains the images and their corresponding names.
    '''
    filtered_dict = image_dict.copy()
    filtered_dict.pop('Original', None)
    for label, image in filtered_dict.items():
        filename = f'{label}.png'
        cv2.imwrite(filename, image)
        print(f"Saved: '{filename}'")


# Image display


def display_image(data: Tuple[np.ndarray, str], figsize: Tuple[int, int] = (18, 18), cmap: Optional = None, filename: Optional[str] = None, visualisation = False, tight = False):
    '''Display an image. 
    If cmap string is provided, the image is displayed with that colourmap.
    If filename is provied, the image is saved with that filename.
    '''
    (image, name) = data
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image, cmap=cmap)
    if name:
        ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename:
        if tight:
            plt.savefig(filename, bbox_inches = 'tight')
        else:
            plt.savefig(filename)
        print('Image saved. Filename: ' + filename)

    if visualisation:
        plt.show()
    plt.close()


def display_image_1D(data_set: List, figsize: Tuple[int, int] = (18, 18), orientation='horizontal', cmap: Optional[list] = None, filename: Optional[str] = None, visualisation = False):
    '''Display multiple images in one direction.
    Input images are stored in a list with tuples containting the images and their titles.
    If cmap list is provided, the images are displayed with their corresponding colourmaps.
    If filename is provided, the images are saved in one single image file.
    '''
    n = len(data_set)
    i = 0
    if orientation == 'horizontal':
        fig, axs = plt.subplots(1, n, figsize=figsize)
    elif orientation == 'vertical':
        fig, axs = plt.subplots(n, 1, figsize=figsize)

    if cmap:
        assert len(data_set) == len(
            cmap), 'Number of colour maps should match number of images'
        for (image, name) in data_set:
            axs[i].imshow(image, cmap=cmap[i])
            axs[i].set_title(name)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

            i += 1
    else:
        for (image, name) in data_set:
            axs[i].imshow(image)
            axs[i].set_title(name)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

            i += 1

    if filename:
        plt.savefig(filename)
        print('Image saved. Filename: ' + filename)
        
    if visualisation:
        plt.show()
    plt.close()


def display_image_2D(data_set: List, rows: int, cols: int, figsize: Tuple[int, int] = (18, 18), cmap: Optional[list] = None, filename: Optional[str] = None, visualisation = False):
    '''Display multiple images in a matrix. 
    Input images are stored in a list with tuples containting the images and their titles.
    If cmap list is provided, the images are displayed with their corresponding colourmaps.
    If filename is provided, the images are saved in one single image file.
    '''
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    row, col, i = 0, 0, 0

    if cmap:
        assert len(data_set) == len(
            cmap), 'Number of colour maps should match number of images'
        assert len(data_set) == rows * \
            cols, 'Product of rows and cols should match numebr of images'
        for (image, name) in data_set:
            axs[row][col].imshow(image, cmap=cmap[i])
            axs[row][col].set_title(name)
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

            i += 1
            col += 1
            if col == cols:
                row += 1
                col = 0
    else:
        assert len(images) == rows * \
            cols, 'Product of rows and cols should match numebr of images'
        for (image, name) in data_set:
            axs[row][col].imshow(image)
            axs[row][col].set_title(name)
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

            col += 1
            if col == cols:
                row += 1
                col = 0

    if filename:
        plt.savefig(filename)
        print('Image saved. Filename: ' + filename)
        
    if visualisation:
        plt.show()
    plt.close()


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


def histogram(image: np.ndarray, figsize: Tuple[int, int] = (18, 18), filename: Optional[str] = None, visualisation = False):
    '''Calculates and visualises the intensity histogram of an image.
    '''
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist)
    
    if filename:
        plt.savefig(filename, bbox_inches = 'tight')
        print('Image saved. Filename: ' + filename)
        
    if visualisation:
        plt.show()
    plt.close()