import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import oversegmentation
import watershed
import fast_Fourier_transform
import image_processing

# Setting parameters
rectangular_masks = [(-30, 50), (65, 45), (89.9, 40)]  # FFT masks

(thersh, kernel, thresh_pre, dia_iter) = (
    0.21, (5, 5), 65, 2)  # Watershed segmentation

merge_thresh = 800  # Merging threshold

# Measure run time
start = time.time()

# Load Image
image_name = 'MIPAR.png'
name = 'MIPAR'
image = cv2.imread(
    'Data/' + image_name)
image_ori = cv2.imread(
    'Data/' + image_name)

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# Visualise denoised and thresholded images
image_processing.display_image_1D(
    denoised,
    thresholded_otsu,
    cmap=[None, 'gray'],
    visualisation=True)

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=rectangular_masks)

masks = fast_Fourier_transform.create_rectangular_masks(
    thresholded_otsu, r_masks=rectangular_masks)

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

# Visualise FFT comparison image
image_processing.display_image_2D(
    fft_comparison['input image'],
    fft_comparison['after FFT'],
    fft_comparison['FFT + mask'],
    fft_comparison['after FFT inverse'],
    rows=2, cols=2,
    cmap=['gray', None, None, 'gray'],
    visualisation=True)

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=thersh, kernel=kernel, thresh_pre=thresh_pre, dia_iter=dia_iter)

# Reducing oversegmentation
unmerged = segmented['modified markers']
merged = oversegmentation.auto_merge(
    segmented['modified markers'], merge_thresh)
merged = oversegmentation.auto_merge(merged, merge_thresh)
removed = oversegmentation.remove_boundary(merged)

# Visualise segmentation results
image_processing.display_image_2D(
    fft_comparison['after FFT inverse'],
    segmented['segmented image'],
    unmerged,
    removed,
    rows=2, cols=2,
    cmap=['gray', None, 'gist_ncar', 'gist_ncar'],
    visualisation=True)

end = time.time()

# Print run time
print('Visualisation took %g seconds to execute.' % round(end-start, 1))
