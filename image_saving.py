import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import oversegmentation
import watershed
import fast_Fourier_transform
import image_processing

os.chdir('Data')

# Setting parameters
rectangular_masks = [(-52, 60), (75, 45), (89.9, 30), (60, 25)]  # FFT masks

(thersh, kernel, thresh_pre, dia_iter) = (0.24, (3, 3), 25, 3)  # Watershed segmentation

merge_thresh = 6500

# Measure run time
start = time.time()

# Load Image
image_name = 'IHPC.png'
name = 'IHPC'
image = cv2.imread(
    image_name)
image_ori = cv2.imread(
    image_name)

os.chdir('Pics')

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# Save denoised and thresholded images
image_processing.display_image_1D(
    denoised,
    thresholded_otsu,
    cmap=[None, 'gray'],
    filename='denoised_n_thresholded_{}.png'.format(name))

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=rectangular_masks)

masks = fast_Fourier_transform.create_rectangular_masks(
    thresholded_otsu, r_masks=rectangular_masks)

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

# Save FFT comparison image
image_processing.display_image_2D(
    fft_comparison['input image'],
    fft_comparison['after FFT'],
    fft_comparison['FFT + mask'],
    fft_comparison['after FFT inverse'],
    rows=2, cols=2,
    cmap=['gray', None, None, 'gray'],
    filename='FFT_{}.png'.format(name))

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=thersh, kernel=kernel, thresh_pre=thresh_pre, dia_iter=dia_iter)

# Reducing oversegmentation
unmerged = segmented['modified markers']
merged = oversegmentation.auto_merge(
    segmented['modified markers'], merge_thresh)
merged = oversegmentation.auto_merge(merged, merge_thresh)
removed = oversegmentation.remove_boundary(merged)

# Save segmentation results
image_processing.display_image_2D(
    image_ori,
    segmented['segmented image'],
    unmerged,
    removed,
    rows=2, cols=2,
    filename='segmentation_{}.png'.format(name))

end = time.time()

# Print run time
print('Visualisation took %g seconds to execute.' % round(end-start, 1))