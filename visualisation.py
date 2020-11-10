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

# Measure run time
start = time.time()

# Load Image
image_name = 'MIPAR_image.png'
name = 'MIPAR'
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
    filename='denoised_n_thresholded_{}.png'.format(name),
    visualisation=False)

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=[(-30, 40), (65, 35), (89.9, 40)])

masks = fast_Fourier_transform.create_rectangular_masks(
    thresholded_otsu, r_masks=[(-30, 40), (65, 35), (89.9, 40)])

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

# Save FFT comparison image
image_processing.display_image_2D(
    fft_comparison['input image'],
    fft_comparison['after FFT'],
    fft_comparison['FFT + mask'],
    fft_comparison['after FFT inverse'],
    rows=2, cols=2,
    cmap=['gray', None, None, 'gray'],
    filename='FFT_{}.png'.format(name),
    visualisation=False)

# Segmentation
segmented = watershed.watershed(
    thresholded_otsu, image, thresh=0.22, kernel=(5, 5), thresh_pre=30, dia_iter=2)

# Reducing oversegmentation
unmerged = segmented['modified markers']
merged = oversegmentation.auto_merge(segmented['modified markers'], 1000)
merged = oversegmentation.auto_merge(merged, 1000)
removed = oversegmentation.remove_boundary(merged)

# Save segmentation results
image_processing.display_image_2D(
    image_ori,
    segmented['segmented image'],
    unmerged,
    removed,
    rows=2, cols=2,
    filename='segmentation_{}_Otsu.png'.format(name),
    visualisation=False)

end = time.time()

# Print run time
print('Visualisation took %g seconds to execute.' %round(end-start, 1))
