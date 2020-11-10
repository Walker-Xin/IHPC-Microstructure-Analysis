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
image_name = 'IHPC.png'
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
    denoised, thresholded_otsu, cmap=[None, 'gray'], filename='denoised_n_thresholded.png')

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

masks = fast_Fourier_transform.create_rectangular_masks(
    thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

# Save FFT comparison image
image_processing.display_image_2D(
    fft_comparison['input image'],
    fft_comparison['after FFT'],
    fft_comparison['FFT + mask'],
    fft_comparison['after FFT inverse'],
    rows=2, cols=2,
    cmap=['gray', None, None, 'gray'],
    filename='FFT.png')

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=0.24, kernel=(3, 3), thresh_pre=25, dia_iter=3)

# Reducing oversegmentation
unmerged = segmented['modified markers']
merged = oversegmentation.auto_merge(segmented['modified markers'], 5000)
merged = oversegmentation.auto_merge(merged, 5000)
removed = oversegmentation.remove_boundary(merged)

# Save segmentation results
image_processing.display_image_2D(
    image_ori,
    segmented['segmented image'],
    unmerged,
    removed,
    rows=2, cols=2,
    filename='segmentation.png')

end = time.time()

# Print run time
print(end-start)
