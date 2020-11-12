import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import data_extraction
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

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=rectangular_masks)

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=thersh, kernel=kernel, thresh_pre=thresh_pre, dia_iter=dia_iter)

# Reducing oversegmentation
merged = oversegmentation.auto_merge(
    segmented['modified markers'], merge_thresh)
merged = oversegmentation.auto_merge(merged, merge_thresh)
removed = oversegmentation.remove_boundary(merged)

# Data extraction and saving data
data_extraction.data_extraction(removed, 'Data/data_{}_FFT'.format(name))

end = time.time()

# Print run time
print('Data generation took %g seconds to execute.' % round(end-start, 1))
