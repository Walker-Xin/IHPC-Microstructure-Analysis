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
merged = oversegmentation.auto_merge(segmented['modified markers'], merge_thresh)
merged = oversegmentation.auto_merge(merged, merge_thresh)
removed = oversegmentation.remove_boundary(merged)

# Data extraction and saving data
data_extraction.data_extraction(removed, 'data_{}'.format(name))

end = time.time()

# Print run time
print('Data generation took %g seconds to execute.' %round(end-start, 1))
