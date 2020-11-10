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

# Measure run time
start = time.time()

# Load Image
image_name = 'MIPAR_image.png'
name = 'MIPAR'
image = cv2.imread(
    image_name)

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=[(-30, 40), (65, 35), (89.9, 40)])

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=0.24, kernel=(5, 5), thresh_pre=55, dia_iter=2)

# Reducing oversegmentation
merged = oversegmentation.auto_merge(segmented['modified markers'], 1500)
merged = oversegmentation.auto_merge(merged, 1500)
removed = oversegmentation.remove_boundary(merged)

# Data extraction and saving data
data_extraction.data_extraction(removed, 'data_{}_FFT'.format(name))

end = time.time()

# Print run time
print('Data generation took %g seconds to execute' %round(end-start, 1))
