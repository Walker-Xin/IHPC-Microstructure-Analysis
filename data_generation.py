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
image_name = 'IHPC.png'
image = cv2.imread(
    image_name)

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=0.24, kernel=(3, 3), thresh_pre=35, dia_iter=3)

# Reducing oversegmentation
merged = oversegmentation.auto_merge(segmented['modified markers'], 6500)
merged = oversegmentation.auto_merge(merged, 6500)
removed = oversegmentation.remove_boundary(merged)

# Data extraction and saving data
data_extraction.data_extraction(removed, 'data_'+image_name)

end = time.time()

# Print run time
print('Took %g seconds to execute' %round(end-start, 1))
