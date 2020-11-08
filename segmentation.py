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

os.chdir(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Code')

# Measure run time
start = time.time()

# Load Image
image = cv2.imread(
    r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Ti6Al4V.png')

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

# Uncomment to visualise FFT images
'''masks = fast_Fourier_transform.create_rectangular_masks(thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

fig, axs = plt.subplots(2, 2, figsize=(30,30))
axs[0,0].imshow(fft['input image'], cmap='gray')
axs[0,1].imshow(fft['after FFT'])
axs[1,0].imshow(fft['FFT + mask'])
axs[1,1].imshow(fft['after FFT inverse'], cmap='gray')
plt.show()'''

# Segmentation
segmented = watershed.watershed(
    fft, image, thresh=0.23, kernel=(3, 3), thresh_pre=30, dia_iter=3)

# Reducing oversegmentation
merged = oversegmentation.auto_merge(segmented['modified markers'], 7000)
merged = oversegmentation.auto_merge(merged, 7000)
removed = oversegmentation.remove_boundary(merged)

# Uncomment to visualise watershed segmentation results
'''fig, axs = plt.subplots(1, 1, figsize=(30,30))
axs.imshow(removed)
plt.show()'''

# Data extraction and saving data
data_extraction.data_extraction(removed, 'data_test')

end = time.time()

# Print run time
print(end-start)
