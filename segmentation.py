import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
imageimport time

os.chdir(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Code')

start = time.time()

image = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Ti6Al4V.png')

import image_processing
import fast_Fourier_transform
import watershed
import oversegmentation
import data_extraction

denoised = image_processing.denoise(image, method='gaussian', ksize=(5, 5), sigmaX=5)

thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

fft = fast_Fourier_transform.fft_rectangular(thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

'''masks = fast_Fourier_transform.create_rectangular_masks(thresholded_otsu, r_masks=[(-52, 60), (75, 160), (89, 2000), (60, 80)])

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

fig, axs = plt.subplots(2, 2, figsize=(30,30))
axs[0,0].imshow(fft['Input Image'], cmap='gray')
axs[0,1].imshow(fft['After FFT'])
axs[1,0].imshow(fft['FFT + Mask'])
axs[1,1].imshow(fft['After FFT Inverse'], cmap='gray')
plt.show()'''

segmented = watershed.watershed(fft, image, thresh=0.23, kernel=(3,3), thresh_pre=30, dia_iter=3)

merged = oversegmentation.auto_merge(segmented[0], 7000)
merged = oversegmentation.auto_merge(merged, 7000)
removed = oversegmentation.remove_boundary(merged)

data_extraction.data_extraction(removed, 'data_test')

end = time.time()

print(end-start)
