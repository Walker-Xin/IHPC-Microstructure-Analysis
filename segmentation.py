import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Code')

image = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Ti6Al4V.png')

import image_processing
import fast_Fourier_transform
import watershed

denoised = Image_processing.denoise(image, method='gaussian', ksize=(5, 5), sigmaX=5)

thresholded_otsu = image_processing.denoise(denoised, method='Otsu')

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', denoised)
cv2.waitKey()