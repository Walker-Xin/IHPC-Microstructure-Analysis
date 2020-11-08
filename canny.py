import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple

img = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\MIPAR_image_cropped_2.png')

denoised = cv2.GaussianBlur(img, (5,5), sigmaX=5)

canny = cv2.Canny(denoised, 9, 25)

cv2.namedWindow('image1', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image1', canny)
cv2.waitKey()

cv2.imwrite('Canny_MIPAR_cropped_2.png', canny)
