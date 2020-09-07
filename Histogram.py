import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\blur_gaussian_bilateral\Individual\Blur\Gaussian 05.png', 0)
img = cv2.imread(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Ti6Al4V_cropped.png', 0)
img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=5)
hist = cv2.calcHist([img], [0], None, [256], [0,256])
hist_norm = hist.ravel()/hist.sum()
plt.plot(hist_norm)
plt.show()