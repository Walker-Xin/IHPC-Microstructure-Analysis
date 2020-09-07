import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple

os.chdir(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC')

img = cv.imread('watershed_coins_01.jpg')

assert type(img) == np.ndarray
#assert len(img.shape) == 2

img = cv.GaussianBlur(img, (9, 9), 9)

ret, thresh = cv.threshold(img, 80, 255, cv.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel, iterations = 2)

sure_bg = cv.dilate(thresh,kernel,iterations=3)

dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 3)
#cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
ret, sure_fg = cv.threshold(dist_transform, 0.01*dist_transform.max(), 255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv.namedWindow('a', cv.WINDOW_NORMAL)
cv.imshow('a', img)
cv.waitKey()

#cv.imwrite('coins_self_final.jpg', img)'''
