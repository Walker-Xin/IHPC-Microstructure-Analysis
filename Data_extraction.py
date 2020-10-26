import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Tuple


def surround(array: np.ndarray, coord: Tuple[int, int]):
    '''Takes in an array and a pixel's coordinate. Returns an array that contains the pixel itself and its eight surrounding pixels, in total nine.
    '''
    lis = coord[0]
    ele = coord[1]
    surround = array[lis-1:lis+2, ele-1:ele+2]
    return surround


def area_circumference(image: np.ndarray):
    '''Takes in a labelled marker image. Returns a dictionary that contains the area, circumference and area/circumference ratio of each grain.
    '''
    blank = np.zeros(
        image.shape, np.uint8)  # A blank image of the same dimension as the mock image. Used for labelling
    labels = np.unique(image)[2:]
    # Get a certain label (positive integer that represent one segmented region)
    for label_no in labels:
        result = np.where(image == label_no)  # Exclude other labels
        # Obtain the coordinates of each label
        coordinates = list(zip(result[0], result[1]))
        for coord in coordinates:
            neighbours = surround(image, coord)
            # Test if the surrounding pixels contain 0, the background.
            if -1 in neighbours:
                # If so, put a mark on the blank label image
                blank[coord] = label_no

    region, circumference = np.unique(blank, return_counts=True)
    circumference = circumference[1:]
    region, area = np.unique(image, return_counts=True)
    region = region[2:]
    area = area[2:]
    ratio = area/circumference
    data = list(zip(region, area, circumference, ratio))

    data_d = {}
    for i in range(0, len(data)):
        data_d[i+1] = data[i]
    return data_d


def width_length_ellipse(image: np.ndarray, label):
    '''Takes in a labelled marker image and a specific label. Returns the minor- and major-axis length of an ellipse that will fit the grain with that label.
    '''
    blank = np.zeros(
        image.shape, np.uint8)  # A blank image of the same dimension as the mock image. Used for extraction grain pixels of one label.
    result = np.where(image == label)  # Obtain the coordinates of each label

    coordinates = list(zip(result[0], result[1]))
    for coord in coordinates:
        blank[coord] = 1
    contours, hierarchy = cv2.findContours(
        blank, cv2.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)  # Get data about the geometry of the ellipse
    blank = cv2.ellipse(blank, ellipse, 1, 2)  # Visualise
    return ellipse[1], blank


def width_length_rectangle(image: np.ndarray, label):
    blank = np.zeros(
        image.shape, np.uint8)  # A blank image of the same dimension as the mock image. Used for extraction grain pixels of one label.
    result = np.where(image == label)  # Obtain the coordinates of each label

    coordinates = list(zip(result[0], result[1]))
    for coord in coordinates:
        blank[coord] = 1
    contours, hierarchy = cv2.findContours(blank, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    blank = cv2.drawContours(blank, [box], 0, (0, 0, 255), 2)
    return rect[1], blank



    coordinates = np.where(image == label_small)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    for coord in coordinates:
        neighbours = surround_1(image, coord)
        if -1 in neighbours:
            positions = np.where(neighbours == -1)
            positions = list(zip(positions[0], positions[1]))

            for i in range(0, len(positions)):
                x, y = positions[i]
                x += -1 + coord[0]
                y += -1 + coord[1]
                positions[i] = (x, y)

            for i in range(0, len(positions)):
                if label_large in surround_1(image, positions[i]):
                    pass
                else:
                    positions[i] = 1

            positions = list(filter((1).__ne__, positions))
            coordinates = coordinates + positions
            coordinates = list(dict.fromkeys(coordinates))

    for coord in coordinates:
        image[coord] = label_large

    return image


# Loading actual marker image
image = np.load(
    r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Data extraction\Marker_IHPC.npy')

fig, axe = plt.subplots(1, 1, figsize=(15,15))
axe.imshow(image)
plt.show()

print(area_circumference(image))

for i in range(2, image.max()+1):
    try:
        print(i)
        print(width_length_rectangle(image, i)[0])
    except:
        pass