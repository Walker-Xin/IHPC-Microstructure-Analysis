import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit

from typing import Dict, List, Optional, Tuple


def surround_1(image: np.ndarray, coord: Tuple[int, int]):
    '''Takes in an array and a pixel's coordinate. Returns an array that contains the pixel itself and its 8 surrounding pixels, in total 9.
    '''
    lis = coord[0]
    ele = coord[1]
    surround_1 = image[lis-1:lis+2, ele-1:ele+2]
    return surround_1


def surround_2(image: np.ndarray, coord: Tuple[int, int]):
    '''Takes in an array and a pixel's coordinate. Returns an array that contains the pixel itself and its 24 surrounding pixels of two layers, in total 25.
    '''
    lis = coord[0]
    ele = coord[1]
    surround_2 = image[lis-2:lis+3, ele-2:ele+3]
    return surround_2


def merge(image: np.ndarray, label_small, label_large):
    '''Takes in an labelled image represented by a numpy array. Merge the region marked by label_small with the region marked by lebel_large. Returns the image after merging.
    '''
    coordinates = np.where(image == label_small)
    coordinates = list(zip(coordinates[0], coordinates[1])) #Get coordinates of pixels of label_small

    for coord in coordinates:
        neighbours = surround_1(image, coord) #Get all label_small pixels

        if -1 in neighbours:
            positions = np.where(neighbours == -1)
            positions = list(zip(positions[0], positions[1])) #Get all boundary pixels

            for i in range(0, len(positions)): #Get the coordinates of the boundary pixels
                x, y = positions[i]
                x += -1 + coord[0]
                y += -1 + coord[1]
                positions[i] = (x, y)

            for i in range(0, len(positions)): #Mark out the boundary pixels that seperate the large region with the small region
                if label_large in surround_1(image, positions[i]):
                    pass
                else:
                    positions[i] = 1

    positions = list(filter((1).__ne__, positions)) #Remove all the marked boundary pixels
    coordinates = coordinates + positions
    coordinates = list(dict.fromkeys(coordinates)) #Combine boundary pixels with label_small pixels

    for coord in coordinates:
        image[coord] = label_large #Transform values of all pixels to label_large

    return image


def nearest_label(image, label):
    '''Retrieves the labels of the surrounding regions of an area.
    '''
    coordinates = np.where(image == label)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    unique = []
    for coord in coordinates:
        neighbours = surround_2(image, coord)
        unique += list(np.unique(neighbours))
    unique = list(dict.fromkeys(unique))

    try: #Remove unwanted labels
        unique.remove(label)
        unique.remove(-1)
        unique.remove(1)
    except:
        pass

    return unique


def area(image: np.ndarray):
    label, area = np.unique(image, return_counts=True)
    data = list(zip(label, area))
    data = data[2:]
    data = {data[i][0]: data[i][1] for i in range(0, len(data))}
    return data


def auto_merge(image, threshold):
    areas = area(image)
    labels = np.unique(image)[2:]
    for label in labels:
        neighbours = nearest_label(image, label)
        neighbours_area = []
        for neighbour in neighbours:
            neighbours_area.append(areas[neighbour])
        if areas[label] < threshold:
            if max(neighbours_area) > threshold:
                index = neighbours_area.index(max(neighbours_area))
                merge(image, label, neighbours[index])
            else:
                pass
        else:
            pass
    
    return image


def remove_boundary(image):
    removed = image

    coordinates = np.where(image == -1)
    coordinates = list(zip(coordinates[0], coordinates[1])) #Get coordinates of boundary pixels

    for coord in coordinates:
        neighbours = surround_1(image, coord)
        unique = np.unique(neighbours)
        if len(unique) == 2:
            removed[coord] = unique[1]
        else:
            pass

    return removed


image = np.load(
    r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Data extraction\Marker_IHPC.npy')

print(area(image))
print(len(area(image)))

start = timeit.default_timer()

merged = auto_merge(image, 7000)
merged = auto_merge(merged, 7000)
merged = remove_boundary(merged)

stop = timeit.default_timer()

fig, axe = plt.subplots(1, 1, figsize=(15, 15))
axe.imshow(merged)
plt.show()

print(area(merged))
print(len(area(image)))
print('Time: ', stop - start)