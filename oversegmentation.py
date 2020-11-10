import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
    '''Takes in an labelled marker image. Merge the region marked by label_small with the region marked by lebel_large. Returns the image after merging.
    '''
    # Get coordinates of pixels of label_small
    coordinates = np.where(image == label_small)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    for coord in coordinates:
        # Transform values of all pixels to label_large
        image[coord] = label_large

    return image


def nearest_label(image, label):
    '''Takes in a labelled marker image and a label. Retrieves the unique labels in the surrounding regions of an area.
    '''
    coordinates = np.where(image == label)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    unique = []
    for coord in coordinates:
        neighbours = surround_2(image, coord)
        unique += list(np.unique(neighbours))
    unique = list(dict.fromkeys(unique))

    try:  # Remove unwanted labels
        unique.remove(label)
        unique.remove(-1)
        unique.remove(1)
    except:
        pass

    return unique


def area(image: np.ndarray):
    '''Takes in a labelled marker image. Returns a list with tuples that contain a label and its area represented by the number of pixels. 
    '''
    label, area = np.unique(
        image, return_counts=True)  # Get the numbers of pixels with each label

    data = list(zip(label, area))
    data = data[2:]  # Discard backgroud and boundary pixels

    data = {data[i][0]: data[i][1] for i in range(0, len(data))}
    return data


def auto_merge(image, threshold):
    '''Takes in a labelled marker image and a threshold. Conducts automatic merging of small regions with large regions. Selection is based on threshold. Returns the merged image. A second round of merging may be required for optimal results.
    '''
    areas = area(image)  # Get area data
    labels = np.unique(image)[2:]  # Get all labels

    for label in labels:
        if areas[label] < threshold:  # Check that the current label is small enough
            # Get surrounding labels for the label
            neighbours = nearest_label(image, label)
            neighbours_area = []  # A list containing the area data of the surrounding labels

            for neighbour in neighbours:
                neighbours_area.append(areas[neighbour])

            if len(neighbours_area) > 0 and max(neighbours_area) > threshold:
                # Get the label of the largest neighbour
                index = neighbours_area.index(max(neighbours_area))
                # Merge with the large neighbour
                merge(image, label, neighbours[index])
            else:
                pass
        else:
            pass

    return image


def remove_boundary(image):
    '''Takes in a labelled marker image. Removes excess boundary lines due to auto merging. Returns the image with excess boundary lines removed.
    '''
    removed = image

    # Get coordinates of boundary pixels
    coordinates = np.where(image == -1)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    for coord in coordinates:
        neighbours = surround_1(image, coord)
        unique = np.unique(neighbours)
        # If the number of unique surrounding labels of a boundary pixels (including the label of the pixels itself) is less than 3 (equal to 2), remove it
        if len(unique) == 2:
            removed[coord] = unique[1]
        else:
            pass

    return removed
