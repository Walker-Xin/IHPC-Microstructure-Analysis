import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy

import data_extraction
import oversegmentation
import watershed
import fast_Fourier_transform
import image_processing

# Set image files and segmentation method
if len(sys.argv) > 1:
    image_name = sys.argv[1]
    assert isinstance(image_name, str), 'Incorrect first variable type passed!'
    seg_method = sys.argv[2]
    assert isinstance(seg_method, str), 'Incorrect second variable type passed!'
else:
    image_name = 'IHPC'
    seg_method = 'otsu'

print('Segmenting {} image based on {} method...'.format(image_name, seg_method))

# Set parameters and load image acoording to image_name
if image_name == 'IHPC':
    image = cv2.imread(
        'Data/' + image_name + '.png')
    image_ori = cv2.imread(
        'Data/' + image_name + '.png')
    
    rectangular_masks = [(-52, 60), (75, 45), (89.9, 30),
                            (60, 25)]  # FFT masks

    if seg_method == 'FFT':
        (thersh, kernel, thresh_pre, dia_iter) = (
            0.24, (3, 3), 25, 3)  # Watershed segmentation

        merge_thresh = 6500  # Merging threshold
    elif seg_method == 'otsu':
        (thersh, kernel, thresh_pre, dia_iter) = (
            0.20, (3, 3), 30, 3)  # Watershed segmentation

        merge_thresh = 6000  # Merging threshold
    else:
        raise ValueError('Incorret seg_method!')
elif image_name == 'MIPAR':
    image = cv2.imread(
        'Data/' + image_name + '.png')
    image_ori = cv2.imread(
        'Data/' + image_name + '.png')

    rectangular_masks = [(-30, 50), (65, 45), (89.9, 40)]  # FFT masks

    if seg_method == 'FFT':
        (thersh, kernel, thresh_pre, dia_iter) = (
            0.21, (5, 5), 65, 2)  # Watershed segmentation

        merge_thresh = 800  # Merging threshold

    elif seg_method == 'otsu':
        (thersh, kernel, thresh_pre, dia_iter) = (
            0.22, (5, 5), 30, 2)  # Watershed segmentation

        merge_thresh = 1000  # Merging threshold
    else:
        raise ValueError('Incorret seg_method!')
else:
    raise ValueError('Incorret image name!')

# Measure run time
start = time.time()

# Denoisng
denoised = image_processing.denoise(
    image, method='gaussian', ksize=(5, 5), sigmaX=5)

# Thresholding
thresholded_otsu = image_processing.threshold(denoised, method='Otsu')

# Visualise denoised and thresholded images
image_processing.display_image_1D(
    [(denoised, 'Denoised Image'),
    (thresholded_otsu, 'Thresholded Image')],
    cmap=[None, 'gray'],
    visualisation=True)

# FFT images
fft = fast_Fourier_transform.fft_rectangular(
    thresholded_otsu, r_masks=rectangular_masks)

masks = fast_Fourier_transform.create_rectangular_masks(
    thresholded_otsu, r_masks=rectangular_masks)

fft_comparison = fast_Fourier_transform.fft_filter(thresholded_otsu, masks)

# Visualise FFT comparison image
image_processing.display_image_2D(
    [(fft_comparison['input image'], 'Input Image'),
    (fft_comparison['after FFT'], 'Frequency Domain'),
    (fft_comparison['FFT + mask'], 'Masked Frequency Domain'),
    (fft_comparison['after FFT inverse'], 'Output Image')],
    rows=2, cols=2,
    cmap=['gray', None, None, 'gray'],
    visualisation=True)

# Segmentation
if seg_method == 'FFT':
    segmented = watershed.watershed(
        fft, image, thresh=thersh, kernel=kernel, thresh_pre=thresh_pre, dia_iter=dia_iter)
else:
    segmented = watershed.watershed(
        thresholded_otsu, image, thresh=thersh, kernel=kernel, thresh_pre=thresh_pre, dia_iter=dia_iter)

# Reducing oversegmentation
unmerged = copy.deepcopy(segmented['modified markers'])
merged = oversegmentation.auto_merge(
    segmented['modified markers'], merge_thresh)
merged = oversegmentation.auto_merge(merged, merge_thresh)
removed = oversegmentation.remove_boundary(merged)

# Visualise circumference image
circum = data_extraction.circumference_visualise(removed)
image_processing.display_image(
    (circum, 'Circumferecne Illustration'),
    cmap='gist_ncar',
    visualisation=True)

# Visualise segmentation results
image_processing.display_image_2D(
    [(image_ori, 'Original Image'),
    (segmented['segmented image'], 'Segmented Original Image'),
    (unmerged, 'Original Marker Image'),
    (removed, 'Merged Marker Image')],
    rows=2, cols=2,
    cmap=[None, None, 'gist_ncar', 'gist_ncar'],
    visualisation=True)

end = time.time()

# Print run time
print('Visualisation took {} seconds to execute.'.format(round(end-start, 1)))
