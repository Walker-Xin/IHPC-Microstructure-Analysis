# Microstructure Image Segmentation with Computer Vision Analysis

## NYJC Scientist-in-School with A*STAR

This is a repository hosting algorithms and codes developed for a student project focusing on segmentation techniques used for titanium alloy microstructure images.

## Algorithms

We make use of Python and OpenCV to conduct image analysis. Several algorithms have also been developed to conduct subsequent data extraction.

### image_processing.py

This script entails most of the pre-segmentation processing techniques, including blurring, thresholding and edge detection The display_image and save_images functions allow for batch saving and comparing of the processed images.

### fast_Fourier_Transform.py

This script is dedicated for FFT and frequency domain operations. Testing of mask size and direction is important to the success of FFT.

### watershed</span>.py

This script includes the function that conducts watershed transform.

### oversegmentation</span>.py

This script contains the algorithm developed for reducing oversegmentation.
Labelled regions with an area below the threshold are merged to their surrounding ones.

### data_extraction.py

This script contains the algorithm used for data extraction.
It also uses the openpyxl package to export the data to a Microsoft Excel workbook.

### segmentation.py

This is the all-in-one script that executes all the image processing steps on a microstructure image.
