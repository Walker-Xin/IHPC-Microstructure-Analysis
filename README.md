# Microstructure Image Segmentation with Computer Vision Analysis

## NYJC Scientist-in-School with A*STAR

This is a repository hosting algorithms and codes developed for a student project focusing on segmentation techniques used for titanium alloy microstructure images.

## Algorithms

We make use of Python and OpenCV to conduct image analysis. Several algorithms have also been developed to conduct subsequent data extraction.

### Image_processing.py

This script entails most of the pre-segmentation processing techniques, including blurring, thresholding and fast Fourier transform (FFT). The display_image and save_images functions allow for batch saving and comparing of the processed images.

### Fast_Fourier_Transform.py

This script is dedicated for experimenting with FFT and frequency domain operations. Testing of mask size and direction is important to the success of FFT.

### Watershed</span>.py

This script conducts watershed transform on a thresholded image, and visualises the segmentation result and the output marker image.

### Oversegmentation</span>.py

This script contains the algorithm developed for reducing oversegmentation.
Labelled regions with an area below the threshold are merged to their surrounding ones.

### Data_extraction.py

This script conducts data extraction from the output marker image from the watershed transform, including area, circumference and diameter.
It also uses the openpyxl package to export the data to a Microsoft Excel workbook.
