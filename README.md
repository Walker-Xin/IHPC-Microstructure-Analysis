# Microstructure Image Segmentation with Computer Vision Analysis

## NYJC Scientist-in-School with A*STAR IHPC

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

### data_generation.py

This script generates a Microsoft Excel workbook that contains the microstructural data about the image.

### visualisation</span>.py

This is the script for visualisation of different stages of the image processing and the final result.

## Acknowledgment

Our project idea was raised during a discussion with our supervising teacher, Dr Nathaniel Ng. We sincerely thank Dr Nathaniel for his guidance and advice. He provided many valuable suggestions on the image processing procedure and on the writing of this report. We thank Institute of Material Research and Engineering of A*STAR for providing us with the IMRE image. We are also grateful to MIPAR for allowing us to use the MIPAR image, one of the many microstructure images available on the website. Lastly, we would like to express our gratitude to Mrs Judy Tan and Mr Goh Kien Soon, both teachers from Nanyang Junior College, for providing us with the opportunity to conduct this research project with Dr Nathaniel.
