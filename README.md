# Papaya-Fruit-Disease-Detection-using-ML-Techniques
This project aims to classify papaya fruits as healthy or diseased through binary classification. The primary objective is to assist farmers in identifying and managing diseased fruits to promote better crop health and yield.

# Proposed Methodology
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Disease-Detection-using-ML-Techniques/assets/84240276/4fb9297a-99f2-4f2a-a93b-71b5b6832318)

## Dataset
The dataset comprises 30 images of healthy papaya fruits and 30 images of diseased papaya fruits. Each image was resized to 100x100 pixels. However, due to the limited dataset, data augmentation techniques were applied to diversify the dataset. This augmentation involved flipping (3 times) and rotation (3 times) for each original image, resulting in a total of 210 images for each category.

## Implementation Overview
### 1. Data Curation:
➢Collected a dataset consisting of 30 images for both healthy and diseased papaya fruits.
### 2. Data Augmentation:
➢Augmented the dataset by applying flip and rotation operations, resulting in a total of 210 images for each category.
### 3. Image Segmentation:
➢Employed eight segmentation techniques to preprocess the images for feature extraction and analysis:<br>
1.Global Thresholding<br>
2.Otsu's Thresholding<br>
3.Adaptive Mean Thresholding<br>
4.Adaptive Gaussian Thresholding<br>
5.Canny Edge Detection<br>
6.Sobel Edge Detection<br>
7.K-means Clustering<br>
8.Fuzzy C-means Clustering<br>
### 4. Model Training:






