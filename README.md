# Papaya Fruit Diseases Detection (Binary Classification) using ML Techniques
This project aims to classify papaya fruits as healthy or diseased through binary classification. The primary objective is to assist farmers in identifying and managing diseased fruits to promote better crop health and yield.

# Proposed Methodology
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Disease-Detection-using-ML-Techniques/assets/84240276/a5e51476-631a-4a4e-baf0-b779a2d0ec73)


## Dataset
Initially the dataset comprises 500 images of healthy papaya fruits and 500 images of diseased papaya fruits. Each image was resized to 100x100 pixels. However, due to the limited dataset, data augmentation techniques were applied to diversify the dataset. This augmentation involved flipping (3 times) and rotation (3 times) for each original image, resulting in a total of 3500 images(500 originals + 3 flips + 3 rotations for each original) for each category.

## Implementation Overview
### 1. Data Curation:
➢Collected a dataset consisting of 500 images for each healthy and diseased papaya fruits.
### 2. Data Augmentation:
➢Augmented the dataset by applying flip and rotation operations, resulting in a total of 3500 images for each category.
### 3. Image Segmentation:
➢Employed eight segmentation techniques to preprocess the images for feature extraction and analysis:
1. Global Thresholding<br>
2. Otsu's Thresholding<br>
3. Adaptive Mean Thresholding<br>
4. Adaptive Gaussian Thresholding<br>
5. Canny Edge Detection<br>
6. Sobel Edge Detection<br>
7. K-means Clustering<br>
8. Fuzzy C-means Clustering<br>
### 4. Model Training:
➢Utilized seven diverse classifiers to train the model and assess performance. The classifiers used were:
1. Decision Tree<br>
2. Naive Bayes<br>
3. Logistic Regression<br>
4. K-Nearest Neighbors (KNN)<br>
5. Ensemble Classifier (Hard Voting)<br>
6. Ensemble Classifier (Soft Voting)<br>
7. Random Forest<br>
### 5. Model Evaluation:
➢Model evaluation employed k-fold cross-validation, analyzing each segmentation technique paired with every classifier to measure accuracy and identify optimal disease detection strategies.<br>
➢The model's robustness & performance was verified by testing it with Diseased and Healthy images after applying each classifier within a segmentation techniques.
### 6. Model Selection:
➢Based on the evaluation results identified Fuzzy C-means segmentation combined with Random Forest as the most accurate model.
### 7. Deployment:
➢Hosted the trained model on the web using Streamlit for easy access and to interact the users with the trained model.<br>
➢Explore the deployed model interface here: https://jhajibhaskar0.streamlit.app/
## Results
Accuracy table showing the performance of each classifier with each segmentation technique, with a special focus on the highest performing combination.
#### Accuracy table
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Binary-Disease-Detection-using-ML-Techniques/assets/84240276/4147caa5-983e-4232-8bf3-c21e16dbee21)







