import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load your machine learning model (model_rf) and Categories here
Categories = ['Diseased', 'Healthy']
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                                  min_samples_split=12, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
                                  max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                  verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                  max_samples=None)

# Function to preprocess the image
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vectorized = img_gray.reshape((1, -1))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img_gray.shape))
    img_resize = resize(result_image, (64, 64))
    return img_resize.flatten().reshape(1, -1)

# Load the data and model
datadir = './DATA SET'
flat_data_arr = []  # input array
target_arr = []  # output array

for i in Categories:
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        preprocessed_img = preprocess_image(img_array)
        flat_data_arr.append(preprocessed_img.flatten())
        target_arr.append(Categories.index(i))

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

model_rf.fit(x_train, y_train)

# Streamlit App
st.title('Papaya Fruit Disease Detection')

st.write("This app can predict whether the papaya fruit in the image is Diseased or Healthy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make prediction
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preprocessed_img = preprocess_image(img)
    prediction = model_rf.predict(preprocessed_img)
    probability = max(model_rf.predict_proba(preprocessed_img)[0])
    predicted_class = Categories[prediction[0]]

    st.write("Prediction:", predicted_class, "Fruit")
    st.write("Probability: {:.2f}%".format(probability * 100))


    # Display the prediction probabilities for each class
    #st.write("Prediction Probabilities:")
    #for ind, val in enumerate(Categories):
        #st.write(f'{val}: {probability * 100:.2f}%')

    # Display additional metrics
    #st.write("Additional Metrics:")
    y_pred_rf = model_rf.predict(x_test)
    accuracy = accuracy_score(y_pred_rf, y_test)
    precision = precision_score(y_pred_rf, y_test, average='weighted')
    recall = recall_score(y_pred_rf, y_test, average='weighted')
    f1 = f1_score(y_test, y_pred_rf, average='weighted')

    #st.write("Accuracy:", round(accuracy * 100, 2), "%")
    #st.write("Precision:", round(precision * 100, 2), "%")
    #st.write("Recall:", round(recall * 100, 2), "%")
    #st.write("F1 Score:", round(f1 * 100, 2), "%")
