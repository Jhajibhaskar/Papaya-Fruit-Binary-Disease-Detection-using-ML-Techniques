import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Define the categories
Categories = ['Diseased', 'Healthy']

# Function to preprocess the image
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = resize(img_gray, (64, 64))
    return img_resized.flatten().reshape(1, -1)

# Load the data
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

# Apply Fuzzy C-Means clustering
n_clusters = len(Categories)
fuzziness = 2.0
max_iter = 100

# Scale the data
scaler = MinMaxScaler()
flat_data_scaled = scaler.fit_transform(flat_data)

# Perform Fuzzy C-Means clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(flat_data_scaled.T, n_clusters, fuzziness, maxiter=max_iter, error=0.005)
cluster_labels = np.argmax(u, axis=0)

# Create a DataFrame for the clustered data
df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
df['Cluster'] = cluster_labels

# Split the data into training and testing sets
x = df.iloc[:, :-2]  # input data
y = df['Cluster']  # output data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# Train the RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                                  min_samples_split=12, min_samples_leaf=5, max_features='log2')
model_rf.fit(x_train, y_train)


# Streamlit App
st.title('Papaya Fruit Binary Disease Detection using ML Techniques')

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

    st.write("Predicted Class: ", predicted_class, "Fruit")


# Run the app with: streamlit run app.py
