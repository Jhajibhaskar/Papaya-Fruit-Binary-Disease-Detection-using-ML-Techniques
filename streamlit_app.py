import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import joblib

st.image("https://iili.io/dj5iBHB.jpg", width=700)
# Add a small caption below the image
st.markdown(
    """
    <style>
    .caption {
        text-align: center;
        font-size: 12px;
        color: white;
        margin-top: 10px;
    }
    </style>
    <div class="caption">The Team Behind This Project with our inspiring guide</div>
    """,
    unsafe_allow_html=True
)

# Define the categories
Categories = ['Diseased', 'Healthy']

# Function to preprocess the image
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = resize(img_gray, (64, 64))
    return img_resized.flatten().reshape(1, -1)

# Load the trained RandomForestClassifier model
model = joblib.load('./model.h5')

# Streamlit App
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .center-button {
        display: flex;
        justify-content: center;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 20px;
    }
    </style>
    <div class="title">Papaya Fruit Binary Disease Detection using Machine Learning Techniques</div>
    """,
    unsafe_allow_html=True
)

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
    
    prediction = model.predict(preprocessed_img)
    probability = max(model.predict_proba(preprocessed_img)[0])
    predicted_class = Categories[prediction[0]]

    st.write("Predicted Class: ", predicted_class, "Fruit")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by <a href="https://www.linkedin.com/in/jhajibhaskar/" target="_blank">CS20B1060</a> , <a href="https://www.linkedin.com/in/abhishektirkey/" target="_blank">CS20B1002</a> , <a href="https://www.linkedin.com/in/vivek140902/" target="_blank">CS20B1065</a>, Empowering agricultural advancements with machine learning!
    </div>
    """,
    unsafe_allow_html=True
)
