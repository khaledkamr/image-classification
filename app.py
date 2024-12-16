import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model = load_model('D:\GitHub\Collage projects\image-classification\Image_classify.keras')

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 
    'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 
    'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 
    'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

# Streamlit app
st.set_page_config(page_title="Fruit & Vegetable Classifier", layout="centered")
st.sidebar.title("About the Model")
st.sidebar.markdown(
    """
    - **Type**: Image Classification
    - **Model**: Convolutional Neural Network (CNN)
    - **Framework**: TensorFlow/Keras
    - **Classes**: 36 Categories
    """
)

st.title("🌟 Fruit & Vegetable Classifier 🌟")
st.write("Upload an image of a fruit or vegetable, and this app will classify it for you!")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process and predict
if uploaded_file:
    try:
        # Load and preprocess image
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)   # Normalize image
        img_bat = tf.expand_dims(img_arr, 0)

        # Make prediction
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Display image
        st.image(uploaded_file, width=250, caption="Uploaded Image")
        
        # Show results
        st.markdown(f"### Predicted Category: **{data_cat[np.argmax(score)]}**")
        st.progress(int(np.max(score) * 100))  # Confidence bar
        st.markdown(f"### 🎯 Confidence: **{np.max(score) * 100:0.2f}%**")
    
    except Exception as e:
        st.error("An error occurred while processing the image. Please try a different file.")
else:
    st.info("👆 Upload an image file to get started!")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by khaledkamr")