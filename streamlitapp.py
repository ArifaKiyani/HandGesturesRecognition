import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

# Load the trained model
model_path = '/workspaces/HandGesturesRecognition/HandGestRecognition_cnn.h5'  # Replace with the correct path to your model
model = tf.keras.models.load_model(model_path)

# Define class names in the correct order
class_names = ['call_me', 'finger_crossed', 'okay', 'paper','peace' 
               'rock', 'rock_on', 'scissor', 'spock', 'thumbs','up']

# Set Streamlit app title
st.title("Hand Gesture Recognition")
st.write("Upload a grayscale hand gesture image, and the app will predict its class.")

# Function to preprocess the image
def preprocess_image(img):
    """
    Preprocess the uploaded image:
    - Convert to grayscale
    - Resize to (128, 128)
    - Normalize pixel values to [0, 1]
    - Add a batch dimension
    """
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to the model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (H, W, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension (1, H, W, 1)
    return img_array

# File uploader for the image
uploaded_file = st.file_uploader("Choose a hand gesture image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the predicted class
    st.write(f"Predicted Gesture: **{predicted_class}**")

