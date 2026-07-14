import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Define absolute paths
MODEL_PATH = r"D:\HMIES MAJOR\real_fake_cnn_model.h5"
CLASSES_PATH = r"D:\HMIES MAJOR\real_fake_classes.npy"

# Load model and class labels safely
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(CLASSES_PATH):
        st.error(f"❌ Class label file not found at: {CLASSES_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return model, classes

# Load model and classes
model, classes = load_model_and_classes()

# Image classification logic with shape check
def classify_image(image, model, classes, image_size=(64, 64)):
    try:
        img = image.convert('RGB').resize(image_size)
        img_arr = np.array(img) / 255.0

        # Check shape
        if img_arr.shape != (64, 64, 3):
            return None, f"Image shape mismatch: expected (64,64,3), got {img_arr.shape}"

        input_arr = np.expand_dims(img_arr, axis=0)
        preds = model.predict(input_arr)

        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]
        return classes[class_idx], float(confidence)
    except Exception as e:
        return None, f"Exception during classification: {str(e)}"

# Streamlit UI
st.title("🧠 Real vs Fake Image Classifier")

uploaded_file = st.file_uploader("Upload a JPG/PNG image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        label, confidence = classify_image(image, model, classes)

        if label:
            st.success(f"✅ Predicted: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.error(f"⚠️ Prediction failed: {confidence}")
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("📁 Please upload an image to classify.")
