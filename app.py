import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- MEMORY SAFETY FIX ---
# This stops TensorFlow from trying to "reserve" all the server's RAM at once
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

st.set_page_config(page_title="AI Vision", page_icon="📸")
st.title("📸 AI Image Classifier (Cloud Version)")

@st.cache_resource
def load_model():
    # Load the "lightweight" version of the brain
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader("Upload a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Image', use_column_width=True)
    
    # Pre-processing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    st.subheader("Results:")
    for i, (id, label, score) in enumerate(decoded):

        st.write(f"{i+1}. *{label.title()}* ({score*100:.1f}%)")
