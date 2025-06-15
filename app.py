import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Oral Cancer Detection",
    page_icon="🦷",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('saved_model/oral_cancer_model.joblib')

model = load_model()

# Title and description
st.title("🦷 Oral Cancer Detection")
st.write("Upload an image of an oral cavity to detect potential signs of cancer.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Preprocess the image for prediction"""
    # Convert to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Resize to match training size
    img = cv2.resize(img, (100, 100))
    # Flatten and normalize
    return img.flatten() / 255.0

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing image...'):
        try:
            # Preprocess image
            processed_img = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict([processed_img])[0]
            proba = model.predict_proba([processed_img])[0]
            
            # Display results
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", 
                         "CANCER" if prediction == 1 else "NON CANCER",
                         delta=f"{max(proba)*100:.2f}% confidence",
                         delta_color="inverse")
            
            with col2:
                st.write("**Probabilities:**")
                st.progress(proba[1], text=f"Cancer: {proba[1]*100:.1f}%")
                st.progress(proba[0], text=f"Non-Cancer: {proba[0]*100:.1f}%")
            
            # Display interpretation
            if prediction == 1:
                st.warning("⚠️ This image shows potential signs of oral cancer. Please consult a healthcare professional for further evaluation.")
            else:
                st.success("✅ No signs of oral cancer detected. However, regular dental check-ups are recommended.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some information
st.markdown("""
### About
This AI model helps in detecting potential signs of oral cancer from images. 

**Note:** This tool is for educational and research purposes only and should not replace professional medical advice.
""")
