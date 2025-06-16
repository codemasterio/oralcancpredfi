import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image, ImageEnhance
import io
import time

# Set page config with wide layout and initial sidebar state
st.set_page_config(
    page_title="Oral Cancer Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
        padding: 2rem 4%;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Upload box */
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        background-color: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .upload-box:hover {
        border-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Result boxes */
    .result-box {
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .result-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    
    .cancer {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
    }
    
    .non-cancer {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(30, 136, 229, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(30, 136, 229, 0.4);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
        .header p {
            font-size: 1rem;
        }
        
        .result-box {
            padding: 1.2rem;
            font-size: 1rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load the pre-trained Random Forest model"""
    try:
        model = joblib.load('oral_cancer_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if it's a color image
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 100x100
        img_resized = cv2.resize(img_array, (100, 100))
        
        # Flatten to 1D array (10000 features)
        img_flattened = img_resized.flatten()
        
        # Reshape for model prediction
        img_processed = img_flattened.reshape(1, -1)
        
        return img_processed, img_resized
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    # Header section
    st.markdown("""
    <div class="header">
        <h1>🦷 Oral Cancer Detection</h1>
        <p>Upload an oral cavity image to detect potential signs of oral cancer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model at the start
    model = load_model()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        st.markdown("""
        <div class="card">
            <h3>Upload Oral Cavity Image</h3>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload an image...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Information card
        st.markdown("""
        <div class="card">
            <h3>How It Works</h3>
            <p>1. Upload an image of the oral cavity</p>
            <p>2. Click 'Analyze Image'</p>
            <p>3. View the analysis results</p>
            <p>4. Consult a professional for medical advice</p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            
            # Display original image in a card with medium size
            with st.container():
                st.markdown("### 📷 Uploaded Image")
                # Display image with medium size (max width 500px for better visibility)
                st.image(image, 
                        width=min(500, image.width),  # Limit width to 500px or original width if smaller
                        caption=f"Original Image ({image.width} × {image.height})", 
                        use_container_width=False)
            
            # Add a nice divider
            st.markdown("<hr style='border: 1px solid #eee; margin: 2rem 0;'>", unsafe_allow_html=True)
            
            # Analyze button with improved styling
            if st.button("🔍 Analyze Image", key="analyze_btn"):
                with st.spinner("🔬 Analyzing image... This may take a moment..."):
                    # Add a progress bar for better UX
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    for percent_complete in range(0, 101, 10):
                        time.sleep(0.1)  # Simulate processing time
                        progress_bar.progress(percent_complete)
                    
                    # Preprocess image
                    img_processed, img_processed_display = preprocess_image(image)
                    
                    if img_processed is not None and model is not None:
                        # Make prediction
                        prediction = model.predict(img_processed)
                        proba = model.predict_proba(img_processed)
                        
                        # Get confidence
                        confidence = max(proba[0]) * 100
                        
                        # Display results in a card
                        st.markdown("<div class='card'><h2>Analysis Results</h2></div>", unsafe_allow_html=True)
                        
                        # Two columns for results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("### Processed Image")
                            # Convert numpy array to PIL Image for consistent display
                            if isinstance(img_processed_display, np.ndarray):
                                processed_img = Image.fromarray(img_processed_display)
                                st.image(processed_img,
                                       width=min(500, processed_img.width),  # Same medium size as uploaded image
                                       caption=f"Preprocessed (Grayscale & Resized to 100×100)",
                                       use_container_width=False)
                            else:
                                st.image(img_processed_display,
                                       width=min(500, img_processed_display.width),
                                       caption=f"Preprocessed (Grayscale & Resized to 100×100)",
                                       use_container_width=False)
                        
                        with col2:
                            result = "CANCER" if prediction[0] == 1 else "NON CANCER"
                            confidence_txt = f"{confidence:.2f}%"
                            
                            # Display result with appropriate styling
                            if result == "CANCER":
                                st.markdown(
                                    f"<div class='result-box cancer'>"
                                    f"<h2 style='color: #c62828;'>⚠️ {result} DETECTED</h2>"
                                    f"<p style='font-size: 1.4rem;'>Confidence: {confidence_txt}</p>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                st.warning("""
                                    **Important Notice:**  
                                    This result indicates potential signs of oral cancer. 
                                    Please consult with a healthcare professional for 
                                    a thorough examination and proper medical advice.
                                """)
                            else:
                                st.markdown(
                                    f"<div class='result-box non-cancer'>"
                                    f"<h2 style='color: #2e7d32;'>✅ NO CANCER DETECTED</h2>"
                                    f"<p style='font-size: 1.4rem;'>Confidence: {confidence_txt}</p>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                st.success("""
                                    **Good news!** No signs of oral cancer were detected.  
                                    However, regular dental check-ups are still recommended 
                                    for maintaining good oral health.
                                """)
                        
                        # Add a section for model insights
                        st.markdown("### 🔍 Model Insights")
                        
                        # Display feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            with st.expander("View Feature Importance"):
                                st.info("""
                                    The heatmap below shows which regions of the image were most important 
                                    for the model's prediction. Warmer colors indicate areas that had a 
                                    stronger influence on the decision.
                                """)
                                
                                # Reshape feature importance to match image dimensions
                                if hasattr(model, 'estimators_'):
                                    # For Random Forest
                                    importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
                                    importances = importances.reshape(100, 100)
                                    
                                    # Normalize for display
                                    importances = (importances - importances.min()) / (importances.max() - importances.min() + 1e-10)
                                    
                                    # Display heatmap with improved styling
                                    st.image(importances, 
                                           caption="Feature Importance Heatmap (Brighter areas were more influential in the prediction)", 
                                           use_column_width=True,
                                           output_format='PNG')
                        
                        # Add a disclaimer
                        st.markdown("""
                        <div class='footer'>
                            <hr>
                            <p><strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
                            Always seek the advice of your dentist or other qualified health provider with any questions you may have regarding a medical condition.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Scroll to results
                        st.markdown("<div id='results'></div>", unsafe_allow_html=True)
                        st.markdown(
                            """
                            <script>
                            document.getElementById('results').scrollIntoView({behavior: 'smooth'});
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    **Note:** This is a machine learning model for research purposes only. 
    Always consult with a healthcare professional for medical diagnosis.
    """)

if __name__ == "__main__":
    main()
