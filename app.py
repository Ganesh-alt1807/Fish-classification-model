"""
Streamlit deployment application for Multiclass Fish Image Classification.
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from utils import get_class_labels, detect_model_type

# Default data directory
DEFAULT_DATA_DIR = './Dataset/data' if os.path.exists('./Dataset/data') else './data'


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (height, width) for resizing
    
    Returns:
        numpy array: Preprocessed image ready for model input
    """
    # Resize image to target size
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure 3 channels (handle grayscale images)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        # Remove alpha channel if present
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_model_safe(model_path):
    """
    Safely load a Keras model with error handling.
    
    Args:
        model_path: Path to model file
    
    Returns:
        tuple: (model, error_message)
            - model: Loaded model or None if error
            - error_message: Error message or None if successful
    """
    try:
        model = load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def get_model_input_size(model_path):
    """
    Get the required input size for a model based on its filename.
    
    Args:
        model_path: Path to model file
    
    Returns:
        tuple: (height, width) input size
    """
    _, target_size = detect_model_type(model_path)
    return target_size


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Fish Classification",
        page_icon="🐟",
        layout="wide"
    )
    
    # Title
    st.title("🐟 Multiclass Fish Image Classification")
    st.markdown("---")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    # Get available models
    models_dir = './models'
    if not os.path.exists(models_dir):
        st.sidebar.error("Models directory not found: ./models/")
        st.stop()
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.h5'))]
    
    if not model_files:
        st.sidebar.warning("No model files found in ./models/")
        st.info("Please train a model first using train.py")
        st.stop()
    
    # Model selection dropdown
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_files,
        help="Choose a trained model to use for prediction"
    )
    
    # Validate selected_model is not None (should not happen, but defensive check)
    if selected_model is None:
        st.error("Error: No model selected. Please select a model from the sidebar.")
        st.stop()
    
    model_path = os.path.join(models_dir, selected_model)
    
    # Get class labels
    class_labels = get_class_labels(data_dir=DEFAULT_DATA_DIR)
    
    if not class_labels:
        st.error(f"No class labels found. Please ensure {DEFAULT_DATA_DIR} directory exists with class subfolders.")
        st.stop()
    
    # Display model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.write(f"**Model:** {selected_model}")
    st.sidebar.write(f"**Classes:** {len(class_labels)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a fish image to classify"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                uploaded_file = None
    
    with col2:
        st.subheader("🔮 Prediction")
        
        # Predict button
        predict_button = st.button(
            "Predict",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None)
        )
        
        # Perform prediction
        if predict_button and uploaded_file is not None:
            with st.spinner("Loading model and making prediction..."):
                # Load model
                model, error = load_model_safe(model_path)
                
                if error:
                    st.error(f"Error loading model: {error}")
                else:
                    try:
                        # Get model input size
                        target_size = get_model_input_size(model_path)
                        
                        # Preprocess image
                        image = Image.open(uploaded_file)
                        processed_image = preprocess_image(image, target_size=target_size)
                        
                        # Make prediction
                        predictions = model.predict(processed_image, verbose=0)
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class_idx]
                        
                        # Get predicted label
                        if predicted_class_idx < len(class_labels):
                            predicted_label = class_labels[predicted_class_idx]
                        else:
                            predicted_label = f"Class {predicted_class_idx}"
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        st.markdown("---")
                        
                        # Predicted class
                        st.markdown(f"### 🎯 Predicted Class")
                        st.markdown(f"**{predicted_label}**")
                        
                        # Confidence score
                        st.markdown(f"### 📊 Confidence Score")
                        st.markdown(f"**{confidence * 100:.2f}%**")
                        
                        # Progress bar for confidence
                        st.progress(float(confidence))
                        
                        # Show top 3 predictions
                        st.markdown("### 📈 Top 3 Predictions")
                        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                        
                        for i, idx in enumerate(top_3_indices, 1):
                            label = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
                            score = predictions[0][idx] * 100
                            st.write(f"{i}. **{label}**: {score:.2f}%")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
        elif predict_button and uploaded_file is None:
            st.warning("⚠️ Please upload an image first!")
        
        else:
            st.info("👆 Upload an image and click 'Predict' to get classification results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Multiclass Fish Image Classification | Built with Streamlit & TensorFlow</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
