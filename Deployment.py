import streamlit as st
import numpy as np
from PIL import Image
import os
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path):
        """Initialize the emotion detector with a model path."""
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Safely load the TensorFlow model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info("Model loaded successfully")
                return True
            else:
                logger.error(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        try:
            import tensorflow as tf
            
            # Log initial image information
            logger.debug(f"Initial image type: {type(image)}")
            logger.debug(f"Initial image size: {image.size}")
            logger.debug(f"Initial image mode: {image.mode}")
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                logger.debug("Converting RGBA to RGB")
                image = image.convert('RGB')
            
            # Convert grayscale to RGB if necessary
            if image.mode == 'L':
                logger.debug("Converting grayscale to RGB")
                image = image.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            logger.debug(f"Numpy array shape after conversion: {img_array.shape}")
            logger.debug(f"Array data type: {img_array.dtype}")
            
            # Check if the image is in the correct format
            if len(img_array.shape) != 3:
                raise ValueError(f"Expected 3 dimensions, got {len(img_array.shape)}")
            
            if img_array.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {img_array.shape[2]}")
            
            # Resize the image
            resize = tf.image.resize(img_array, (256, 256))
            logger.debug(f"Resized shape: {resize.shape}")
            
            # Ensure the values are in float format
            resize = tf.cast(resize, tf.float32)
            
            # Normalize the image
            normalized_img = resize / 255.0
            
            # Check the value range
            logger.debug(f"Min value: {np.min(normalized_img)}, Max value: {np.max(normalized_img)}")
            
            return normalized_img
            
        except Exception as e:
            logger.error(f"Error in preprocess_image: {str(e)}")
            logger.exception("Detailed traceback:")
            return None
    
    def predict_emotion(self, image):
        """Predict emotion from the image."""
        try:
            if self.model is None:
                return "Error: Model not loaded", 0.0
            
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return "Error: Image preprocessing failed", 0.0
            
            # Expand dimensions and make prediction
            input_img = np.expand_dims(processed_img, axis=0)
            logger.debug(f"Input shape to model: {input_img.shape}")
            
            prediction = self.model.predict(input_img)
            logger.debug(f"Raw prediction: {prediction}")
            
            emotion = "Sad" if prediction[0][0] > 0.5 else "Happy"
            confidence = float(abs(prediction[0][0] - 0.5) * 2)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return "Error: Prediction failed", 0.0

def main():
    st.set_page_config(
        page_title="Emotion Detection App",
        page_icon="😊",
        layout="centered"
    )
    
    st.title("🎭 Emotion Detection App")
    st.write("Upload an image and the app will predict if the person appears happy or sad.")
    
    # Display debug information in sidebar
    # with st.sidebar:
    #     st.header("Debug Information")
    #     st.text(f"Python Version: {sys.version}")
    #     try:
    #         import tensorflow as tf
    #         st.text(f"TensorFlow Version: {tf.__version__}")
    #     except ImportError:
    #         st.error("TensorFlow not installed")
    
    # Initialize the detector
    detector = EmotionDetector('models/happysadmodel.h5')
    
    # Model loading status
    with st.sidebar:
        st.header("Model Status")
        if detector.load_model():
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Error loading model")
            st.stop()
    
    # File uploader with clear instructions
    st.write("### Upload Image")
    st.write("Please upload a clear image containing a face. Supported formats: JPG, JPEG, PNG")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="For best results, use a well-lit, front-facing photo"
    )
    
    if uploaded_file is not None:
        try:
            # Read and display the image
            image = Image.open(uploaded_file)
            
            # Display image information
            st.write("### Image Information")
            st.write(f"Image size: {image.size}")
            st.write(f"Image mode: {image.mode}")
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a prediction button
            if st.button("Predict Emotion"):
                with st.spinner("Analyzing image..."):
                    emotion, confidence = detector.predict_emotion(image)
                    
                    if isinstance(emotion, str) and emotion.startswith("Error"):
                        st.error(emotion)
                        # Show debug logs
                        with st.expander("Show Debug Information"):
                            st.write("If you're seeing an error, please ensure:")
                            st.write("1. The image contains a clear, well-lit face")
                            st.write("2. The image is not corrupted")
                            st.write("3. The image is in RGB format")
                            logger_messages = logging.getLogger().handlers[0].stream.getvalue()
                            st.code(logger_messages)
                    else:
                        # Create columns for the result
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Emotion", emotion)
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Add emoji based on prediction
                        emoji = "😊" if emotion == "Happy" else "😢"
                        st.markdown(f"### {emoji} Detected emotion: {emotion}")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Error in main app flow: {str(e)}")
            
            # Show technical details in expander
            with st.expander("Technical Details"):
                st.code(str(e))

if __name__ == "__main__":
    main()