import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv

# Load the pre-trained model
model = load_model('model2.keras')

# Streamlit App Configuration
st.title("Real-Time Face Mask Detection")
st.markdown(
    """
    Welcome to the Real-Time Face Mask Detection App! 
    Enable your webcam to check for face masks in real-time. 
    - **Green** text indicates a mask is detected.
    - **Red** text indicates no mask is detected.
    """
)
run = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([])

# Frame preprocessing function
def preprocess_frame(frame):
    """
    Preprocesses the frame to match the model input size.

    Parameters:
        frame (np.ndarray): Input video frame.

    Returns:
        np.ndarray: Preprocessed frame ready for prediction.
    """
    img = cv.resize(frame, (224, 224))  # Resize to model's input size
    img = img.reshape(1, 224, 224, 3)  # Add batch dimension
    return img

if run:
    # Open the webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please ensure it is connected and try again.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video frame.")
            break

        # Preprocess the frame and make a prediction
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)[0]
        label = "NO-Mask" if prediction[0] >= 0.5 else "Mask"

        # Set the label color based on the prediction
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert the frame to RGB and display it in Streamlit
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    
    cap.release()
else:
    st.info("Webcam is not running. Check the box above to start detection.")

