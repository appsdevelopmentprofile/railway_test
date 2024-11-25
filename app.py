import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import pytesseract


st.set_page_config(page_title="applus", layout="wide", page_icon="🧑‍⚕️")

# User Authentication
# [Implement fixed authentication mechanism]

# Sidebar
selected = st.sidebar.selectbox("Select Module", ["Doc Intelligence", "Field AI Assistant", "AI Testing"])

# Main Modules
if selected == "Doc Intelligence":

    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    
    # Streamlit frontend
    st.title("Image Processing: Detect Shapes")
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
        # Display the uploaded image
        st.subheader("Uploaded Image:")
        st.image(img, channels="BGR")
        
        # Apply thresholding
        result, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Convert to grayscale
        if len(thresh.shape) == 3:  # If the image is not grayscale
            thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        else:
            thresh_gray = thresh
        
        # Detect contours
        contours = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Draw bounding boxes and extract shapes
        detected_shapes = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop detected shape
            shape_crop = img[y:y+h, x:x+w]
            detected_shapes.append(shape_crop)
        
        # Display the processed image
        st.subheader("Processed Image with Bounding Boxes:")
        st.image(img, channels="BGR")
        
        # Create a table of detected shapes
        if detected_shapes:
            st.subheader("Detected Shapes:")
            cols = st.columns(3)  # Adjust the number of columns based on preference
            for i, shape in enumerate(detected_shapes):
                with cols[i % 3]:  # Cycle through columns
                    st.image(shape, caption=f"Shape {i+1}")
        else:
            st.write("No shapes detected.")



elif selected == "Field AI Assistant":
    st.title("AI Field Assistant")
    # [Speech-to-Text and Audio Processing Code Here]

elif selected == "AI Testing":
    st.title("Automated AI Testing")
    # [Selenium or Test Automation Code Here]
