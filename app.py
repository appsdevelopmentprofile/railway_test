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

    import cv2
    import numpy as np
    import streamlit as st
    
    # Streamlit UI for uploading image
    st.title("Image Analysis with Frame Cleanup")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
        # Display the original uploaded image
        st.subheader("Original Image")
        st.image(img, channels="BGR")
    
        # Step 1: Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Step 2: Thresholding to find the main frame
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
        # Step 3: Contour detection to isolate the main frame
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Step 4: Create a mask for the main frame
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
        # Step 5: Apply the mask to the original image
        cleaned_img = cv2.bitwise_and(img, img, mask=mask)
    
        # Display the cleaned image
        st.subheader("Cleaned Image (Without External Text/Noise)")
        st.image(cleaned_img, channels="BGR")
    
        # Step 6: Perform analysis on the cleaned image
        # Apply thresholding for contour detection
        _, thresh_cleaned = cv2.threshold(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV)
    
        # Detect contours
        contours_cleaned, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Step 7: Draw bounding boxes and extract shapes
        detected_shapes = []
        for i, contour in enumerate(contours_cleaned):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cleaned_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
            # Crop detected shape
            shape_crop = cleaned_img[y:y + h, x:x + w]
            detected_shapes.append(shape_crop)
    
        # Display the processed image
        st.subheader("Processed Image with Bounding Boxes:")
        st.image(cleaned_img, channels="BGR")
    
        # Display the detected shapes in a table format
        if detected_shapes:
            st.subheader("Detected Shapes:")
            cols = st.columns(3)  # Adjust the number of columns based on preference
            for i, shape in enumerate(detected_shapes):
                with cols[i % 3]:  # Cycle through columns
                    st.image(shape, caption=f"Shape {i + 1}")
        else:
            st.write("No shapes detected.")


elif selected == "Field AI Assistant":
    st.title("AI Field Assistant")
    # [Speech-to-Text and Audio Processing Code Here]

elif selected == "AI Testing":
    st.title("Automated AI Testing")
    # [Selenium or Test Automation Code Here]
