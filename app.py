import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify language(s) to recognize

# Streamlit frontend
st.title("Instrumentation Plan Processor")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR")

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect contours to find shapes
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect circles using Hough Circle Transform
    blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    # Initialize list to store detected shapes and text
    detected_shapes = []

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop detected shape
        shape_crop = gray[y:y + h, x:x + w]
        detected_shapes.append({"image": shape_crop, "bbox": (x, y, w, h)})

    # Process detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)

            # Crop detected circle
            x, y, r = center_x - radius, center_y - radius, radius * 2
            circle_crop = gray[y:y + r, x:x + r]
            detected_shapes.append({"image": circle_crop, "bbox": (x, y, r, r)})

    # Display the processed image with bounding boxes and circles
    st.subheader("Processed Image with Shapes and Circles:")
    st.image(img, channels="BGR")

    # Extract text from each detected shape and circle
    st.subheader("Detected Shapes with Extracted Text:")
    cols = st.columns(3)  # Adjust number of columns as needed

    for i, shape in enumerate(detected_shapes):
        shape_image = shape["image"]
        bbox = shape["bbox"]

        # Use EasyOCR to extract text
        text = reader.readtext(shape_image, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"

        # Display the shape and extracted text
        with cols[i % 3]:
            st.image(shape_image, caption=f"Shape {i + 1}")
            st.write(f"Extracted Text: {extracted_text}")
