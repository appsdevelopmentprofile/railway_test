import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], verbose=True)

# Streamlit frontend
st.title("Instrumentation Plan Processing with Text Extraction")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = img.copy()

    # Display the uploaded image
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR")

    # Preprocessing: Remove external frame, text, and continuous lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mask creation and contour filtering
    instrument_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 500 and 50 < h < 500:  # Adjust size thresholds as needed
            instrument_shapes.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the processed image
    st.subheader("Processed Image with Detected Shapes:")
    st.image(img, channels="BGR")

    # Detect circular symbols using Hough Circle Transform
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # Draw circles on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # x, y center
            radius = circle[2]  # radius
            cv2.circle(original_img, center, radius, (0, 255, 0), 2)

    # Display the detected circles
    st.subheader("Processed Image with Detected Circles:")
    st.image(original_img, channels="BGR")

    # Display detected shapes and circles with extracted text
    st.subheader("Extracted Shapes and Text:")
    cols = st.columns(3)  # Adjust the number of columns as needed

    # Process instrument shapes
    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        with cols[i % 3]:
            st.image(cropped_shape, caption=f"Shape {i + 1}")
            st.write(f"Text: {extracted_text}")

    # Process detected circles
    if circles is not None:
        for i, circle in enumerate(circles[0, :]):
            x, y, r = circle
            cropped_circle = original_img[y-r:y+r, x-r:x+r]
            if cropped_circle.size > 0:
                text = reader.readtext(cropped_circle, detail=0)
                extracted_text = " ".join(text) if text else "No text detected"
                with cols[(i + len(instrument_shapes)) % 3]:
                    st.image(cropped_circle, caption=f"Circle {i + 1}")
                    st.write(f"Text: {extracted_text}")
