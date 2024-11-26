import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], verbose=False)

# Define the path to the YOLO model file
model_file = "best.pt"
if not os.path.exists(model_file):
    st.error(f"YOLO model file '{model_file}' not found. Ensure it is in the same directory as this script.")
    st.stop()

# Load the YOLO model
model = YOLO(model_file)

# Streamlit app title
st.title("P&ID Instrumentation and Symbol Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = img.copy()

    # Display the uploaded image
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR", caption="Uploaded Image")

    # YOLO Detection
    st.subheader("Symbol Detection with YOLO")
    results = model(img)

    # Annotate detected symbols
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0])
        label = model.names[cls]

        x_min, y_min, x_max, y_max = map(int, xyxy)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display annotated image
    st.image(img, caption="YOLO Detection Results", use_column_width=True)

    # EasyOCR Text Detection
    st.subheader("Text Detection with EasyOCR")
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect instrument shapes
    instrument_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 500 and 50 < h < 500:  # Thresholds to filter shapes
            instrument_shapes.append((x, y, w, h))
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hough Circle Detection
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50
    )

    # Annotate circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(original_img, center, radius, (0, 255, 0), 2)

    # Display annotated shapes and circles
    st.image(original_img, caption="Detected Shapes and Circles", channels="BGR")

    # Extract text from detected regions
    st.subheader("Extracted Text from Detected Shapes and Circles")
    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        st.image(cropped_shape, caption=f"Shape {i + 1}: {extracted_text}")

    if circles is not None:
        for i, circle in enumerate(circles[0, :]):
            x, y, r = circle
            cropped_circle = original_img[y-r:y+r, x-r:x+r]
            if cropped_circle.size > 0:
                text = reader.readtext(cropped_circle, detail=0)
                extracted_text = " ".join(text) if text else "No text detected"
                st.image(cropped_circle, caption=f"Circle {i + 1}: {extracted_text}")
