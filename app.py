import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import onnxruntime as ort

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], verbose=True)

# Path to the ONNX model
onnx_model_path = '/content/drive/MyDrive/saved_models/my_pid_model.onnx'

# Load the ONNX model
session = ort.InferenceSession(onnx_model_path)

# Streamlit app title
st.title("P&ID Instrumentation and Symbol Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png", "PNG"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = img.copy()

    # Display the uploaded image
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR")

    # ONNX Symbol Detection
    st.subheader("ONNX Symbol Detection")

    # Preprocess image for ONNX model
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(input_image, (640, 640))  # Resize to model input size
    input_tensor = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

    # Perform inference with ONNX model
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

    # Parse ONNX model output
    detections = outputs[0][0]  # Assuming output is in YOLO format
    for detection in detections:
        confidence = detection[4]
        if confidence > 0.5:  # Filter by confidence threshold
            x_min, y_min, x_max, y_max = (detection[:4] * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).astype(int)
            label = int(detection[5])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            st.write(f"Detected: Class {label} with confidence {confidence:.2f}")

    # Display annotated image with ONNX results
    st.image(img, caption="ONNX Annotated Image", use_column_width=True)

    # EasyOCR Text Detection and Instrument Shapes
    st.subheader("Text Extraction and Shape Detection")

    # Preprocessing for contours
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect and annotate instrument shapes
    instrument_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 500 and 50 < h < 500:  # Adjust thresholds as needed
            instrument_shapes.append((x, y, w, h))
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect circles using Hough Circle Transform
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

    # Display detected shapes and text
    st.subheader("Processed Image with Detected Shapes and Circles")
    st.image(original_img, channels="BGR")

    # Extract text from detected shapes
    st.subheader("Extracted Text from Detected Shapes and Circles")
    cols = st.columns(3)

    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        with cols[i % 3]:
            st.image(cropped_shape, caption=f"Shape {i + 1}")
            st.write(f"Text: {extracted_text}")

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
