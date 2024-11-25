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

    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold the image to make it binary (helps for contour detection)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Get the largest contour (which should be the main content area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 5: Create an empty mask (black image)
    mask = np.zeros_like(gray)

    # Step 6: Draw the largest contour on the mask (white)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Step 7: Apply the mask to the original image
    cleaned_img = cv2.bitwise_and(img, img, mask=mask)

    # Display the cleaned image (frame removed)
    st.subheader("Cleaned Image (Without External Text/Noise)")
    st.image(cleaned_img, channels="BGR")

    # Step 8: Apply further processing to the cleaned image
    # Thresholding to detect contours of the shapes inside the frame
    _, thresh_cleaned = cv2.threshold(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV)

    # Detect contours in the cleaned image
    contours_cleaned, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 9: Draw bounding boxes around detected shapes
    detected_shapes = []
    for i, contour in enumerate(contours_cleaned):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(cleaned_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop detected shape
        shape_crop = cleaned_img[y:y + h, x:x + w]
        detected_shapes.append(shape_crop)

    # Display the processed image with bounding boxes
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
