import cv2
import numpy as np
import streamlit as st

# Streamlit UI for uploading image
st.title("Image Analysis: Frame and Text Removal with Shape Detection")
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

    # Step 2: Apply adaptive thresholding for better contrast on variable lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # Step 3: Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter contours to remove small noise, keep only large enough contours
    min_contour_area = 1000  # Minimum contour area to consider
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Step 5: Get the bounding box of the largest contour (this should cover the main content area)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Step 6: Crop the image to this bounding box to remove unwanted frame areas
        cropped_img = img[y:y+h, x:x+w]

        # Display the cropped image (without the frame)
        st.subheader("Cropped Image (Frame Removed)")
        st.image(cropped_img, channels="BGR")

        # Step 7: Further image processing (detecting shapes) without the frame
        gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresh_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)

        # Step 8: Remove small contours at the edges (text near the edges)
        # Create a mask to remove small contours around the edges
        mask = np.zeros_like(thresh_cropped)

        # Apply contours filtering and keep only large contours (removing small text-like contours)
        contours_cropped, _ = cv2.findContours(thresh_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours_cropped if cv2.contourArea(cnt) > 500]  # Filter small contours

        # Draw the filtered contours on the mask
        cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

        # Use the mask to filter the cropped image
        filtered_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        # Display the filtered image
        st.subheader("Filtered Image (Small Text Removed)")
        st.image(filtered_img, channels="BGR")

        # Step 9: Detect shapes in the filtered image
        gray_filtered = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        _, thresh_filtered = cv2.threshold(gray_filtered, 150, 255, cv2.THRESH_BINARY_INV)

        # Detect contours again in the filtered image
        contours_filtered, _ = cv2.findContours(thresh_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected shapes
        detected_shapes = []
        for i, contour in enumerate(contours_filtered):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(filtered_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop detected shape
            shape_crop = filtered_img[y:y + h, x:x + w]
            detected_shapes.append(shape_crop)

        # Display the processed image (with bounding boxes)
        st.subheader("Processed Image with New Bounding Boxes:")
        st.image(filtered_img, channels="BGR")

        # Display the detected shapes in a table format
        if detected_shapes:
            st.subheader("Detected Shapes:")
            cols = st.columns(3)  # Adjust the number of columns based on preference
            for i, shape in enumerate(detected_shapes):
                with cols[i % 3]:  # Cycle through columns
                    st.image(shape, caption=f"Shape {i + 1}")
        else:
            st.write("No shapes detected.")
    else:
        st.write("No contours found for the frame cleanup.")
