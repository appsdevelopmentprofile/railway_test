# -*- coding: utf-8 -*-
"""app.py

Streamlit App for Document Intelligence Demo.
"""

# Imports
import os
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pytesseract
import tempfile
import fitz  # PyMuPDF for handling PDFs
import requests
import streamlit_authenticator as stauth
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Allnorth Consultants - RFO Central Application",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --- USER AUTHENTICATION ---
# Define credentials (Note: In production, use hashed passwords for security)
credentials = {
    "usernames": {
        "rfo_central": {
            "name": "RFO Central",
            "password": "1234"  # Plain-text password for demo purposes
        }
    }
}

# Initialize authenticator
authenticator = stauth.Authenticate(credentials, "cookie_name", "key", 30)

# Authentication check
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("Username/password is incorrect")

elif authentication_status is None:
    st.warning("Please enter your username and password")

elif authentication_status:
    # Sidebar and main app if logged in
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            'RFO Central Application',
            [
                "Doc Intelligence",
                "Predictive Analytics for Operational Planning",
                "Real-Time Fault Monitoring",
                "Project Completion Reporting"
            ],
            menu_icon='building',
            icons=['file-earmark-text', 'graph-up', 'exclamation-circle', 'clipboard-check'],
            default_index=0
        )

    # Create temporary directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Doc Intelligence Section
    if selected == 'Doc Intelligence':
        st.title("Document Intelligence with OCR")

        # File uploader for multiple files (images, PDFs)
        uploaded_files = st.file_uploader("Upload your documents (images, PDFs)", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)

        # OneDrive Document Upload Section
        st.header("OneDrive Document Uploader")
        one_drive_upload_url = st.text_input("Enter your OneDrive Upload URL:")
        upload_file = st.file_uploader("Choose a file to upload to OneDrive", type=['pdf', 'jpg', 'jpeg', 'png'])

        # OneDrive upload action
        if upload_file and one_drive_upload_url and st.button("Upload to OneDrive"):
            temp_file_path = os.path.join("temp", upload_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(upload_file.getbuffer())
            with open(temp_file_path, "rb") as f:
                files = {'file': f}
                response = requests.post(one_drive_upload_url, files=files)
            if response.status_code == 200:
                st.success("Upload successful!")
            else:
                st.error("Upload failed. Please check the OneDrive link and try again.")
            os.remove(temp_file_path)

        # Document Processing
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")

                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Handle image files
                if uploaded_file.type.startswith("image/"):
                    image = Image.open(temp_file_path)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    if st.button(f"Extract Text from {uploaded_file.name}", key=f"extract_{uploaded_file.name}"):
                        extracted_text = pytesseract.image_to_string(image)
                        st.subheader("Extracted Text:")
                        st.write(extracted_text if extracted_text else "No text found.")

                # Handle PDF files
                elif uploaded_file.type == "application/pdf":
                    doc_text = ""
                    pdf = fitz.open(temp_file_path)
                    for page_num in range(pdf.page_count):
                        page = pdf[page_num]
                        doc_text += page.get_text("text")
                    st.subheader("Extracted Text fro
