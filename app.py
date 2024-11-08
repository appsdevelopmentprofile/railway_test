
import pickle
from pathlib import Path

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator


# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")


# --- USER AUTHENTICATION ---
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:

    # ---- SIDEBAR ----
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
                    st.subheader("Extracted Text from PDF:")
                    st.write(doc_text if doc_text else "No text found in PDF.")
                    pdf.close()

                # Remove temporary file
                os.remove(temp_file_path)

        # Document Analysis Button
        if st.button('Analyze Document Content'):
            # Placeholder for future document content analysis
            st.success("Feature extraction and analysis results will be displayed here.")

