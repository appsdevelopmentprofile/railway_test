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
import pickle
from pathlib import Path
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier  # For demonstration of fault detection

# Set page configuration
st.set_page_config(
    page_title="Allnorth Consultants - RFO Central Application",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --- USER AUTHENTICATION ---
names = ["allnorth_consultants"]
usernames = ["rfocentral"]

# Load hashed passwords
file_path = Path("hashed_pw.pkl")
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords, "allnorth_consultants", "rfocentral", cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", "main")

# If the user clicks "Logout", update session to reflect the logout
if "logged_out" not in st.session_state:
    st.session_state["logged_out"] = False  # Initial state for tracking logout

if authentication_status == False:
    st.error("Username/password is incorrect")

elif authentication_status == None:
    st.warning("Please enter your username and password")

elif authentication_status:
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

        from huggingface_hub import from_pretrained_keras

        # Load the model from Hugging Face Hub
        model = from_pretrained_keras("appsdevelopmentprofile/doc_intelligence_model")

        
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
            st.success("Feature extraction and analysis results will be displayed here.")



 ## Module 2 - Predictive Analytics for Operational Planning
    
    
    # Bring up the page when it is selected on the side bar
    if selected == 'Predictive Analytics for Operational Planning':
        # Importing necessary libraries to make the front end work
        import pandas as pd
        import matplotlib.pyplot as plt
        import streamlit as st
    
        # Dummy NLP class to simulate the behavior of spaCy's NLP model
        class DummyNLP:
            def __call__(self, text):
                # Dummy entity extraction: Just return some mock entities for demonstration
                return DummyDoc(text)
    
        class DummyDoc:
            def __init__(self, text):
                self.text = text
                self.ents = [DummyEntity("mock_entity_1", "ENTITY_TYPE_1"),
                             DummyEntity("mock_entity_2", "ENTITY_TYPE_2")]
    
        class DummyEntity:
            def __init__(self, text, label):
                self.text = text
                self.label_ = label
    
        # Loading the dummy model
        nlp = DummyNLP()
    
        # Displaying Streamlit, starting with the title
        st.title('Predictive Analytics for Operational Planning and Project Management')
    
        # Input section for Work Descriptions and Incident Reports
        st.subheader("Input Work Descriptions and Incident Reports")
        col1, col2 = st.columns(2)
    
        with col1:
            work_descriptions = st.text_area("Enter Work Descriptions (comma-separated):",
                                              "Replace pressure-safety-valve on wellhead, Inspect hydraulics for leaks, Routine maintenance on compressor, Test gas leak detection system, Install new fire safety cabinet")
    
        with col2:
            incident_reports = st.text_area("Enter Incident Reports (comma-separated):",
                                             "Trapped finger while replacing valve, Hydraulic fluid leakage caused fire, Compressor malfunctioned during routine maintenance, Gas detector failed to trigger during test, Fire safety cabinet installation delayed due to missing parts")
    
        # Setting the code for Prediction
        predictive_analytics = ''
    
        # Creating a button for Prediction
        if st.button("Analyze"):
            # Preprocess input data
            work_list = [desc.strip() for desc in work_descriptions.split(",")]
            incident_list = [report.strip() for report in incident_reports.split(",")]
    
            # Create a DataFrame
            data = {
                'Work_Description': work_list,
                'Incident_Report': incident_list
            }
            df = pd.DataFrame(data)
    
            # Making the pre-trained model work with the input data (in this case, the dummy model)
            # Function to extract entities
            def extract_entities(text):
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                return entities
    
            # Apply entity extraction
            df['Work_Entities'] = df['Work_Description'].apply(extract_entities)
            df['Incident_Entities'] = df['Incident_Report'].apply(extract_entities)
    
            # Function to match risks
            def match_risks(work_entities, incident_entities):
                common_entities = set(work_entities).intersection(set(incident_entities))
                return len(common_entities) > 0, common_entities
    
            # Apply risk matching
            df['Risk_Match'], df['Common_Entities'] = zip(*df.apply(lambda row: match_risks(row['Work_Entities'], row['Incident_Entities']), axis=1))
    
            # Displaying the results
            st.subheader("Results")
            st.write(df[['Work_Description', 'Incident_Report', 'Risk_Match', 'Common_Entities']])
    
            # Visualize matched risks
            matched_risks = df['Risk_Match'].value_counts()
            plt.figure()
            matched_risks.plot(kind='bar', color=['green', 'red'])
            plt.title('Risk Matches in Operational Planning')
            plt.xlabel('Risk Match')
            plt.ylabel('Count')
            st.pyplot(plt)
    
        # Finishing the code
        st.success(predictive_analytics)
    
    
    
    ## Module 3 - Automated Field Tech Assistance - Fault Detection
    
    # Bring up the page when it is selected on the sidebar
    if selected == "Real-Time Fault Monitoring":
        # Importing necessary libraries to make the front end work
        import streamlit as st
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        import matplotlib.pyplot as plt
        from sklearn.dummy import DummyClassifier  # Dummy model for demonstration
    
        # Creating a dummy model (replace with actual model later)
        fault_model = DummyClassifier(strategy='most_frequent')
    
        # Displaying Streamlit, starting with the title
        st.title("Real-Time Fault Monitoring")
    
        # Input data submission with the 10 variables
        st.subheader("Submit Your Data")
        col1, col2 = st.columns(2)
    
        # Step 1: Data Submission
        with col1:
            uploaded_file = st.file_uploader("Upload a CSV file with your data (10 variables, rows of samples)", type=["csv"])
    
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(data.head())
    
        with col2:
            # Step 2: Normalize the data
            if uploaded_file is not None:
                scaler = StandardScaler()
                data_normalized = scaler.fit_transform(data)
    
                # Step 3: Apply PCA for initial fault detection
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_normalized)
    
                # Plot the PCA result for fault detection visualization
                st.subheader("PCA for Fault Detection")
                fig1, ax1 = plt.subplots()
                ax1.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='Normal')
                ax1.set_title('PCA for Fault Detection')
                ax1.set_xlabel('Principal Component 1')
                ax1.set_ylabel('Principal Component 2')
                ax1.legend()
                st.pyplot(fig1)
    
        # Step 4: User Inputs for Prediction
        st.subheader("Input Parameters for Fault Diagnosis")
        input_values = []
        if uploaded_file is not None:
            columns = [
                "Temperature (°C)",
                "Pressure (bar)",
                "Vibration Level (mm/s)",
                "Flow Rate (L/min)",
                "Voltage (V)",
                "Current (A)",
                "RPM (Revolutions Per Minute)",
                "Humidity (%)",
                "Power Consumption (kW)",
                "Runtime (hours)"
            ]
            for column in columns:
                input_value = st.number_input(f"Enter value for {column}", value=0.0)  # Default to 0.0
                input_values.append(input_value)
    
        # Making the pre-trained model work with the input data
        # Code for Prediction
        fault_diagnosis = ''
        if st.button("Fault Diagnosis Result"):
            # Normalize user input
            user_input_normalized = scaler.transform([input_values])
    
            # Make the prediction using the dummy model
            fault_prediction = fault_model.fit(data_normalized, np.random.randint(2, size=len(data))).predict(user_input_normalized)
    
            # Set diagnosis message based on prediction
            if fault_prediction[0] == 1:
                fault_diagnosis = "There is a fault in this asset"
            else:
                fault_diagnosis = "The asset is not presenting any fault"
    
            # Displaying the results
            st.success(fault_diagnosis)
    
        # Step 5: Hierarchical clustering for fault classification
        if uploaded_file is not None:
            clustering = AgglomerativeClustering(n_clusters=3)
            clusters = clustering.fit_predict(pca_result)
    
            # Plot the clustering result
            st.subheader("Hierarchical Clustering of Faults")
            fig2, ax2 = plt.subplots()
            scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='rainbow', label='Fault Cluster')
            ax2.set_title('Hierarchical Clustering of Faults')
            ax2.set_xlabel('Principal Component 1')
            ax2.set_ylabel('Principal Component 2')
            plt.colorbar(scatter, ax=ax2, label='Cluster Label')
            st.pyplot(fig2)
    
            # Step 6: Simulate and Plot LLEDA for fault identification
            st.subheader("Fault Identification (LLEDA Simulation)")
            fig3, ax3 = plt.subplots()
            for cluster_label in np.unique(clusters):
                ax3.scatter(pca_result[clusters == cluster_label, 0], pca_result[clusters == cluster_label, 1],
                            label=f'Fault Type {cluster_label + 1}')
            ax3.set_title('LLEDA for Fault Identification')
            ax3.set_xlabel('Principal Component 1')
            ax3.set_ylabel('Principal Component 2')
            ax3.legend()
            st.pyplot(fig3)
    
        # Feedback section from the field engineer
        st.subheader("Provide Feedback on Detected Faults")
        feedback = st.text_area("Enter your feedback or additional observations:")
        if st.button("Submit Feedback"):
            st.success("Feedback submitted successfully!")
            st.write("Your feedback:", feedback)
            # Optionally, save the feedback to a file or database (implement as needed)
    
    
    # Module 4 - Project Completion Reporting for Oil and Gas
    
    # Bring up the page when it is selecte on the side bar
    if selected == "Project Completion Reporting":
    
        import streamlit as st
        import base64
        import docx
        import datetime
        import os
    
    # Dummy model class for demonstration
        class DummyModel:
            def predict(self, input_data):
                return "This is dummy generated content based on input data."
        
        # Load the dummy model (no actual model file)
        report_generator_model = DummyModel()
        
        # Displaying Streamlit, starting with the title
        st.title("Project Completion Reporting for Oil and Gas")
        
        # Sidebar selection for project stage
        selected = st.sidebar.selectbox("In which stage are you in the project?", ["30", "60", "80"])
        
        if selected == "30":
            st.header("Project Stage: 30% Completed")
            st.write("At this stage, focus on initial assessments and planning.")
            # Add inputs and functionalities specific to 30% stage here
        
            # Customer Details Input
            company_name = st.text_input("Company Name:")
            project_name = st.text_input("Project Name:")
            start_date = st.date_input("Project Start Date:", datetime.date.today())
            budget = st.number_input("Initial Budget ($):", min_value=0.0)
        
        elif selected == "60":
            st.header("Project Stage: 60% Completed")
            st.write("At this stage, progress tracking and risk assessment are crucial.")
            # Add inputs and functionalities specific to 60% stage here
        
            # Customer Details Input
            company_name = st.text_input("Company Name:")
            project_name = st.text_input("Project Name:")
            risk_summary = st.text_area("Risk Summary (if any):")
            actual_expenditure = st.number_input("Actual Expenditure ($):", min_value=0.0)
        
        elif selected == "80":
            st.header("Project Stage: 80% Completed")
            st.write("At this stage, quality control and final assessments are essential.")
            # Add inputs and functionalities specific to 80% stage here
        
            # Customer Details Input
            company_name = st.text_input("Company Name:")
            project_name = st.text_input("Project Name:")
            quality_issues = st.text_area("Quality Control Issues (if any):")
            end_date = st.date_input("Expected Completion Date:", datetime.date.today())
        
        # Common functionality to generate the report
        def generate_report():
            doc = docx.Document()
            doc.add_heading('Project Completion Report', 0)
            doc.add_paragraph(f'Created On: {str(datetime.date.today())}')
            doc.add_paragraph(f'Company: {company_name}')
            doc.add_paragraph(f'Project Name: {project_name}')
            doc.add_paragraph(f'Start Date: {start_date}')
        
            if selected == "30":
                doc.add_paragraph(f'Initial Budget: ${budget:.2f}')
        
            elif selected == "60":
                doc.add_paragraph(f'Actual Expenditure: ${actual_expenditure:.2f}')
                doc.add_paragraph(f'Risk Summary: {risk_summary}')
        
            elif selected == "80":
                doc.add_paragraph(f'Quality Issues: {quality_issues}')
                doc.add_paragraph(f'Expected Completion Date: {end_date}')
        
            # Using the dummy model to add generated content
            generated_content = report_generator_model.predict("")  # Pass relevant input data if needed
            doc.add_paragraph(generated_content)
        
            return doc
        
        # Button to generate report
        if st.button("Generate Report"):
            report_doc = generate_report()
            report_path = 'Project_Completion_Report.docx'
            report_doc.save(report_path)
        
            # Provide download link for the finalized report
            with open(report_path, 'rb') as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()
                st.download_button('Download Report', data=encoded, file_name='Project_Completion_Report.docx')
        
            st.success("Report generated successfully! You can download it using the button above.")

