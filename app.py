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
from transformers import BertTokenizerFast, TFBertForSequenceClassification, pipeline
import pyvista as pv
import laspy
import rasterio
from rasterio.plot import show
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
import io

# --- Set page configuration ---
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
try:
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)
except FileNotFoundError:
    st.error("Password file not found.")
except Exception as e:
    st.error(f"An error occurred while loading passwords: {e}")

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
    # ---- LOGOUT AND GREETING ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    
        selected = option_menu(
            'RFO Central Application & AI Modules',  # Combined title
            [
                "Doc Intelligence",
                "Predictive Analytics for Operational Planning",
                "Real-Time Fault Monitoring",
                "Project Completion Reporting",
                "AI-based GIS From Images to GeoTiff",
                "AI + BIM - from BIM to 4D schedule",
                "3D Point Clouds – AI for Digital Twins",
                "AI-Enhanced Drone Mapping - LiDAR"
            ],
            menu_icon='layers',  # Single menu icon
            icons=['file-earmark-text', 'graph-up', 'exclamation-circle', 'clipboard-check', 'map', 'calendar', 'cube', 'airplane'],
            default_index=0
        )

        # Display content based on the selected module
        st.header(selected)
        st.write(f"This is the content for {selected} module.")  # Replace with actual content
    # Split layout into two columns: left for menu, right for chatbot
    col1, col2 = st.columns([2, 1])

    with col1:
        # Sidebar Menu

        # Doc Intelligence Section
        if selected == 'Doc Intelligence':
            # Load the model and tokenizer using TensorFlow
            model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            
            # Initialize a pipeline for prediction
            nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="tf")
    
            # Function to process the uploaded files
            def doc_intelligence():
                # Set up the page for document intelligence
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
                                
                                # Make prediction using the Hugging Face model
                                if extracted_text:
                                    prediction = nlp_pipeline(extracted_text)
                                    st.subheader("Prediction Result:")
                                    st.write(prediction)
            
                        # Handle PDF files
                        elif uploaded_file.type == "application/pdf":
                            doc_text = ""
                            pdf = fitz.open(temp_file_path)
                            for page_num in range(pdf.page_count):
                                page = pdf[page_num]
                                doc_text += page.get_text("text")
                            st.subheader("Extracted Text from PDF:")
                            st.write(doc_text if doc_text else "No text found in PDF.")
                            
                            # Make prediction using the Hugging Face model
                            if doc_text:
                                prediction = nlp_pipeline(doc_text)
                                st.subheader("Prediction Result:")
                                st.write(prediction)
                            pdf.close()
            
                        # Remove temporary file
                        os.remove(temp_file_path)
            
                # Document Analysis Button
                if st.button('Analyze Document Content'):
                    st.success("Feature extraction and analysis results will be displayed here.")
    
    
     ## Module 2 - Predictive Analytics for Operational Planning
        
        
        # Bring up the page when it is selected on the side bar
        elif selected == 'Predictive Analytics for Operational Planning':
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
        elif selected == "Real-Time Fault Monitoring":
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
        elif selected == "Project Completion Reporting":
        
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
    
    # Function to load and process GeoTIFF file
    # Module 1: AI-based GIS - From Images to GeoTiff
        elif selected == "AI-based GIS From Images to GeoTiff":
            st.header("AI-based GIS - GeoTiff Segmentation")
            
            # Function to load and process GeoTIFF file
            def load_geotiff(uploaded_file):
                with rasterio.open(uploaded_file) as src:
                    image = src.read([1, 2, 3])  # Read RGB bands
                    image = np.moveaxis(image, 0, -1)  # Reorder axes to (height, width, channels)
                    image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure image is within RGB range
                return image
            
            # Function to segment the image using DeepLabV3 from TensorFlow Hub
            def segment_image(image):
                # Load DeepLabV3 pre-trained model from TensorFlow Hub
                deeplab_model = hub.load("https://tfhub.dev/tensorflow/deeplabv3/1")
            
                # Preprocess image for DeepLabV3 input
                img = Image.fromarray(image)
                img = img.resize((256, 256))  # Resize to model input size
                img = img_to_array(img)  # Convert to array
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                img = img / 255.0  # Normalize image
            
                # Perform segmentation
                pred = deeplab_model(img)  # Predict segmentation mask
                pred = np.argmax(pred, axis=-1)[0]  # Get the most probable class for each pixel
                return pred
            
            # Streamlit interface
            def app():
                st.title("AI-Based GIS - GeoTIFF Raster Segmentation")
            
                st.write("""
                    Upload a GeoTIFF orthophoto to visualize and apply AI-based segmentation.
                    This tool allows you to apply DeepLabV3 segmentation on your geospatial data.
                    Please upload a GeoTIFF file and click 'Segment Image' to view the results.
                """)
            
                # File upload with a progress bar
                uploaded_file = st.file_uploader("Choose a GeoTIFF file", type=["tif", "tiff"])
                
                if uploaded_file is not None:
                    # Display the uploaded GeoTIFF image
                    st.subheader("Uploaded GeoTIFF Image")
                    image = load_geotiff(uploaded_file)
                    st.image(image, caption="Uploaded GeoTIFF Image", use_column_width=True)
            
                    # Display a processing message
                    st.info("The image is being processed. This might take a few moments depending on the file size.")
            
                    # Progress bar for the segmentation process
                    with st.spinner("Processing..."):
                        st.progress(0)  # Initial progress (0%)
                        # Apply segmentation after a slight delay
                        segmented_image = segment_image(image)
                        for i in range(1, 101, 20):
                            st.progress(i)  # Update progress bar
                        st.success("Segmentation completed!")
            
                    # Display the segmented image result
                    st.subheader("Segmented Image")
                    st.image(segmented_image, caption="Segmented Image", use_column_width=True, clamp=True)
            
                    # Additional information about the segmentation process
                    st.write("""
                        The segmentation process has classified different regions in the orthophoto.
                        You can now analyze the segmented image for various features or further refine the analysis.
                    """)
            
            if __name__ == "__main__":
                app()
        
        
        # Module 2: AI + BIM - From BIM to 4D Schedule
        elif selected == "AI + BIM - from BIM to 4D schedule":
            st.header("AI + BIM - 4D Schedule: Classify and Visualize 3D Point Clouds")
        
            # Function to load and process 3D Point Cloud (PLY, XYZ, etc.)
            def load_point_cloud(uploaded_file):
                if uploaded_file is not None:
                    file_path = '/tmp/uploaded_point_cloud.ply'
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    point_cloud = pv.read(file_path)
                    return point_cloud
                return None
        
            # Function to classify point cloud using a pre-trained PointNet model
            def classify_point_cloud(pcd):
                # Load PointNet pre-trained model from TensorFlow Hub or a custom model
                model_url = "https://tfhub.dev/google/pointnet/1"
                pointnet_model = hub.load(model_url)
                
                # Convert point cloud to numpy array
                points = np.asarray(pcd.points)
                
                # PointNet input requires (N, 3) point cloud array
                points = np.expand_dims(points, axis=0)  # Adding batch dimension
                
                # Preprocess points (normalize or other steps as per model requirements)
                points = points / np.max(np.abs(points), axis=1, keepdims=True)  # Normalize
                
                # Run the PointNet model
                predictions = pointnet_model(points)
                return predictions
        
            # Function to visualize point cloud using PyVista
            def visualize_point_cloud(pcd):
                # Create a PyVista plotter object
                plotter = pv.Plotter()
                plotter.add_mesh(pcd, color="lightblue", show_edges=True)
                plotter.add_axes()
                plotter.show()
        
            # Streamlit interface
            def app():
                st.title("AI + BIM - 4D Schedule: Classify and Visualize 3D Point Clouds")
            
                st.write("""
                    Upload a 3D point cloud for BIM classification and visualization. 
                    You can classify the point cloud using a PointNet model and see the result in 3D.
                """)
        
                # Upload file with point cloud
                uploaded_file = st.file_uploader("Choose a 3D Point Cloud file (PLY, XYZ, etc.)", type=["ply", "xyz", "pcd"])
        
                if uploaded_file is not None:
                    # Load and display the 3D point cloud
                    pcd = load_point_cloud(uploaded_file)
                    st.write("Point Cloud Loaded Successfully")
                    st.write(f"Point Cloud with {len(pcd.points)} points loaded.")
            
                    # Display point cloud in 3D using PyVista
                    st.subheader("3D Point Cloud Visualization")
                    st.text("This will open a 3D view of your point cloud (use mouse to rotate).")
                    visualize_point_cloud(pcd)
                        
                    # Button to classify the point cloud
                    if st.button("Classify Point Cloud"):
                        with st.spinner("Classifying..."):
                            predictions = classify_point_cloud(pcd)
                            st.success("Classification completed!")
            
                        # Display classification results
                        st.subheader("Classification Results")
                        st.write(predictions)  # Assuming classification provides some kind of result
                        st.text("Classified labels for each point (or overall classification).")
            
                    # Option for 4D scheduling (e.g., tasks for BIM)
                    st.subheader("4D Scheduling")
                    st.write("""
                        The following section allows you to add tasks for BIM based on the 3D point cloud.
                        You can define tasks like 'Modeling,' 'Construction,' etc., and associate them with time intervals.
                    """)
                    
                    task_name = st.text_input("Task Name", "Modeling")
                    task_start = st.date_input("Start Date")
                    task_end = st.date_input("End Date")
                    
                    if st.button("Add Task"):
                        st.write(f"Task '{task_name}' scheduled from {task_start} to {task_end}.")
            
            if __name__ == "__main__":
                app()
    
    
    # Module 3: 3D Point Clouds – AI for Digital Twins
    elif selected == "3D Point Clouds – AI for Digital Twins":
        st.header("AI-based Surveying Tool for Digital Twins")
    
        # Sidebar for uploading point cloud data
        st.subheader("Upload Point Cloud Data")
        uploaded_file = st.file_uploader("Upload a .las, .ply, or .pcd file", type=["las", "ply", "pcd"])
        
        # Option for AI Model Selection
        model_select = st.selectbox(
            "Choose AI Model for Point Cloud Classification",
            ["PointCNN", "DGCNN", "SCAN"]
        )
        
        # Function to process point cloud and visualize using PyVista
        def visualize_point_cloud(coords):
            st.write("### Visualizing Point Cloud")
            cloud = pv.PolyData(coords)
            plotter = pv.Plotter()
            plotter.add_mesh(cloud, color="cyan", point_size=5)
            plotter.set_background("white")
            plotter.show()
    
        # Option to upload point cloud data
        if uploaded_file is not None:
            coords = np.loadtxt(uploaded_file)  # Load the data
            visualize_point_cloud(coords)
            
        st.write("AI-based Digital Twin operations")
        st.text("You can classify and extract features from point clouds here.")
        
        
    # Module 4: AI-Enhanced Drone Mapping - LiDAR
    elif selected == "AI-Enhanced Drone Mapping - LiDAR":
        st.header("AI for LiDAR-based Drone Mapping")
    
        st.write("""
            Upload a LiDAR point cloud file to classify and process drone mapping data. 
            This tool leverages AI-based models to classify ground and non-ground points.
        """)
        uploaded_file = st.file_uploader("Upload LiDAR Point Cloud", type=["las", "laz", "ply"])
    
        if uploaded_file is not None:
            coords = np.loadtxt(uploaded_file)  # Load the point cloud data
            visualize_point_cloud(coords)  # Visualize point cloud with PyVista
    
        st.write("Further process LiDAR data for classification or other tasks.")


        

    with col2:
        # --- CHATBOT IN RIGHT COLUMN ---
        st.subheader("💬 Chatbot")
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")

        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.write("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
        st.write("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # User input
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()

            # Send user message
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages
            )
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)


