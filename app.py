import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_webrtc import webrtc_streamer, WebRTCConfiguration


# --- Set page configuration ---
st.set_page_config(
    page_title="corsarious",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --- USER AUTHENTICATION ---
names = ["corsarious"]
usernames = ["corsarious"]

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
    names, usernames, hashed_passwords, "corsarious", "corsarious", cookie_expiry_days=30
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
    # Sidebar
    selected = st.sidebar.selectbox("Select Module", ["Doc Intelligence", "Field AI Assistant", "AI Testing"])

        # Split layout into two columns: left for selected content, right for chatbot
    col1, col2 = st.columns([2, 1])

    with col1:

    
        # Main Modules
        if selected == "Doc Intelligence":
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], verbose=True)
            
            # Define the path to the YOLO model file (assuming it's in the same directory as the script)
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            
            # Load the YOLO model
            model = YOLO(model_path)
            
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
            
                # ONNX Symbol Detection (Using the YOLO model)
                st.subheader("Symbol Detection with YOLO (best.pt)")
            
                # Perform inference with the YOLO model
                results = model(img)
            
                # Display the results
                st.subheader("Detection Results:")
                
                # Access bounding boxes, labels, and confidence scores
                for *xyxy, conf, cls in results[0].boxes.data:  # Get bounding boxes and other info
                    label = model.names[int(cls)]
                    x_min, y_min, x_max, y_max = map(int, xyxy)  # Get bounding box coordinates
                    st.write(f"Detected: {label} with confidence {conf:.2f}")
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
                # Display annotated image with YOLO results
                st.image(img, caption="YOLO Annotated Image", use_column_width=True)
            
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
        
        elif selected == "Field AI Assistant":
            import os
            import io
            from gtts import gTTS
            from pydub import AudioSegment
            import speech_recognition as sr
            import streamlit as st
            from streamlit_webrtc import webrtc_streamer, WebRTCConfiguration
            
            # Function to process audio for speech recognition
            def process_audio(audio_file):
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text.lower()
                except sr.UnknownValueError:
                    return "Error: Unable to recognize speech."
                except sr.RequestError as e:
                    return f"Error: {e}"
            
            # Function to speak and save text as audio
            def speak(text, output_file="output.mp3"):
                tts = gTTS(text=text, lang='en')
                tts.save(output_file)
                os.system(f"afplay {output_file}")  # Use 'afplay' for macOS or adjust for your OS
            
            # Function to identify equipment type based on keywords
            def determine_equipment(equipment_name):
                equipment_keywords = {
                    "electric unit heater": ["electric", "unit", "heater"],
                    "air intake": ["air", "intake"],
                    "ridge vent": ["ridge", "vent"],
                    "exhaust air fan": ["exhaust", "fan"]
                }
                for equipment, keywords in equipment_keywords.items():
                    if any(keyword in equipment_name.lower() for keyword in keywords):
                        return equipment
                return None
            
            # Function to capture voice response and save as audio
            def capture_response():
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("Listening...")
                    audio = recognizer.listen(source)
                try:
                    response_text = recognizer.recognize_google(audio)
                    return response_text
                except sr.UnknownValueError:
                    return "Error: Unable to understand."
                except sr.RequestError:
                    return "Error: Service unavailable."
            
            # UI for Streamlit
            st.title("AI Field Assistant")
            
            # Step 1: Record Voice Command
            if st.button("Record Command"):
                response = capture_response()
                st.write("You said:", response)
                if response:
                    equipment = determine_equipment(response)
                    if equipment:
                        st.write(f"Equipment detected: {equipment}")
                        speak(f"{equipment} detected.")
                    else:
                        st.write("No known equipment detected.")
                        speak("I could not identify the equipment.")
            
            # Display help information
            st.write("""
            Instructions:
            - Click on "Record Command" and speak the equipment name.
            - The assistant will respond with detected equipment.
            """)
            
    with col2:
                # --- CHATBOT IN RIGHT COLUMN ---
        st.subheader(f"Hello {name}, I am your upskiller!")
        
        # OpenAI API Key input
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Let's start with the demo. Write 'start' and click enter."}
            ]
        
        # Restart Chat button
        if st.button("Restart Chat"):
            # Clear chat messages to restart the conversation
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Let's start with the demo. Write 'start' and click enter."}
            ]
        
        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Accept user input
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
        
            # Send user message
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
        
            # Define bot response logic
            if prompt.strip().lower() == "start":
                # Start demo process
                msg = "Thank you! Let's add some sample files for demoing purposes. Do you have Orthos or Lidar files?"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                
                # Display "Yes" and "No" buttons
                has_files = st.radio("Do you have Orthos or Lidar files?", ["Yes", "No"])
                
                if has_files == "No":
                    # If customer doesn't have files, display links
                    no_files_response = (
                        "No problem! You can use these sample datasets:\n\n"
                        f"**TIFF**: [SAIT Orthos TIFF file](https://mysait-my.sharepoint.com/:i:/r/personal/rick_duchscher_sait_ca/Documents/UAV%20Projects/SAIT%20OGL%202018/SAIT_OGL_Orthos_2018/R004_C001.tif?csf=1&web=1&e=Q0f8v7)\n"
                        f"**LAS**: [SAIT Lidar LAS file](https://mysait-my.sharepoint.com/:u:/r/personal/rick_duchscher_sait_ca/Documents/UAV%20Projects/SAIT%20OGL%202018/SAIT_OGL_Lidar_2018/Colorized/Sait_CIR_NonGround.las?csf=1&web=1&e=Ky9ftA)"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": no_files_response})
                    st.chat_message("assistant").write(no_files_response)
        
                elif has_files == "Yes":
                    # If customer has files, prompt for the file location link
                    file_location_response = "Great! Please enter the link to where your files are located."
                    st.session_state.messages.append({"role": "assistant", "content": file_location_response})
                    st.chat_message("assistant").write(file_location_response)
        
                    # Display text box for customer to input file location link
                    user_file_link = st.text_input("Enter your file location link here:")
                    if user_file_link:
                        st.session_state.messages.append({"role": "user", "content": f"Customer provided file link: {user_file_link}"})
                        st.chat_message("user").write(f"Customer provided file link: {user_file_link}")
                        
                        # Proceed with using `user_file_link` in your main app functionality as needed
        
            else:
                # Other non-start responses - get response from OpenAI
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
