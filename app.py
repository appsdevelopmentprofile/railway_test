import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import pytesseract
import tempfile

st.set_page_config(page_title="applus", layout="wide", page_icon="🧑‍⚕️")

# User Authentication
# [Implement fixed authentication mechanism]

# Sidebar
selected = st.sidebar.selectbox("Select Module", ["Doc Intelligence", "Field AI Assistant", "AI Testing"])

# Main Modules
if selected == "Doc Intelligence":

    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    
    # Streamlit frontend
    st.title("Instrumentation Plan Processing")
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
        # Display the uploaded image
        st.subheader("Uploaded Image:")
        st.image(img, channels="BGR")
    
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
        # Step 3: Use edge detection (Canny)
        edges = cv2.Canny(blurred, 50, 150)
    
        # Step 4: Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
    
        # Step 5: Detect contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Create a mask to remove the frame and unwanted lines
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
        # Step 6: Remove text and continuous lines by filtering contour size
        instrument_shapes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
    
            # Filter based on size (ignore very large/small contours)
            if 50 < w < 500 and 50 < h < 500:  # Adjust these thresholds as needed
                instrument_shapes.append((x, y, w, h))
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        # Display the processed image
        st.subheader("Processed Image with Detected Shapes:")
        st.image(img, channels="BGR")
    
        # Display individual detected shapes
        st.subheader("Extracted Instrument Shapes:")
        if instrument_shapes:
            cols = st.columns(3)  # Adjust number of columns for layout
            for i, (x, y, w, h) in enumerate(instrument_shapes):
                cropped_shape = img[y:y + h, x:x + w]
                with cols[i % 3]:
                    st.image(cropped_shape, caption=f"Shape {i + 1}")
        else:
            st.write("No instrument shapes detected.")



"""
elif selected == "Field AI Assistant":
    st.title("AI Field Assistant")
    # [Speech-to-Text and Audio Processing Code Here]

# Import required modules
    import streamlit as st
    import speech_recognition as sr
    from gtts import gTTS
    import os
    from pydub import AudioSegment
    import tempfile

    import streamlit as st
    from st_custom_components import st_audio_recorder
    import speech_recognition as sr
    from pydub import AudioSegment
    import os
    
    # Title
    st.title("Engineer Voice Input Recorder")
    
    # Display instruction
    st.write("Click the button below to start recording your answer.")
    
    # Record audio
    audio_data = st_audio_recorder(key="audio_recorder")
    
    if audio_data:
        st.success("Recording captured successfully!")
        
        # Save the audio file locally as a WAV (you can customize format)
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data)
    
        # Initialize speech recognition
        recognizer = sr.Recognizer()
        
        # Load the WAV file
        with sr.AudioFile("recorded_audio.wav") as source:
            st.write("Processing the audio...")
            audio = recognizer.record(source)
        
        # Perform speech-to-text
        try:
            transcription = recognizer.recognize_google(audio)
            st.write("Transcription of your answer:")
            st.text_area("Engineer Response", value=transcription, height=200)
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please ensure the recording is clear.")
        except sr.RequestError as e:
            st.error(f"Speech Recognition service error: {e}")
        
        # Cleanup temporary file
        os.remove("recorded_audio.wav")


    
    # Define function to recognize speech from audio file
    def recognize_speech_from_file(audio_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Could not understand the audio."
            except sr.RequestError as e:
                return f"Error: {e}"
    
    # Function to save audio responses
    def save_audio_response(audio_file, filename="response.wav"):
        audio = AudioSegment.from_file(audio_file)
        file_path = os.path.join(tempfile.gettempdir(), filename)
        audio.export(file_path, format="wav")
        return file_path
    
    # Function to play text as audio
    def speak_text(text):
        tts = gTTS(text=text, lang='en')
        audio_path = os.path.join(tempfile.gettempdir(), "question.mp3")
        tts.save(audio_path)
        st.audio(audio_path, format="audio/mp3")
    
    # Streamlit frontend
    st.title("Field Engineer Diagnosis System")
    
    # STEP 1: Initial Questions
    st.header("Step 1: Equipment Details")
    equipment_name = st.text_input("What is the equipment name?")
    equipment_id = st.text_input("What is the equipment ID?")
    equipment_status = st.selectbox("What is the current status of the equipment?", ["Working", "Faulty"])
    
    if st.button("Submit Equipment Details"):
        st.write(f"Equipment Name: {equipment_name}")
        st.write(f"Equipment ID: {equipment_id}")
        st.write(f"Equipment Status: {equipment_status}")
    
        # Check equipment type
        if "heater" in equipment_name.lower():
            selected_equipment = "electric_unit_heater"
        elif "intake" in equipment_name.lower():
            selected_equipment = "air_intake"
        elif "vent" in equipment_name.lower():
            selected_equipment = "ridge_vent"
        elif "fan" in equipment_name.lower():
            selected_equipment = "exhaust_air_fan"
        else:
            selected_equipment = None
    
        if selected_equipment:
            st.success(f"Identified Equipment: {selected_equipment.replace('_', ' ').title()}")
        else:
            st.error("Equipment not recognized. Please try again.")
    
    # STEP 2: Play Questions for Identified Equipment
    if selected_equipment:
        st.header(f"Step 2: Questionnaire for {selected_equipment.replace('_', ' ').title()}")
        questions = {
            "electric_unit_heater": [
                "Is the unit heater operational?",
                "Are the power supply connections intact?",
                "Is the thermostat working accurately?"
            ],
            "air_intake": [
                "Is the air intake free from obstructions?",
                "Are the air filters clean and in good condition?",
                "Is the airflow consistent?"
            ],
            "ridge_vent": [
                "Are the ridge vents free of debris?",
                "Are there any visible damages?",
                "Is there any sign of water intrusion?"
            ],
            "exhaust_air_fan": [
                "Is the fan operational?",
                "Are the blades clean?",
                "Are the motor and bearings lubricated?"
            ]
        }
    
        for question in questions[selected_equipment]:
            st.write(question)
            speak_text(question)
    
    # STEP 3: Capture Engineer's Responses
    st.header("Step 3: Record Engineer's Responses")
    audio_file = st.file_uploader("Upload your response audio file (wav format)", type=["wav"])
    
    if audio_file:
        file_path = save_audio_response(audio_file)
        st.success("Audio uploaded successfully. Processing transcription...")
        transcription = recognize_speech_from_file(file_path)
        st.write("Transcription:", transcription)
    
    # STEP 4: Match Responses with Checklist
    st.header("Step 4: Match Responses with Checklist")
    checklist = {
        "electric_unit_heater": ["operational", "power supply intact", "thermostat working"],
        "air_intake": ["free from obstructions", "filters clean", "airflow consistent"],
        "ridge_vent": ["free of debris", "no visible damages", "no water intrusion"],
        "exhaust_air_fan": ["fan operational", "blades clean", "bearings lubricated"]
    }
    
    if selected_equipment and transcription:
        st.subheader("Checklist Matching Results")
        for item in checklist[selected_equipment]:
            if item.lower() in transcription.lower():
                st.success(f"Matched: {item}")
            else:
                st.warning(f"Missing: {item}")

"""
elif selected == "AI Testing":
    st.title("Automated AI Testing")
    # [Selenium or Test Automation Code Here]
