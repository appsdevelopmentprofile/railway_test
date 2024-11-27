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
            import streamlit as st
            from gtts import gTTS
            import pyaudio

            # STEP 1: Diagnose and recognition
            # Function to speak text using gTTS
            def speak(text):
                tts = gTTS(text=text, lang='en')
                tts.save("output.mp3")
                os.system("afplay output.mp3")  # Use 'afplay' for macOS or modify for other platforms
            
            # Equipment categories and associated keywords for identification
            equipment_keywords = {
                "electric unit heater": ["electric", "unit", "heater"],
                "air intake": ["air", "intake"],
                "ridge vent": ["ridge", "vent"],
                "exhaust air fan": ["exhaust", "fan"]
            }
            
            # Function to match the response to the equipment based on keywords
            def match_equipment(response):
                response = response.lower()
                for equipment, keywords in equipment_keywords.items():
                    if any(keyword in response for keyword in keywords):
                        return equipment
                return None
            
            # Main function for conducting the field engineer questionnaire
            def field_engineer_questionnaire():
                st.title("Field Engineer Questionnaire")
            
                # Initial identification questions
                identification_questions = [
                    "What is the equipment name?",
                    "What is the equipment ID?",
                    "What is the current status of the equipment? (Working or Faulty)"
                ]
            
                responses = {}
            
                # Ask and speak the questions, record responses using the microphone
                for question in identification_questions:
                    speak(question)
                    st.write(question)
                    response = st.text_input("Your answer here:")
            
                    if st.button("Record"):
                        responses[question] = response
                        st.write(f"Recorded: {response}")
            
                # Match the response to the equipment
                equipment_name = responses.get("What is the equipment name?", "").lower()
                matched_equipment = match_equipment(equipment_name)
            
                if matched_equipment:
                    st.write(f"Identified equipment: {matched_equipment}")
                    speak(f"The identified equipment is {matched_equipment}.")
                    
                    # Here we can proceed to extract the PDF follow-up questions in another stage
                    st.write("Proceeding to the next stage for follow-up questions...")
                else:
                    st.write("Equipment type not recognized or supported.")
                    speak("Equipment type not recognized or supported.")
            
                st.write("All responses recorded:", responses)
                return responses
            
            # Run the questionnaire in the Streamlit app
            if __name__ == "__main__":
                field_engineer_questionnaire()





            

            
            # Step 2: QUESTIONAIRE - RECORDING

            import os
            import streamlit as st
            import speech_recognition as sr
            from playsound import playsound
            
            # Function to find the MP3 file based on the equipment name
            def find_mp3_file(equipment_name, base_path="local_dataset/mp3"):
                # Construct the file name based on the equipment type
                mp3_filename = f"{equipment_name.replace(' ', '_')}.mp3"
                mp3_path = os.path.join(base_path, mp3_filename)
            
                # Check if the file exists
                if os.path.isfile(mp3_path):
                    return mp3_path
                else:
                    st.write("No MP3 file found for the identified equipment.")
                    return None
            
            # Function to play the MP3 file
            def play_mp3(mp3_path):
                if mp3_path:
                    st.write(f"Playing the follow-up questions for {equipment_name}.")
                    playsound(mp3_path)
                else:
                    st.write("No audio to play.")
            
            # Function to record speech from the engineer and extract text
            def record_and_extract_text():
                recognizer = sr.Recognizer()
                mic = sr.Microphone()
            
                st.write("Please respond to the questions after listening to the audio. Say 'COMPLETED' when you're done.")
            
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    st.write("Recording...")
                    audio = recognizer.listen(source)
                    st.write("Recording stopped.")
            
                # Process the recorded audio
                try:
                    text = recognizer.recognize_google(audio)
                    st.write(f"Recorded response: {text}")
                    if "completed" in text.lower():
                        st.write("Recording completed successfully.")
                        return text
                    else:
                        st.write("Engineer did not say 'COMPLETED'. Recording might not have finished.")
                        return text
                except sr.UnknownValueError:
                    st.write("Could not understand the audio.")
                    return "unrecognized"
                except sr.RequestError as e:
                    st.write(f"Error: {e}")
                    return f"error: {e}"
            
            # Main function for the follow-up stage
            def follow_up_questions_stage(equipment_name):
                # Find the corresponding MP3 file
                mp3_path = find_mp3_file(equipment_name)
            
                # Play the MP3 if it exists
                play_mp3(mp3_path)
            
                # Record and extract the response text
                response_text = record_and_extract_text()
            
                return response_text
            
            # Example usage in a Streamlit app
            if __name__ == "__main__":
                st.title("Follow-Up Questions Stage")
                
                # Simulate the equipment type (this would come from the previous stage)
                equipment_name = "air intake"  # Example matched equipment type
            
                # Run the follow-up stage
                response = follow_up_questions_stage(equipment_name)
                st.write(f"Final recorded response: {response}")


            
            # Step 3: Record Responses to fill the checklist taken from the data Lake
            import streamlit as st
            import speech_recognition as sr
            from pydub import AudioSegment
            import io
            from docx import Document
            
            # Streamlit header for UI
            st.header("Step 4: Record Responses")
            
            # Function to recognize speech from an MP3 file
            def recognize_speech_from_mp3(mp3_file_path):
                recognizer = sr.Recognizer()
                audio = AudioSegment.from_mp3(mp3_file_path)
                
                # Export audio data to a bytes buffer in WAV format
                audio_data = io.BytesIO()
                audio.export(audio_data, format="wav")
                audio_data.seek(0)
            
                with sr.AudioFile(audio_data) as source:
                    audio_recorded = recognizer.record(source)
                    try:
                        # Return the recognized text in lowercase
                        return recognizer.recognize_google(audio_recorded).lower()
                    except sr.UnknownValueError:
                        return "unrecognized"
                    except sr.RequestError as e:
                        return f"error: {e}"
            
            # Function to analyze response for 'yes' or 'no' based on keywords
            def analyze_response(response_text):
                yes_keywords = ["yes", "yeah", "yep", "affirmative", "sure"]
                no_keywords = ["no", "nope", "nah", "negative", "not"]
            
                response_text = response_text.lower()
            
                if any(yes_keyword in response_text for yes_keyword in yes_keywords):
                    return "yes"
                elif any(no_keyword in response_text for no_keyword in no_keywords):
                    return "no"
                else:
                    return "unknown"
            
            # Function to fill out the checklist based on responses
            def fill_checklist(checklist_template, responses):
                for item, response in responses.items():
                    for paragraph in checklist_template.paragraphs:
                        if item in paragraph.text:
                            # Update the checkbox based on response
                            if response == "yes":
                                paragraph.text = paragraph.text.replace('☐', '☑')
                            elif response == "no":
                                paragraph.text = paragraph.text.replace('☐', '☐')
                return checklist_template
            
            # Function to save the filled checklist to a document
            def save_filled_checklist(checklist_template, output_filename="filled_checklist.docx"):
                checklist_template.save(output_filename)
                print(f"Document '{output_filename}' created successfully.")
                st.success(f"Document '{output_filename}' created successfully.")
            
            # Main function to orchestrate the process
            def main():
                # Specify the MP3 file and checklist file paths
                mp3_file_path = "/content/drive/MyDrive/engineer_equipment.mp3"  # Update as needed
                checklist_filename = "/content/drive/MyDrive/predefined_checklist.docx"  # Update as needed
            
                # Step 1: Extract text from the MP3 file
                print("Extracting text from MP3 file...")
                response_text = recognize_speech_from_mp3(mp3_file_path)
            
                # Step 2: Analyze the response
                if response_text and response_text != "unrecognized":
                    print("Recognized response:", response_text)
                    response = analyze_response(response_text)
                    print("Analyzed response:", response)
            
                    # Step 3: Load the checklist and update it based on the response
                    checklist_template = Document(checklist_filename)
                    responses = {
                        "Is the unit heater operational?": response,
                        "Are the power supply connections intact?": response,
                        "Is the heating element functioning?": response
                    }
            
                    # Step 4: Fill and save the checklist
                    filled_checklist = fill_checklist(checklist_template, responses)
                    save_filled_checklist(filled_checklist)
            
                else:
                    print("Speech recognition failed or no speech was recognized.")
                    st.error("Speech recognition failed or no speech was recognized.")
            
            # Run the main function if this script is executed directly
            if __name__ == "__main__":
                main()


            
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
