from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import tempfile
import fitz  # PyMuPDF for handling PDFs
import shutil
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

# Helper function to save the uploaded file
def save_uploaded_file(uploaded_file):
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.filename)
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(uploaded_file.file, temp_file)
    return temp_file_path

# Endpoint to upload and process files
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = save_uploaded_file(file)

    try:
        if file.content_type.startswith("image/"):
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
            result = {"text": extracted_text if extracted_text else "No text found in image."}
        
        elif file.content_type == "application/pdf":
            doc_text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    doc_text += page.get_text("text")
            result = {"text": doc_text if doc_text else "No text found in PDF."}
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        os.remove(file_path)  # Ensure the file is removed after processing

    return JSONResponse(content=result)

# Endpoint to analyze document content
@app.post("/analyze/")
async def analyze_document(file: UploadFile = File(...)):
    file_path = save_uploaded_file(file)
    
    try:
        # Placeholder for analysis result
        doc_intelligence = "Feature extraction and analysis results for uploaded document."
        result = {"analysis": doc_intelligence}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")
    
    finally:
        os.remove(file_path)  # Clean up the temporary file after processing

    return JSONResponse(content=result)
