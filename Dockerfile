# Use a Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install system dependencies (if needed for specific libraries)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Set the default command to run your app (adjust if needed)
CMD ["streamlit", "run", "app.py"]
