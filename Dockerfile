# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port specified by Railway
EXPOSE 8000

# Run the application using the dynamically assigned $PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]

