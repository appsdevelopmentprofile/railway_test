# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]



# Use a base image with Python 3.12
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . /app

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the application with Streamlit
CMD ["streamlit", "run", "app.py"]


