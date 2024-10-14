# Use the official Python image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /Classification_Deep_Learning_project

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Specify the command to run your application
CMD ["python3", "app.py"]