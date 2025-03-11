# Use the official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the content of the current directory into the container
COPY . .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (this is the default port Cloud Run uses)
EXPOSE 8080

# Set environment variable to specify the Flask app
ENV FLASK_APP=app.py

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
