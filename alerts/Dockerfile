# Use an official Python runtime as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY src/ /app/src/
COPY logging.yaml /app/logging.yaml

# Set the command to run when the container starts
CMD [ "python", "/app/src/main.py" ]

