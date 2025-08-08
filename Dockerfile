# Use an official Python runtime as a parent image
# python:3.12-slim is a good choice for a smaller image size
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
# This includes all your .py files and the 'model' directory
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Run the command to start the Uvicorn server when the container launches
# This command runs your webhook_server.py file
CMD ["uvicorn", "webhook_server:app", "--host", "0.0.0.0", "--port", "8000"]