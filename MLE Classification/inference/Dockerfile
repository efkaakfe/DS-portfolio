# Dockerfile for training
# Start from a base Python image
FROM python:3.11

WORKDIR /app

# Copy data folder to Docker
COPY data_process/ /app/data_process

# Copy training for importing modules
COPY training /app/training

# Copy models folder to Docker
COPY model/ /app/model/

# Copy the code itself
COPY inference /app/inference

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
CMD ["python3", "inference/run.py"]