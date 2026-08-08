# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddleOCR
# FIXED: Changed --no-install-libraries to the correct --no-install-recommends flag
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    glib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 10000 for Render compatibility
EXPOSE 10000

# Start the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
