# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddleOCR
# FIXED: Swapped deprecated libgl1-mesa-glx for modern libgl1 and libglx-mesa0 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
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
