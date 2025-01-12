# Use a lightweight Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch from the custom index
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies from default PyPI
RUN pip install --no-cache-dir flask pandas scikit-learn && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
