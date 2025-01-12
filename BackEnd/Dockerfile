# Use Python 3.9 slim base image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies (if needed, e.g., for numpy or PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version (optimized for production) with compatible NumPy
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --default-timeout=100

# Install the remaining Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port that the Flask app runs on
EXPOSE 8080

# Set environment variables for production
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the Flask app
CMD ["python", "app.py"]
