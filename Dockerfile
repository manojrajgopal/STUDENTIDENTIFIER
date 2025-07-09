# Use a base image with Python
FROM python:3.9-slim

# Install system dependencies (for opencv, deepface)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Start the Flask app using Gunicorn (better than raw python for production)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
