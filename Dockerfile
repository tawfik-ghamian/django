FROM python:3.11-slim

# Install system dependencies required for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    ffmpeg \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    gfortran \
    wget \
    pkg-config \
    postgresql-client \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories for media files
RUN mkdir -p /app/media/videos /app/media/processed_videos /app/media/video_data

# Expose port
EXPOSE $PORT

# Run migrations, collect static files, and start the server
CMD python manage.py migrate && \
    python manage.py collectstatic --noinput && \
    gunicorn mysite.wsgi:application --bind 0.0.0.0:$PORT