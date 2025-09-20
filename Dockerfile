FROM python:3.11-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    postgresql-client \
    libpq-dev \
    gcc \
    redis \
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
CMD python -m manage.py migrate \
    gunicorn mysite.wsgi:application --bind 0.0.0.0:8000 --log-level info   