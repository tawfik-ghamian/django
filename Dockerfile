# FROM python:3.11-slim

# # Install only essential system dependencies
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libgl1 \
#     libgomp1 \
#     ffmpeg \
#     postgresql-client \
#     libpq-dev \
#     gcc \
#     redis \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Create necessary directories for media files
# RUN mkdir -p /app/media/videos /app/media/processed_videos /app/media/video_data /app/logs

# # Run migrations, collect static files, and start the server
# CMD python manage.py migrate 

# # Expose port
# EXPOSE $PORT


FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    postgresql-client \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/media/videos \
    /app/media/processed_videos \
    /app/media/video_data \
    /app/logs \
    /app/staticfiles

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Expose port (Railway will set this)
EXPOSE 8000

# Note: CMD is set by railway.json for each service