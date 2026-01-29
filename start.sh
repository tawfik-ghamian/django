#!/bin/bash
set -e

# Setup
mkdir -p /app/data/videos /app/data/processed_videos /app/data/video_data
python manage.py migrate

# Start Celery
echo "Starting Celery worker..."
celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10 > /tmp/celery.log 2>&1 &
CELERY_PID=$!
echo "Celery started with PID: $CELERY_PID"

# Give Celery time to start
sleep 5

# Check if Celery is running
if ps -p $CELERY_PID > /dev/null; then
   echo "✅ Celery worker is running"
else
   echo "❌ Celery worker failed to start"
   cat /tmp/celery.log
   exit 1
fi

# Start Gunicorn (foreground)
echo "Starting Gunicorn..."
exec gunicorn mysite.wsgi:application --timeout 120 --workers 2 --bind 0.0.0.0:$PORT