#!/bin/bash
set -e

echo "=========================================="
echo "🚀 STARTING DEPLOYMENT"
echo "=========================================="

# Setup
echo "📁 Creating directories..."
mkdir -p /app/data/videos /app/data/processed_videos /app/data/video_data
echo "✅ Directories created"

# Run migrations
echo "🗄️  Running database migrations..."
python manage.py migrate
echo "✅ Migrations completed"

# Start Celery worker
echo "=========================================="
echo "🔧 STARTING CELERY WORKER"
echo "=========================================="
celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10 > /tmp/celery.log 2>&1 &
CELERY_PID=$!
echo "📋 Celery PID: $CELERY_PID"

# Give Celery time to initialize
echo "⏳ Waiting for Celery to start..."
sleep 8

# Check if Celery process is running
if ps -p $CELERY_PID > /dev/null 2>&1; then
   echo "✅ CELERY WORKER IS RUNNING (PID: $CELERY_PID)"
   echo "📜 Celery logs:"
   head -n 20 /tmp/celery.log
else
   echo "❌ CELERY WORKER FAILED TO START"
   echo "📜 Error logs:"
   cat /tmp/celery.log
   exit 1
fi

# Start Gunicorn in foreground
echo "=========================================="
echo "🌐 STARTING GUNICORN WEB SERVER"
echo "=========================================="
echo "🔗 Binding to 0.0.0.0:$PORT"
exec gunicorn mysite.wsgi:application --timeout 120 --workers 2 --bind 0.0.0.0:$PORT