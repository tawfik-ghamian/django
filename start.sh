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

# Start Celery worker in background with proper logging
echo "=========================================="
echo "🔧 STARTING CELERY WORKER"
echo "=========================================="
echo "📋 Redis URL: ${REDIS_URL:0:50}..."

# Start celery with nohup to ensure it keeps running
nohup celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10 >> /tmp/celery.log 2>&1 &
CELERY_PID=$!
echo "📋 Celery PID: $CELERY_PID"

# Give Celery more time to fully initialize
echo "⏳ Waiting for Celery to initialize..."
sleep 10

# Check if Celery is still running
if ps -p $CELERY_PID > /dev/null 2>&1; then
   echo "✅ CELERY WORKER IS RUNNING (PID: $CELERY_PID)"
   echo "📜 First 50 lines of Celery logs:"
   head -n 50 /tmp/celery.log
   echo "=========================================="
else
   echo "❌ CELERY WORKER CRASHED"
   echo "📜 Full error logs:"
   cat /tmp/celery.log
   exit 1
fi

# Tail celery logs in background for monitoring
tail -f /tmp/celery.log &

# Start Gunicorn in foreground
echo "=========================================="
echo "🌐 STARTING GUNICORN WEB SERVER"
echo "=========================================="
echo "🔗 Binding to 0.0.0.0:$PORT"
exec gunicorn mysite.wsgi:application --timeout 120 --workers 2 --bind 0.0.0.0:$PORT