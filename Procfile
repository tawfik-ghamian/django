web: gunicorn mysite.wsgi:application --timeout 900 --workers 2 --max-requests 100 --preload
worker: celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10