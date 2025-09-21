web: gunicorn mysite.wsgi:application --timeout 60 --workers 3 --max-requests 1000
worker: celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10