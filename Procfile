web: python manage.py migrate && gunicorn mysite.wsgi:application --timeout 120 --workers 2 --bind 0.0.0.0:$PORT
worker: celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10
