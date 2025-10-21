web:
worker: python manage.py migrate && celery -A mysite worker --loglevel=info --concurrency=2 --max-tasks-per-child=10